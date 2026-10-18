"""Verify sealed input bytes and make an independent, read-only input copy.

No native imports and no environment/truth handles. The four dependency aliases
are SHA256 of canonical {relative_path: verified_sha256} maps: perception covers
all base_model files; preprocessing covers its processor/normalizer files;
config covers evaluation/control profiles and all source configs; control covers
ALL frozen sources (conservatively including planners, environment and assets).
Individual file hashes are retained verbatim for native source-origin checks.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys

from .artifacts import BudgetExceeded, GIB, safe_path
from .config import FrozenConfig
from .contracts import InputIdentity

SEALED_R8_MANIFEST_HASH = "59d9f08b19a0784b5668dad2676c25c9cd29c1654617fea0870b2e61a15de59a"


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    fd = os.open(safe_path(path), os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("invalid manifest path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("manifest path traversal/noncanonical path")
    return name


def _manifest(root):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate manifest key")
            result[key] = value
        return result
    raw = json.loads((root / "manifest.json").read_bytes(), object_pairs_hook=unique)
    if raw.get("schema") != 1 or raw.get("base_step") != 100000 or raw.get("model") != "base_model":
        raise ValueError("unsupported frozen manifest")
    entries = {}
    for group, prefix in (("files", ""), ("sources", "source/")):
        if not isinstance(raw.get(group), dict) or not raw[group]:
            raise ValueError("missing manifest file maps")
        for name, digest in raw[group].items():
            name = prefix + _relative(name)
            if name in entries or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("invalid/duplicate manifest digest")
            try:
                bytes.fromhex(digest)
            except ValueError as exc:
                raise ValueError("invalid manifest digest") from exc
            entries[name] = digest
    return entries


def _tree_files(root):
    files, pending = set(), [root]
    while pending:
        parent = pending.pop()
        for path in parent.iterdir():
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError(f"symlink in input tree: {path}")
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode):
                files.add(path.relative_to(root).as_posix())
            else:
                raise ValueError(f"nonregular input: {path}")
    return files


def verify_inputs(source, *, expected_model_hash=FrozenConfig().model_sha256,
                  expected_manifest_hash=SEALED_R8_MANIFEST_HASH,
                  expected_entries=None, allow_unlisted=True) -> InputIdentity:
    root = safe_path(source)
    files = _tree_files(root)  # Before reading manifest: reject manifest symlinks too.
    entries = _manifest(root)
    manifest_hash = file_hash(root / "manifest.json")
    if manifest_hash != expected_manifest_hash:
        raise ValueError("sealed manifest identity mismatch")
    sealed = set(entries) | {"manifest.json"}
    if not sealed <= files or (not allow_unlisted and files != sealed):
        raise ValueError("unlisted or missing sealed input files")
    if expected_entries is not None and len(entries) != expected_entries:
        raise ValueError("unexpected sealed manifest item count")
    for name, expected in entries.items():
        if file_hash(root / name) != expected:
            raise ValueError(f"input hash mismatch: {name}")
    model_hash = entries.get("base_model/model.safetensors")
    if model_hash != expected_model_hash:
        raise ValueError("frozen ACT model hash mismatch")
    groups = {
        "perception": {p: h for p, h in entries.items() if p.startswith("base_model/")},
        "preprocessing": {p: h for p, h in entries.items() if p.startswith("base_model/") and ("processor" in p or "normalizer" in p)},
        "config": {p: h for p, h in entries.items() if p in {"evaluation_config.yaml", "control_profile.json", "base_model/config.json"} or p.startswith("source/configs/")},
        "control": {p: h for p, h in entries.items() if p.startswith("source/")},
    }
    if any(not values for values in groups.values()):
        raise ValueError("missing native dependency coverage")
    hashes = dict(entries, **{name: digest_json(values) for name, values in groups.items()})
    return InputIdentity(str(root), manifest_hash, model_hash, hashes)


def snapshot_inputs(source, destination, budget, *,
                    expected_model_hash=FrozenConfig().model_sha256,
                    expected_manifest_hash=SEALED_R8_MANIFEST_HASH,
                    expected_entries=None) -> InputIdentity:
    source, destination = safe_path(source), budget.ensure_owned(destination)
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError("snapshot destination must be a new independent path")
    budget.admit().require()  # Before even hashing potentially large source files.
    expected = dict(expected_model_hash=expected_model_hash, expected_manifest_hash=expected_manifest_hash,
                    expected_entries=expected_entries)
    before = verify_inputs(source, **expected)
    files = sorted(set(_manifest(source)) | {"manifest.json"})
    # Unsealed source documents/tools/locks are not dependencies and are NEVER
    # copied or imported. Still verify their bytes remain unchanged by our copy.
    extras_before = {p: file_hash(source / p) for p in _tree_files(source) - set(files)}
    size = sum((source / name).stat().st_size for name in files)
    # Conservative destination plus temporary-copy headroom; no hardlinks.
    with budget.reserve(2 * size):
        destination.mkdir(parents=True)
        for name in files:
            src, dst = safe_path(source / name), safe_path(destination / name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src_fd = os.open(src, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(src_fd, "rb") as reader, dst.open("xb") as writer:
                for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                    if budget.disk_probe(dst.parent).free - len(chunk) < budget.config.disk_min_free_gib * GIB:
                        raise BudgetExceeded("disk floor changed during snapshot")
                    writer.write(chunk)
                writer.flush()
                os.fsync(writer.fileno())
        copied = verify_inputs(destination, allow_unlisted=False, **expected)
        after = verify_inputs(source, **expected)
        extras_after = {p: file_hash(source / p) for p in _tree_files(source) - set(files)}
        if asdict(before) != asdict(after) or copied.manifest_hash != before.manifest_hash or copied.native_hashes != before.native_hashes or extras_before != extras_after:
            raise ValueError("source changed while snapshotting")
        for name in files:
            os.chmod(destination / name, 0o444)
        for path in sorted((p for p in destination.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            os.chmod(path, 0o555)
        os.chmod(destination, 0o555)
    return copied


def assert_native_origin(identity: InputIdentity, modules=None):
    """Call BEFORE loading native runtime; contamination requires a new process.

    Do not repair sys.path/sys.modules or mix checkouts. Every already-loaded
    native module must be a manifest-verified file in this snapshot's source.
    The later backend still owns checking modules it subsequently loads.
    """
    root = safe_path(identity.root)
    for name, module in (sys.modules if modules is None else modules).copy().items():
        if name.split(".")[0] not in {"sim", "pac_act", "residual_rl", "hybrid_trial4_snapshot"}:
            continue
        filename = getattr(module, "__file__", None)
        try:
            path = safe_path(filename) if filename else None
            relative = path.relative_to(root).as_posix() if path else ""
            if not relative.startswith("source/") or relative not in identity.native_hashes or file_hash(path) != identity.native_hashes[relative]:
                raise ValueError("unverified native file")
        except (ValueError, OSError, TypeError) as exc:
            raise RuntimeError(f"native module {name} came from another/unverified checkout; start a new process") from exc
