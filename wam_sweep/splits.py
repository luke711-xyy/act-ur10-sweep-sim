"""Predetermined parent families, then budgeted native reset-pose binding.

Provisional seed reservations are NEVER ready for collection/training/evaluation.
Task5 must supply a source-checked resolver returning actual post-reset/settled
component positions + quaternions, not sample_layout output. Task12/13 must bind
and freeze EVERY parent before outcomes. This module does not import MuJoCo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .artifacts import atomic_checkpoint, safe_path
from .config import WAMConfig
from .contracts import InputIdentity
from .inputs import digest_json, file_hash


@dataclass(frozen=True, slots=True)
class LayoutRecord:
    parent_id: str
    seed: int
    target_count: int
    split: str
    phase: int
    purpose: str
    physical_fingerprint: str | None = None


def _families(config):
    groups = [("train", i, count, seed) for i, (count, seed) in enumerate(zip(
        config.data.main_trajectories, (90000000, 90001000, 90002000)))]
    groups += [("bootstrap", -1, config.data.perception_bootstrap_trajectories, 90010000),
               ("validation", -1, config.evaluation.validation_parent_layouts, 90020000),
               ("calibration", -1, config.evaluation.calibration_parent_layouts, 90030000),
               ("preflight", -1, config.evaluation.preflight_layouts, 90040000),
               ("test", -1, config.evaluation.formal_layouts, 90050000)]
    result = []
    for split, phase, count, first in groups:
        for i in range(count):
            purpose = split
            if split == "validation":
                purpose = "model_selection" if i < 30 else "development_calibration"
            result.append(LayoutRecord(f"wam-v1/{split}/{phase}/{first+i}", first+i,
                                       i % 6 + 1, split, phase, purpose))
    return tuple(result)


def _exclusions(value):
    if isinstance(value, (str, Path)):
        value = json.loads(safe_path(value).read_bytes())
    if not isinstance(value, Mapping) or value.get("complete_known_inputs") is not True:
        raise ValueError("sealed exclusions must have complete_known_inputs=true")
    if not all(k in value for k in ("seeds", "fingerprints", "reserved_seed_ranges")):
        raise ValueError("missing sealed exclusions")
    if any(type(s) is not int or s < 0 for s in value["seeds"]):
        raise ValueError("invalid excluded seed")
    for low, high in value["reserved_seed_ranges"]:
        if type(low) is not int or type(high) is not int or low < 0 or high < low:
            raise ValueError("invalid reserved seed range")
    for fp in value["fingerprints"]:
        _hash(fp)
    return json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))


def _hash(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("SHA256 required")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("SHA256 required") from exc
    return value


def assert_family_isolation(records):
    """Accept parent or derivative mappings/EpisodeSpecs without truth handles.

    Different N, starts, branches and overlapping windows retain parent_id.
    Repeated seed/fingerprint aliases cannot disguise a second parent family.
    """
    parent_split, aliases = {}, {"seed": {}, "physical_fingerprint": {}}
    for record in records:
        def get(key):
            return record.get(key) if isinstance(record, Mapping) else getattr(record, key, None)
        parent, split = get("parent_id"), get("split")
        if not isinstance(parent, str) or not parent or not isinstance(split, str) or not split:
            raise ValueError("family parent_id/split required")
        if parent in parent_split and parent_split[parent] != split:
            raise ValueError("parent family crosses splits")
        parent_split[parent] = split
        for field, known in aliases.items():
            value = get(field)
            if field == "physical_fingerprint" and value is None:
                value = get("layout_fingerprint")
            if value is not None:
                if value in known and known[value] != parent:
                    raise ValueError(f"same {field} aliases another parent family")
                known[value] = parent


@dataclass(frozen=True, slots=True)
class SplitManifest:
    records: tuple[LayoutRecord, ...]
    config_hash: str
    generator_hash: str
    exclusions_json: str
    binding_json: str | None = None
    frozen: bool = False

    def __post_init__(self):
        if type(self.records) is not tuple or type(self.frozen) is not bool:
            raise ValueError("immutable records and explicit frozen state required")
        config = WAMConfig()
        if self.config_hash != digest_json(asdict(config)) or self.generator_hash != file_hash(__file__):
            raise ValueError("config/generator identity drift")
        expected = _families(config)
        if tuple(replace(r, physical_fingerprint=None) for r in self.records) != expected:
            raise ValueError("predetermined parent families changed; no outcome-based replacement")
        exclusions = _exclusions(json.loads(self.exclusions_json))
        seeds = set(exclusions["seeds"])
        fingerprints = set(exclusions["fingerprints"])
        for record in self.records:
            # Inclusive endpoints conservatively exclude the entire sealed range.
            if record.seed in seeds or any(a <= record.seed <= b for a, b in exclusions["reserved_seed_ranges"]):
                raise ValueError(f"excluded/reserved seed: {record.seed}")
            if record.physical_fingerprint is not None:
                _hash(record.physical_fingerprint)
                if record.physical_fingerprint in fingerprints:
                    raise ValueError("physical fingerprint is excluded")
        assert_family_isolation(self.records)
        has_bound = any(r.physical_fingerprint is not None for r in self.records)
        if has_bound != (self.binding_json is not None):
            raise ValueError("bound fingerprints require native binding metadata")
        if self.binding_json is not None:
            metadata = self.binding_metadata
            if set(metadata) != {"input_identity", "resolver_hash", "config_hash", "generator_hash", "native_generator_hash"}:
                raise ValueError("incomplete binding metadata")
            verified = InputIdentity(**metadata["input_identity"])
            _hash(metadata["resolver_hash"])
            if metadata["config_hash"] != self.config_hash or metadata["generator_hash"] != self.generator_hash or metadata["native_generator_hash"] != verified.native_hashes.get("source/sim/environments/layout.py"):
                raise ValueError("binding dependency drift")
        if self.frozen and not all(r.physical_fingerprint is not None for r in self.records):
            raise ValueError("unbound physical fingerprints cannot be frozen")

    @property
    def binding_metadata(self):
        return json.loads(self.binding_json) if self.binding_json else None

    @property
    def ready(self):
        return self.frozen and all(r.physical_fingerprint is not None for r in self.records)

    def require_ready(self, consumer):
        if not self.ready:
            raise ValueError(f"split manifest is not frozen/ready for {consumer}")
        return self

    def _document(self):
        return dict(schema=1, records=[asdict(r) for r in self.records], config_hash=self.config_hash,
                    generator_hash=self.generator_hash, exclusions_json=self.exclusions_json,
                    binding_json=self.binding_json, frozen=self.frozen)

    @property
    def manifest_hash(self):
        return digest_json(self._document())

    def save(self, path, budget):
        raw = dict(self._document(), manifest_hash=self.manifest_hash)
        atomic_checkpoint(path, json.dumps(raw, sort_keys=True, allow_nan=False).encode(), budget)

    @classmethod
    def load(cls, path):
        raw = json.loads(safe_path(path).read_bytes())
        digest = raw.pop("manifest_hash")
        if raw.pop("schema") != 1 or set(raw) != {"records", "config_hash", "generator_hash", "exclusions_json", "binding_json", "frozen"}:
            raise ValueError("invalid split manifest schema")
        raw["records"] = tuple(LayoutRecord(**r) for r in raw["records"])
        result = cls(**raw)
        if digest != result.manifest_hash:
            raise ValueError("split manifest hash mismatch")
        return result

    def bind(self, resolver, budget, *, input_identity: InputIdentity,
             resolver_hash: str, seconds_per_layout: float, parent_ids=None):
        """Return a new provisional manifest after real reset-pose resolution.

        An exception leaves this immutable manifest unchanged; callers may save
        successful batches to resume. Already-bound families cannot be replaced.
        Resolver provenance is supplied by Task5's source-checked native adapter.
        """
        if self.frozen:
            raise ValueError("frozen manifest cannot be rebound")
        _hash(resolver_hash)
        native_generator = input_identity.native_hashes.get("source/sim/environments/layout.py")
        _hash(native_generator)
        metadata = dict(input_identity=asdict(input_identity), resolver_hash=resolver_hash,
                        config_hash=self.config_hash, generator_hash=self.generator_hash,
                        native_generator_hash=native_generator)
        binding = json.dumps(metadata, sort_keys=True, allow_nan=False)
        if self.binding_json is not None and binding != self.binding_json:
            raise ValueError("native resolver/input identity changed during binding")
        selected = {r.parent_id for r in self.records if r.physical_fingerprint is None} if parent_ids is None else set(parent_ids)
        lookup = {r.parent_id: r for r in self.records}
        if not selected or not selected <= lookup.keys():
            raise ValueError("unknown/empty binding family set")
        if any(lookup[p].physical_fingerprint is not None for p in selected):
            raise ValueError("family is already bound")
        if type(seconds_per_layout) not in (int, float) or not np.isfinite(seconds_per_layout) or seconds_per_layout <= 0:
            raise ValueError("positive reset time budget estimate required")
        records = list(self.records)
        excluded = set(json.loads(self.exclusions_json)["fingerprints"])
        known = {r.physical_fingerprint for r in records if r.physical_fingerprint is not None}
        with budget.reserve(0, seconds=len(selected) * seconds_per_layout):
            with budget.activity("native_reset_fingerprint"):
                for i, record in enumerate(records):
                    if record.parent_id not in selected:
                        continue
                    budget.poll().require()
                    with np.errstate(over="ignore", invalid="ignore"):
                        poses = np.asarray(resolver(record), dtype="<f4", order="C")
                    if poses.shape != (6, 7) or not np.isfinite(poses).all():
                        raise ValueError("resolver must return finite post-reset (6,7) physical poses")
                    fp = hashlib.sha256(poses.tobytes(order="C")).hexdigest()
                    if fp in excluded:
                        raise ValueError("physical fingerprint is excluded")
                    if fp in known:
                        raise ValueError("physical fingerprint aliases another parent family")
                    known.add(fp)
                    records[i] = replace(record, physical_fingerprint=fp)
                budget.poll().require()
        return replace(self, records=tuple(records), binding_json=binding)

    def freeze(self):
        return replace(self, frozen=True)


def make_split_manifest(config: WAMConfig, exclusions) -> SplitManifest:
    sealed = _exclusions(exclusions)
    return SplitManifest(_families(config), digest_json(asdict(config)), file_hash(__file__),
                         json.dumps(sealed, sort_keys=True, allow_nan=False))
