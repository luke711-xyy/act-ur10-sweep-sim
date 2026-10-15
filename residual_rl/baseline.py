"""Process-local pre-DINO sim namespace; no checkout, file replacement or model IO.

The historical sealed SHA256 map is the authority. Identical working-tree bytes
are reused; only the two DINO-modified files are read from the pinned Git commit.
All sim imports are allowlisted, including packages, so added DINO modules cannot
fall through to the ordinary disk importer. Activation requires a fresh process.
"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import linecache
from pathlib import Path, PurePosixPath
import subprocess
import sys
from types import MappingProxyType


PRE_DINO_COMMIT = '9102680d8e018b225ab34d36422572cd4b547166'
_GIT_FILES = frozenset(('sim/act/policy.py', 'sim/act/train.py'))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def disk_sources(project):
    return {str(path.relative_to(project)): _digest(path.read_bytes())
            for path in sorted((project/'sim').rglob('*.py'))}


def source_difference(expected, actual):
    return dict(matches=actual == expected, expected_files=len(expected), actual_files=len(actual),
                added=sorted(actual.keys()-expected.keys()), missing=sorted(expected.keys()-actual.keys()),
                changed=sorted(k for k in actual.keys() & expected.keys() if actual[k] != expected[k]))


def pre_dino_sources(project, expected):
    """Verify before activation; never import sim/torch/MuJoCo or edit evidence."""
    project = Path(project).resolve()
    sources, fallback = {}, []
    for name, digest in sorted(expected.items()):
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or path.parts[0] != 'sim'
                or path.suffix != '.py' or str(path) != name):
            raise ValueError('invalid frozen source path')
        local = project/name
        if not local.resolve().is_relative_to(project/'sim'):
            raise ValueError('frozen source path escapes sim')
        try:
            data = local.read_bytes()
        except FileNotFoundError as error:
            raise RuntimeError('frozen Hybrid source drift: missing '+name) from error
        if _digest(data) != digest:
            if name not in _GIT_FILES:
                raise RuntimeError('frozen Hybrid source drift outside approved DINO files: '+name)
            try:
                data = subprocess.check_output(['git', 'show', f'{PRE_DINO_COMMIT}:{name}'],
                    cwd=project, stderr=subprocess.PIPE, timeout=5)
            except (OSError, subprocess.SubprocessError) as error:
                raise RuntimeError('frozen Hybrid source drift: pinned Git blob unavailable: '+name) from error
            if _digest(data) != digest:
                raise RuntimeError('frozen Hybrid source drift: pinned Git SHA256 mismatch: '+name)
            fallback.append(name)
        sources[name] = data
    if not sources:
        raise RuntimeError('frozen Hybrid source drift: empty source manifest')
    return PinnedSources(project, sources, fallback, disk_sources(project))


class PinnedSources(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, project, sources, fallback, current):
        self.project = project
        self.sources = MappingProxyType(dict(sources))
        self._fallback = tuple(fallback)
        self._disk = dict(current)
        modules = {}
        for path in sources:
            parts = list(PurePosixPath(path).with_suffix('').parts)
            package = parts[-1] == '__init__'
            if package:
                parts.pop()
            modules['.'.join(parts)] = (path, package)
        self._modules = modules

    def manifest(self):
        effective = {p: _digest(s) for p, s in self.sources.items()}
        return dict(selection='pre_dino', commit=PRE_DINO_COMMIT, effective_matches=True,
                    effective_files=len(effective), git_fallback_files=list(self._fallback),
                    disk_source_audit=source_difference(effective, self._disk),
                    disk_source_fingerprints=dict(self._disk),
                    import_mode='verified_in_memory_allowlist_no_disk_rollback')

    def activate(self):
        installed = [f for f in sys.meta_path if isinstance(f, PinnedSources)]
        if installed:
            if len(installed) != 1 or installed[0].manifest() != self.manifest() or installed[0].project != self.project:
                raise RuntimeError('different pinned baseline already installed; use a fresh process')
            owner = installed[0]
            if any(getattr(m, '__loader__', None) is not owner for n, m in sys.modules.items()
                   if n == 'sim' or n.startswith('sim.')):
                raise RuntimeError('foreign sim already imported; use a fresh process')
            return owner
        if any(n == 'sim' or n.startswith('sim.') for n in sys.modules):
            raise RuntimeError('sim already imported; refusing to replace live modules; use a fresh process')
        sys.meta_path.insert(0, self)
        return self

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'sim' and not fullname.startswith('sim.'):
            return None
        if fullname not in self._modules:
            raise ModuleNotFoundError('sim module outside frozen pre-DINO source set: '+fullname, name=fullname)
        relative, package = self._modules[fullname]
        physical = self.project/relative
        # A nonexistent sibling filename preserves dirname/parents used for
        # assets and configs while keeping module-level inspect on pinned text.
        origin = str(physical.with_name(f'.residual_pre_dino_{PRE_DINO_COMMIT}_{physical.name}'))
        return importlib.util.spec_from_loader(fullname, self,
            origin=origin, is_package=package)

    def create_module(self, spec):
        return None

    def get_source(self, fullname):
        return self.sources[self._modules[fullname][0]].decode('utf-8')

    def exec_module(self, module):
        relative, package = self._modules[module.__name__]
        # Only the filename is virtual, not its parent directories. No file or
        # bytecode is written. Both module and function inspect see pinned text.
        module.__file__ = module.__spec__.origin
        if package:
            module.__path__ = [str((self.project/relative).parent)]
        source = self.get_source(module.__name__)
        filename = module.__spec__.origin
        linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
        exec(compile(source, filename, 'exec'), module.__dict__)
