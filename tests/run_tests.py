"""Test runner.

``pytest`` is the supported way to run this suite::

    pytest -q

This script simply delegates to pytest when it is installed.  When it is not
(some minimal/offline environments), it falls back to a small built-in runner
that understands the subset of the pytest API this suite uses: ``approx``,
``raises``, ``fixture``, ``mark.parametrize``, ``mark.skipif``,
``importorskip``, ``skip`` and the ``tmp_path`` fixture.  The fallback exists so
the MuJoCo-free tests can always be executed; it is not a pytest replacement.

    python tests/run_tests.py [-k PATTERN] [-v]
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import sys
import tempfile
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------
# minimal pytest shim
# --------------------------------------------------------------------------
class Skipped(Exception):
    pass


class _Approx:
    # stop numpy from broadcasting `ndarray == approx(...)` element-wise
    __array_ufunc__ = None
    __array_priority__ = 1000

    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _close(self, a, b) -> bool:
        if self.abs is None and self.rel is None:
            tol = 1e-6 * max(1.0, abs(b))
        else:
            tol = 0.0
            if self.abs is not None:
                tol = max(tol, float(self.abs))
            if self.rel is not None:
                tol = max(tol, float(self.rel) * abs(b))
        return abs(a - b) <= tol

    def __eq__(self, other):
        import numpy as np

        exp = self.expected
        if isinstance(exp, (list, tuple)) or hasattr(exp, "__len__"):
            a = np.asarray(other, dtype=float).ravel()
            b = np.asarray(exp, dtype=float).ravel()
            return a.shape == b.shape and all(self._close(x, y) for x, y in zip(a, b))
        return self._close(float(other), float(exp))

    def __req__(self, other):  # pragma: no cover
        return self.__eq__(other)

    def __repr__(self):
        return f"approx({self.expected!r})"


class _Raises:
    def __init__(self, exc, match=None):
        self.exc = exc
        self.match = match
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(exc_type, self.exc):
            return False
        self.value = exc
        if self.match is not None:
            import re

            if not re.search(self.match, str(exc)):
                raise AssertionError(
                    f"exception message {str(exc)!r} does not match {self.match!r}"
                )
        return True


def _make_shim() -> types.ModuleType:
    module = types.ModuleType("pytest")

    def fixture(func=None, **kwargs):
        def wrap(fn):
            fn._pytest_fixture = True
            fn._pytest_fixture_name = kwargs.get("name", fn.__name__)
            return fn
        return wrap(func) if callable(func) else wrap

    class _Mark:
        @staticmethod
        def parametrize(argnames, argvalues):
            names = [n.strip() for n in argnames.split(",")]

            def wrap(fn):
                cases = getattr(fn, "_pytest_params", [])
                cases.append((names, list(argvalues)))
                fn._pytest_params = cases
                return fn
            return wrap

        @staticmethod
        def skipif(condition, reason=""):
            def wrap(fn):
                if condition:
                    fn._pytest_skip = reason or "skipped"
                return fn
            return wrap

        def __getattr__(self, _name):  # pragma: no cover - unused marks are no-ops
            def wrap(*_a, **_kw):
                return lambda fn: fn
            return wrap

    def importorskip(name, reason=None, **_kw):
        try:
            return importlib.import_module(name)
        except ImportError:
            raise Skipped(reason or f"{name} is not installed")

    def skip(reason=""):
        raise Skipped(reason)

    module.approx = lambda expected, rel=None, abs=None: _Approx(expected, rel, abs)
    module.raises = _Raises
    module.fixture = fixture
    module.mark = _Mark()
    module.importorskip = importorskip
    module.skip = skip
    module.Skipped = Skipped
    return module


def _ensure_pytest_shim() -> bool:
    try:
        import pytest  # noqa: F401
        return True
    except ImportError:
        sys.modules["pytest"] = _make_shim()
        return False


# --------------------------------------------------------------------------
# fallback collection / execution
# --------------------------------------------------------------------------
def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"testmod_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_fixture(name, fixtures, tmp_root, cache):
    if name == "tmp_path":
        path = Path(tempfile.mkdtemp(dir=tmp_root))
        return path
    if name in cache:
        return cache[name]
    if name not in fixtures:
        raise AssertionError(f"unknown fixture {name!r}")
    fn = fixtures[name]
    kwargs = {p: _resolve_fixture(p, fixtures, tmp_root, cache)
              for p in inspect.signature(fn).parameters}
    value = fn(**kwargs)
    cache[name] = value
    return value


def _expand(fn):
    """Expand stacked parametrize decorators into concrete keyword sets."""
    cases = getattr(fn, "_pytest_params", None)
    if not cases:
        return [({}, "")]
    combos = [({}, "")]
    for names, values in cases:
        expanded = []
        for base, label in combos:
            for value in values:
                items = value if len(names) > 1 else (value,)
                kwargs = dict(base)
                kwargs.update(dict(zip(names, items)))
                suffix = "-".join(str(v) for v in items)
                expanded.append((kwargs, f"{label}[{suffix}]" if label else f"[{suffix}]"))
        combos = expanded
    return combos


def run_fallback(pattern=None, verbose=False) -> int:
    test_dir = Path(__file__).resolve().parent
    files = sorted(test_dir.glob("test_*.py"))
    passed = failed = skipped = 0
    failures = []
    tmp_root = tempfile.mkdtemp(prefix="sweepsim-tests-")

    for path in files:
        try:
            module = _load_module(path)
        except Skipped as exc:
            print(f"  SKIP  {path.name}: {exc}")
            skipped += 1
            continue
        except Exception:
            print(f"  ERROR importing {path.name}")
            traceback.print_exc()
            failed += 1
            failures.append(path.name)
            continue

        fixtures = {getattr(fn, "_pytest_fixture_name", name): fn
                    for name, fn in vars(module).items()
                    if callable(fn) and getattr(fn, "_pytest_fixture", False)}
        tests = [(name, fn) for name, fn in vars(module).items()
                 if name.startswith("test_") and callable(fn)]
        tests.sort(key=lambda item: inspect.getsourcelines(item[1])[1])

        for name, fn in tests:
            for kwargs, label in _expand(fn):
                full = f"{path.stem}::{name}{label}"
                if pattern and pattern not in full:
                    continue
                if getattr(fn, "_pytest_skip", None):
                    skipped += 1
                    continue
                cache = {}
                try:
                    params = inspect.signature(fn).parameters
                    call = dict(kwargs)
                    for pname in params:
                        if pname not in call:
                            call[pname] = _resolve_fixture(pname, fixtures, tmp_root, cache)
                    fn(**call)
                    passed += 1
                    if verbose:
                        print(f"  PASS  {full}")
                except Skipped as exc:
                    skipped += 1
                    if verbose:
                        print(f"  SKIP  {full}: {exc}")
                except Exception:
                    failed += 1
                    failures.append(full)
                    print(f"  FAIL  {full}")
                    traceback.print_exc(limit=6)

    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")
    if failures:
        print("  failing tests:")
        for item in failures:
            print(f"    - {item}")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the test suite.")
    parser.add_argument("-k", dest="pattern", default=None)
    parser.add_argument("-v", dest="verbose", action="store_true")
    parser.add_argument("--force-fallback", action="store_true")
    args = parser.parse_args(argv)

    has_pytest = _ensure_pytest_shim()
    if has_pytest and not args.force_fallback:
        import pytest

        extra = ["-k", args.pattern] if args.pattern else []
        return pytest.main([str(Path(__file__).resolve().parent), "-q", *extra])
    print("  pytest not available -- using the built-in fallback runner\n")
    return run_fallback(args.pattern, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
