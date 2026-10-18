"""Real filesystem regressions; only disk/time probes stand in for the OS."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest

from wam_sweep.config import WAMConfig


@pytest.fixture
def api():
    # Import inside the fixture so RED reports an absent implementation per test.
    assert importlib.util.find_spec("wam_sweep.artifacts"), "Task2 artifacts module is absent"
    from wam_sweep import artifacts, inputs
    return artifacts, inputs


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "source-bundle"
    contents = {
        "base_model/model.safetensors": b"frozen-weights",
        "base_model/config.json": b"{}",
        "base_model/policy_preprocessor.json": b"preprocessing",
        "base_model/policy_postprocessor.json": b"postprocessing",
        "evaluation_config.yaml": b"evaluation",
        "control_profile.json": b"control",
        "exclusions.json": b'{"complete_known_inputs":true}',
        "source/sim/__init__.py": b"# frozen sim\n",
        "source/sim/environments/layout.py": b"# frozen layout\n",
    }
    for name, data in contents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = {"schema": 1, "base_step": 100000, "model": "base_model",
                "files": {}, "sources": {}}
    for name, data in contents.items():
        group = "sources" if name.startswith("source/") else "files"
        key = name.removeprefix("source/") if group == "sources" else name
        manifest[group][key] = hashlib.sha256(data).hexdigest()
    manifest_bytes = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(manifest_bytes)
    contents["manifest.json"] = manifest_bytes
    return root, contents


def expected_identity(contents):
    return dict(expected_model_hash=hashlib.sha256(b"frozen-weights").hexdigest(),
                expected_manifest_hash=hashlib.sha256(contents["manifest.json"]).hexdigest())


def ledger(api, tmp_path, free_gib=100, clock=None, roots=None):
    artifacts, _ = api
    return artifacts.BudgetLedger(
        tmp_path / "ledger.json", owned_roots=roots or [tmp_path],
        config=WAMConfig().budget,
        disk_probe=lambda _: SimpleNamespace(free=int(free_gib * 2**30)),
        clock=clock or __import__("time").time,
    )


def test_admission_rejects_24gib_free(api, tmp_path):
    assert ledger(api, tmp_path, 24).admit().allowed is False


def test_admission_rejects_46gib_and_peak_old_new_temporary(api, tmp_path):
    budget = ledger(api, tmp_path)
    assert budget.admit(expected_bytes=46 * 2**30).allowed is False
    with budget.reserve(23 * 2**30):
        assert budget.admit(expected_bytes=23 * 2**30).allowed is False
    assert budget.admit().allowed is True


def test_disk_probe_changes_and_warning_thresholds(api, tmp_path):
    budget = ledger(api, tmp_path, 29)
    assert budget.poll().warnings
    budget.disk_probe = lambda _: SimpleNamespace(free=24 * 2**30)
    assert budget.poll().allowed is False


def test_nested_roots_count_once_and_never_follow_inventory_symlinks(api, tmp_path):
    owned = tmp_path / "owned"
    nested = owned / "cache"
    nested.mkdir(parents=True)
    (nested / "a").write_bytes(b"bytes")
    a = ledger(api, tmp_path / "a", roots=[owned, nested])
    b = ledger(api, tmp_path / "b", roots=[owned])
    assert a.poll().owned_bytes == b.poll().owned_bytes
    external = tmp_path / "external"
    external.mkdir()
    (external / "large").write_bytes(b"x" * 100000)
    before = a.poll().owned_bytes
    (owned / "escape").symlink_to(external)
    assert a.poll().owned_bytes - before < 10000
    with pytest.raises(ValueError, match="symlink"):
        a.ensure_owned(owned / "escape" / "large")


def test_effective_clock_pause_resume_recovery_and_72h_admission(api, tmp_path):
    now = [1000.]
    budget = ledger(api, tmp_path, clock=lambda: now[0])
    budget.record("prior", seconds=410, kind="testing")
    budget.record("start", kind="collection")
    now[0] += 10
    budget.record("pause")
    now[0] += 10000
    restored = ledger(api, tmp_path, clock=lambda: now[0])
    assert restored.poll().effective_seconds == 420
    restored.record("start", kind="training")
    now[0] += 20
    restored.record("pause")
    restored.record("prior", seconds=72 * 3600 - 440, kind="testing")
    assert restored.admit(seconds=1).allowed is False
    assert json.loads((tmp_path / "ledger.json").read_text())["effective_seconds"] == 72 * 3600


def test_pilot_cannot_exceed_two_hours_or_evade_total(api, tmp_path):
    budget = ledger(api, tmp_path)
    budget.record("prior", seconds=7200, kind="pilot", pilot=True)
    assert budget.admit(seconds=1, pilot=True).allowed is False
    assert budget.poll().effective_seconds == 7200


def test_running_pilot_poll_stops_after_two_hours(api, tmp_path):
    now = [1000.]
    b = ledger(api, tmp_path, clock=lambda: now[0])
    b.record("start", kind="throughput", pilot=True)
    now[0] += 7201
    assert b.poll().allowed is False
    b.record("pause")


def test_active_crash_recovery_charges_elapsed_and_clock_reversal_rejects(api, tmp_path):
    now = [1000.]
    b = ledger(api, tmp_path, clock=lambda: now[0])
    b.record("start", kind="testing")
    now[0] += 15
    recovered = ledger(api, tmp_path, clock=lambda: now[0])
    assert recovered.poll().effective_seconds == 15
    now[0] -= 1
    with pytest.raises(ValueError, match="backwards"):
        recovered.poll()


def test_explicit_orphan_reservation_recovery_preserves_partial_bytes(api, tmp_path):
    b = ledger(api, tmp_path)
    # Actual child exits without context-manager unwinding, as in a process crash.
    script = f'''
import os
from pathlib import Path
from types import SimpleNamespace
from wam_sweep.artifacts import BudgetLedger
from wam_sweep.config import WAMConfig
b=BudgetLedger({str(b.path)!r}, owned_roots=[{str(tmp_path)!r}], config=WAMConfig().budget, disk_probe=lambda _: SimpleNamespace(free=100*2**30))
with b.reserve(1024):
    Path({str(tmp_path / 'partial-evidence')!r}).write_bytes(b"retained partial")
    os._exit(0)
'''
    child = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=10)
    assert child.returncode == 0, child.stderr.decode()
    assert b.poll().reserved_bytes == 1024
    token, = json.loads(b.path.read_bytes())["reservations"]
    b.record("release", reservation=token, kind="reconciled crashed writer")
    assert b.poll().reserved_bytes == 0
    assert (tmp_path / "partial-evidence").read_bytes() == b"retained partial"


def test_owned_root_policy_resumes_across_hash_randomized_processes(api, tmp_path):
    roots = [tmp_path / "owned-a", tmp_path / "owned-b", tmp_path / "owned-c"]
    b = ledger(api, tmp_path, roots=roots)
    script = f'''
from wam_sweep.artifacts import BudgetLedger
from wam_sweep.config import WAMConfig
BudgetLedger({str(b.path)!r}, owned_roots={[str(p) for p in roots]!r}, config=WAMConfig().budget)
'''
    for seed in range(4):
        child = subprocess.run([sys.executable, "-c", script], capture_output=True,
                               env=dict(os.environ, PYTHONHASHSEED=str(seed)), timeout=10)
        assert child.returncode == 0, child.stderr.decode()


def test_interrupted_checkpoint_preserves_previous(api, tmp_path):
    artifacts, _ = api
    budget = ledger(api, tmp_path)
    path = tmp_path / "last.bin"
    path.write_bytes(b"previous-complete-checkpoint")

    def interrupted():
        yield b"partial-new-checkpoint"
        raise OSError("simulated interrupted disk write")

    with pytest.raises(OSError, match="interrupted"):
        artifacts.atomic_checkpoint(path, interrupted(), budget, expected_bytes=100)
    assert path.read_bytes() == b"previous-complete-checkpoint"
    assert any(p.read_bytes() == b"partial-new-checkpoint" for p in tmp_path.glob(".last.bin.*.partial"))
    artifacts.atomic_checkpoint(path, b"new-complete", budget)
    assert path.read_bytes() == b"new-complete"


def test_checkpoint_admission_precedes_any_write_and_symlink_escape(api, tmp_path):
    artifacts, _ = api
    path = tmp_path / "last"
    path.write_bytes(b"previous")
    with pytest.raises(artifacts.BudgetExceeded):
        artifacts.atomic_checkpoint(path, b"new", ledger(api, tmp_path, 24))
    assert path.read_bytes() == b"previous"
    external = tmp_path / "outside"
    external.mkdir()
    (tmp_path / "link").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        artifacts.atomic_checkpoint(tmp_path / "link" / "x", b"bad", ledger(api, tmp_path))
    assert not (external / "x").exists()


def test_nonblocking_lease_real_process_and_stale_file(api, tmp_path):
    artifacts, _ = api
    lock = tmp_path / "run.lock"
    native = tmp_path / "explicit-native.lock"
    lock.write_text("stale pid text is not liveness")
    with artifacts.RunLease(lock, native_lock=native):
        script = ("from wam_sweep.artifacts import RunLease; "
                  f"RunLease({str(lock)!r}).__enter__()")
        child = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=10)
        assert child.returncode != 0
        assert b"LeaseBusy" in child.stderr
        with pytest.raises(artifacts.LeaseBusy):
            with artifacts.RunLease(native):
                pass
    with artifacts.RunLease(lock, native_lock=native):
        pass


def test_verified_snapshot_copies_not_hardlinks_readonly_and_source_unchanged(api, tmp_path, bundle):
    _, inputs = api
    root, contents = bundle
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    expected = expected_identity(contents)
    identity = inputs.verify_inputs(root, **expected)
    assert identity.model_hash == hashlib.sha256(b"frozen-weights").hexdigest()
    assert {"perception", "control", "preprocessing", "config"} <= identity.native_hashes.keys()
    dest = tmp_path / "inputs" / "r8"
    copied = inputs.snapshot_inputs(root, dest, ledger(api, tmp_path), **expected)
    assert copied.manifest_hash == identity.manifest_hash
    for name, data in contents.items():
        path = dest / name
        assert path.read_bytes() == data
        assert path.stat().st_ino != (root / name).stat().st_ino
        assert path.stat().st_mode & 0o222 == 0
    assert before == {p: p.read_bytes() for p in before}
    assert copied.native_hashes["source/sim/environments/layout.py"] == hashlib.sha256(contents["source/sim/environments/layout.py"]).hexdigest()


@pytest.mark.parametrize("poison", ["hash", "symlink", "traversal", "unlisted"])
def test_input_verification_rejects_corruption_escape_and_unsealed_files(api, tmp_path, bundle, poison):
    _, inputs = api
    root, contents = bundle
    if poison == "hash":
        (root / "base_model/config.json").write_bytes(b"changed")
    elif poison == "symlink":
        (root / "source" / "escape").symlink_to(tmp_path)
    elif poison == "unlisted":
        (root / "source" / "extra.py").write_text("unsealed code")
    else:
        raw = json.loads((root / "manifest.json").read_text())
        raw["sources"]["../../outside"] = "0" * 64
        (root / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        inputs.verify_inputs(root, allow_unlisted=False, **expected_identity(contents))


def test_verification_pins_model_count_and_rejects_duplicate_manifest_keys(api, bundle):
    _, inputs = api
    root, contents = bundle
    expected = expected_identity(contents)
    with pytest.raises(ValueError, match="model hash"):
        inputs.verify_inputs(root, expected_manifest_hash=expected["expected_manifest_hash"])
    with pytest.raises(ValueError, match="item count"):
        inputs.verify_inputs(root, expected_entries=195, **expected)
    manifest = root / "manifest.json"
    manifest.write_text(manifest.read_text().replace('"schema": 1', '"schema": 0, "schema": 1'))
    with pytest.raises(ValueError, match="duplicate"):
        inputs.verify_inputs(root, **expected)


def test_self_consistent_manifest_edit_cannot_replace_trusted_identity(api, bundle):
    root, contents = bundle
    expected = expected_identity(contents)
    raw = json.loads((root / "manifest.json").read_bytes())
    (root / "control_profile.json").write_bytes(b"edited control")
    raw["files"]["control_profile.json"] = hashlib.sha256(b"edited control").hexdigest()
    (root / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="manifest identity"):
        api[1].verify_inputs(root, **expected)


def test_snapshot_omits_unsealed_source_files_and_runs(api, tmp_path, bundle):
    root, contents = bundle
    (root / "source/runs").mkdir()
    (root / "source/runs/.act_inference.lock").write_bytes(b"user lock evidence")
    (root / "source/README.md").write_text("unsealed document")
    dest = tmp_path / "snapshot"
    api[1].snapshot_inputs(root, dest, ledger(api, tmp_path), **expected_identity(contents))
    assert not (dest / "source/runs").exists()
    assert not (dest / "source/README.md").exists()
    assert (root / "source/runs/.act_inference.lock").read_bytes() == b"user lock evidence"


def test_snapshot_refuses_before_copy_and_rejects_imported_other_checkout(api, tmp_path, bundle):
    artifacts, inputs = api
    root, contents = bundle
    dest = tmp_path / "snapshot"
    with pytest.raises(artifacts.BudgetExceeded):
        inputs.snapshot_inputs(root, dest, ledger(api, tmp_path, 24))
    assert not dest.exists()
    identity = inputs.verify_inputs(root, **expected_identity(contents))
    with pytest.raises(RuntimeError, match="new process"):
        inputs.assert_native_origin(identity, {"sim": SimpleNamespace(__file__="/other/checkout/sim/__init__.py")})
    inputs.assert_native_origin(identity, {"sim": SimpleNamespace(__file__=str(root / "source/sim/__init__.py"))})


def test_modules_do_not_import_native_or_truth_at_import_time(api):
    child = subprocess.run([sys.executable, "-c", "import sys; import wam_sweep.inputs, wam_sweep.artifacts, wam_sweep.splits; assert not any(n.split('.')[0] in {'sim','pac_act','mujoco'} for n in sys.modules)"], capture_output=True, timeout=10)
    assert child.returncode == 0, child.stderr.decode()


# Names read from sealed pac_act/hybrid.py, evaluate_hybrid_paired_batch.runtime
# and the actual nested load helpers; no legacy modules are imported here.
@pytest.mark.parametrize("alias", [
    "_pac_frozen_hybrid_formal", "hybrid_formal_trial4_phases",
    "hybrid_formal_trial4_runtime", "frozen_small_motion_controller",
    "frozen_z_follow_controller", "reviewed_geometry_adapter",
    "sealed_wrist_phase_machine", "preserved_rate_phase_dependency",
    "approved_ik_helper", "reviewed_hybrid_helper", "reviewed_metrics_helper",
    "sealed_simultaneous_helper", "scoped_one_second_handoff", "phases", "runner",
])
def test_native_guard_rejects_foreign_loader_alias_before_loading(api, bundle, monkeypatch, alias):
    root, contents = bundle
    inputs = api[1]
    verified = inputs.verify_inputs(root, **expected_identity(contents))
    foreign = ModuleType(alias)
    foreign.__file__ = "/other/checkout/frozen-runtime.py"
    monkeypatch.setitem(sys.modules, alias, foreign)
    with pytest.raises(RuntimeError, match="new process"):
        inputs.assert_native_origin(verified)
    # Origin/content checks apply equally when the alias is a namespace root.
    with pytest.raises(RuntimeError, match="new process"):
        inputs.assert_native_origin(verified, {alias + ".nested": foreign})
    local = ModuleType(alias)
    local.__file__ = str(root / "source/sim/__init__.py")
    inputs.assert_native_origin(verified, {alias: local})
    (root / "source/sim/__init__.py").write_bytes(b"tampered native module")
    with pytest.raises(RuntimeError, match="new process"):
        inputs.assert_native_origin(verified, {alias: local})


def test_native_guard_extensible_alias_still_requires_verified_file(api, bundle):
    root, contents = bundle
    inputs = api[1]
    verified = inputs.verify_inputs(root, **expected_identity(contents))
    with pytest.raises(RuntimeError, match="new process"):
        inputs.assert_native_origin(verified, {"wam_backend_fixture": SimpleNamespace(__file__="/other/backend.py")},
                                    native_aliases=("wam_backend_fixture",))
    with pytest.raises(RuntimeError, match="new process"):
        inputs.assert_native_origin(verified, {"sealed_wrist_phase_machine": ModuleType("missing_origin")})


@pytest.mark.parametrize("pilot", [False, True])
def test_consumed_reservation_keeps_admitted_work_running_until_hard_boundary(api, tmp_path, pilot):
    now = [1000.]
    b = ledger(api, tmp_path, clock=lambda: now[0])
    cap = 7200 if pilot else 259200
    b.record("prior", seconds=cap - 100, kind="prior compute", pilot=pilot)
    with b.reserve(1234, seconds=100, pilot=pilot) as token:
        b.record("start", kind="reserved work", pilot=pilot, reservation=token)
        for consumed, remaining in [(1, 99), (20, 80), (99, 1)]:
            now[0] = 1000. + consumed
            assert b.poll().allowed is True
            state = json.loads(b.path.read_bytes())
            assert state["effective_seconds"] == cap - 100 + consumed
            assert state["reservations"][token]["seconds"] == remaining
            assert state["reservations"][token]["bytes"] == 1234
            assert state["pilot_seconds"] == (cap - 100 + consumed if pilot else 0)
        now[0] = 1100.
        assert b.admit(seconds=0.01, pilot=pilot).allowed is False
        now[0] = 1101.
        assert b.poll().allowed is False
        state = json.loads(b.path.read_bytes())
        assert state["effective_seconds"] == cap + 1
        assert state["reservations"][token]["seconds"] == 0
        assert state["pilot_seconds"] == (cap + 1 if pilot else 0)
        b.record("pause")


@pytest.mark.parametrize("pilot", [False, True])
def test_consumption_preserves_unrelated_future_reservation_and_paused_idle(api, tmp_path, pilot):
    now = [1000.]
    b = ledger(api, tmp_path, clock=lambda: now[0])
    cap = 7200 if pilot else 259200
    b.record("prior", seconds=cap - 100, kind="prior compute", pilot=pilot)
    with b.reserve(23, seconds=60, pilot=pilot) as future:
        with b.reserve(17, seconds=40, pilot=pilot) as current:
            b.record("start", kind="current", pilot=pilot, reservation=current)
            now[0] += 10
            assert b.poll().allowed is True
            b.record("pause")
            now[0] += 10000
            assert b.poll().effective_seconds == cap - 90
            state = json.loads(b.path.read_bytes())
            assert state["reservations"][future]["seconds"] == 60
            assert state["reservations"][current]["seconds"] == 30
            b.record("start", kind="continued", pilot=pilot, reservation=current)
            now[0] += 5
            assert b.poll().allowed is True
            b.record("pause")
            assert json.loads(b.path.read_bytes())["reservations"][current]["seconds"] == 25
        assert b.admit(seconds=25, pilot=pilot).allowed is True
        assert b.admit(seconds=26, pilot=pilot).allowed is False
        state = json.loads(b.path.read_bytes())
        assert state["reservations"] == {future: {"bytes": 23, "seconds": 60, "pilot": pilot}}


@pytest.mark.parametrize("pilot", [False, True])
def test_linked_reservation_resume_consumes_once_across_ledger_reload(api, tmp_path, pilot):
    now = [1000.]
    b = ledger(api, tmp_path, clock=lambda: now[0])
    with b.reserve(0, seconds=40, pilot=pilot) as future:
        with b.reserve(0, seconds=30, pilot=pilot) as current:
            b.record("start", kind="interrupted", pilot=pilot, reservation=current)
            now[0] += 7
            recovered = ledger(api, tmp_path, clock=lambda: now[0])
            assert recovered.poll().effective_seconds == 7
            state = json.loads(b.path.read_bytes())
            assert state["reservations"][current]["seconds"] == 23
            assert state["reservations"][future]["seconds"] == 40
            now[0] += 5
            assert recovered.poll().effective_seconds == 12
            assert recovered.poll().effective_seconds == 12
            state = json.loads(b.path.read_bytes())
            assert state["reservations"][current]["seconds"] == 18
            assert state["pilot_seconds"] == (12 if pilot else 0)
            recovered.record("pause")


@pytest.mark.parametrize("pilot", [False, True])
def test_reservation_overrun_is_fully_charged_without_consuming_other_work(api, tmp_path, pilot):
    now = [1000.]
    b = ledger(api, tmp_path, clock=lambda: now[0])
    cap = 7200 if pilot else 259200
    b.record("prior", seconds=cap - 40, kind="prior compute", pilot=pilot)
    with b.reserve(0, seconds=20, pilot=pilot) as future:
        with b.reserve(0, seconds=5, pilot=pilot) as current:
            b.record("start", kind="overrun", pilot=pilot, reservation=current)
            now[0] += 10
            b.record("pause")
            state = json.loads(b.path.read_bytes())
            assert state["reservations"][current]["seconds"] == 0
            assert state["reservations"][future]["seconds"] == 20
            assert state["effective_seconds"] == cap - 30
            assert b.admit(seconds=10, pilot=pilot).allowed is True
            assert b.admit(seconds=11, pilot=pilot).allowed is False


def test_reservation_link_validation_and_active_release_refusal(api, tmp_path):
    b = ledger(api, tmp_path)
    with pytest.raises(ValueError):
        b.record("start", kind="unknown reservation", reservation="not-a-token")
    with b.reserve(0, seconds=20, pilot=True) as token:
        with pytest.raises(ValueError):
            b.record("start", kind="wrong pilot accounting", reservation=token)
        b.record("start", kind="pilot", pilot=True, reservation=token)
        with pytest.raises(ValueError):
            b.record("release", reservation=token, kind="cannot orphan active work")
        b.record("pause")


def test_linked_activity_consumes_duration_without_duplicating_forecast(api, tmp_path):
    now = [1000.]
    b = ledger(api, tmp_path, clock=lambda: now[0])
    b.record("prior", seconds=259190, kind="prior compute")
    with b.reserve(0, seconds=10) as token:
        with b.activity("linked", reservation=token):
            now[0] += 3
            assert b.poll().allowed is True
        assert json.loads(b.path.read_bytes())["reservations"][token]["seconds"] == 7


def test_unlinked_legacy_active_state_keeps_unrelated_reservations(api, tmp_path):
    now = [1000.]
    b = ledger(api, tmp_path, clock=lambda: now[0])
    with b.reserve(0, seconds=20) as future:
        b.record("start", kind="legacy unlinked")
        state = json.loads(b.path.read_bytes())
        state["active"].pop("reservation", None)  # Prior schema-1 active record.
        b.path.write_text(json.dumps(state))
        now[0] += 3
        resumed = ledger(api, tmp_path, clock=lambda: now[0])
        assert resumed.poll().effective_seconds == 3
        assert json.loads(b.path.read_bytes())["reservations"][future]["seconds"] == 20
        resumed.record("pause")
