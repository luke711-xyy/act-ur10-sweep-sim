"""DINO sync5 force-control parity without a MuJoCo episode."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = PROJECT / "hybrid_trial4_snapshot" / "dinov3_sync5"


def _control_api():
    try:
        return importlib.import_module("pac_act.dinov3_control")
    except ModuleNotFoundError as error:
        if error.name != "pac_act.dinov3_control":
            raise
        pytest.fail("DINO sync5 controller-profile adapter is not implemented")


def test_sync5_profile_loads_exact_contact_and_relief_contract():
    api = _control_api()
    profile = api.load_sync5_profile(PROFILE_ROOT)
    effective = profile["effective_profile"]

    assert profile["profile_id"] == "dinov3_sync5"
    assert effective["target_force_N"] == 1.0
    assert effective["total_Z_limit_m"] == 0.002
    assert effective["seek_MBK"] == [10.0, 400.0, 10.0]
    assert effective["maintain_MBK"] == [10.0, 800.0, 10.0]
    assert effective["gate_m"] == 0.001
    assert effective["contact_threshold_N"] == 0.5
    assert effective["contact_dwell_s"] == 0.1
    assert effective["stable_band_N"] == [0.8, 1.2]
    assert effective["stable_dwell_s"] == 0.2
    assert effective["handoff_window_s"] == 2.0
    assert effective["XY_acceleration_m_s2"] == 0.05
    assert effective["yaw_acceleration_rad_s2"] == 0.5
    assert effective["after_handoff_window"] == "exact_native_ACT_without_continued_acceleration_limiting"
    assert effective["ACT_predicted_Z_executed"] is False
    assert effective["maintain_force_compliance"] == {
        "gain_m_per_N": 5e-05,
        "offset_cap_m": 0.0005,
        "active_phase": "maintain",
        "feedback": "compensated_filtered_wrist_Fz",
    }
    assert effective["overforce_relief"] == {
        "activation_force": 1.5,
        "release_force": 1.1,
        "upward_rate": 0.0005,
        "return_rate": 0.0005,
        "max_offset": 0.00025,
    }


def test_sync5_admittance_adds_bounded_compliance_only_while_maintaining():
    api = _control_api()
    profile = api.load_sync5_profile(PROFILE_ROOT)

    class BaseAdmittance:
        def __init__(self, phase, correction=0.0):
            self.phase = phase
            self.correction = correction
            self.calls = []

        def step(self, desired_force, wrist_force):
            self.calls.append((desired_force, wrist_force))
            return self.correction

    compliant_type = api.compliant_admittance_type(BaseAdmittance, profile)
    seeking = compliant_type("seek")
    settling = compliant_type("settle")
    maintaining = compliant_type("maintain")

    assert seeking.step(1.0, 3.0) == 0.0
    assert settling.step(1.0, 3.0) == 0.0
    assert maintaining.step(1.0, 3.0) == pytest.approx(0.0001)
    assert maintaining.step(1.0, 25.0) == pytest.approx(0.0005)
    assert maintaining.step(1.0, -20.0) == pytest.approx(-0.0005)
    assert maintaining.calls == [(1.0, 3.0), (1.0, 25.0), (1.0, -20.0)]


def test_sync5_admittance_respects_total_Z_envelope_after_compliance():
    api = _control_api()
    profile = api.load_sync5_profile(PROFILE_ROOT)

    class BaseAdmittance:
        phase = "maintain"

        def step(self, desired_force, wrist_force):
            return 0.0019

    compliant_type = api.compliant_admittance_type(BaseAdmittance, profile)
    controller = compliant_type()
    assert controller.step(1.0, 25.0) == pytest.approx(0.002)
    assert controller.step(1.0, -20.0) == pytest.approx(0.0014)


def test_sync5_configuration_is_episode_local_and_restores_sealed_defaults():
    from sim.config import Config

    api = _control_api()
    profile = api.load_sync5_profile(PROFILE_ROOT)

    class TrialSpec:
        def __init__(self, force, seek, maintain):
            self.force_N, self.seek, self.maintain = force, seek, maintain

        def check_clock(self, dt):
            assert dt == pytest.approx(0.01)

    class BaseAdmittance:
        def __init__(self, dt, spec):
            self.dt, self.spec, self.phase = dt, spec, "maintain"

        def step(self, desired_force, wrist_force):
            return 0.0

    original_spec = TrialSpec(1.0, (3.0, 4.0, 5.0), (6.0, 7.0, 8.0))
    module = SimpleNamespace(TwoStageAdmittance=BaseAdmittance, TRIAL_SPEC=original_spec)
    cfg = Config({
        "sim": {"control_hz": 100.0},
        "act": {"hybrid_admittance": {
            "seek_MBK": [10.0, 400.0, 10.0],
            "maintain_MBK": [10.0, 800.0, 10.0],
            "force_compliance_m_per_N": 5e-5,
        }},
    })

    api.configure_sync5_admittance(module, cfg, profile)
    assert module.TRIAL_SPEC.force_N == 1.0
    assert module.TRIAL_SPEC.seek == (10.0, 400.0, 10.0)
    assert module.TRIAL_SPEC.maintain == (10.0, 800.0, 10.0)
    assert issubclass(module.TwoStageAdmittance, BaseAdmittance)

    api.reset_trial4_admittance(module)
    assert module.TwoStageAdmittance is BaseAdmittance
    assert module.TRIAL_SPEC is original_spec


def test_sync5_relief_activation_release_and_offset_cap_are_configured():
    from sim.act.rollout import _overforce_relief_loop
    from sim.config import load_config

    api = _control_api()
    profile = api.load_sync5_profile(PROFILE_ROOT)
    cfg = load_config(str(PROFILE_ROOT / "evaluation_config.yaml"))
    relief = _overforce_relief_loop(cfg)

    assert relief.params.activation_force == profile["effective_profile"]["overforce_relief"]["activation_force"]
    assert relief.params.release_force == profile["effective_profile"]["overforce_relief"]["release_force"]
    assert relief.params.upward_rate == profile["effective_profile"]["overforce_relief"]["upward_rate"]
    assert relief.params.return_rate == profile["effective_profile"]["overforce_relief"]["return_rate"]
    assert relief.params.max_offset == profile["effective_profile"]["overforce_relief"]["max_offset"]
    assert relief.step(1.49) == 0.0
    one_step = relief.step(1.5)
    assert one_step == pytest.approx(5e-6)
    for _ in range(60):
        relief.step(1.5)
    assert relief.offset == pytest.approx(0.00025)
    assert relief.step(1.1) == pytest.approx(0.00025)
    assert relief.step(1.09) == pytest.approx(0.000245)


def test_cfg_for_propagates_verified_bundle_profile_and_rejects_unknown(tmp_path, monkeypatch):
    from pac_act import hybrid
    from pac_act.lifecycle import cfg_for

    class Formal:
        @staticmethod
        def candidate_config(cfg, target):
            cfg.set_path("task.target_count", target)
            return cfg

    monkeypatch.setattr(hybrid, "load_trial4", lambda: (Formal(), None))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "evaluation_config.yaml").write_bytes(
        (PROFILE_ROOT / "evaluation_config.yaml").read_bytes())
    (bundle / "control_profile.json").write_bytes(
        (PROFILE_ROOT / "control_profile.json").read_bytes())
    (bundle / "manifest.json").write_text(json.dumps({"control_profile_id": "dinov3_sync5"}))

    cfg = cfg_for(bundle, 4)
    assert cfg.get_path("act.hybrid_control_profile_id") == "dinov3_sync5"
    assert cfg.get_path("act.hybrid_control_profile_root") == str(bundle.resolve())
    assert cfg.task.target_count == 4

    (bundle / "manifest.json").write_text(json.dumps({"control_profile_id": "resnet_magic"}))
    with pytest.raises(ValueError, match="unknown controller profile"):
        cfg_for(bundle, 4)


def test_sync5_profile_loader_rejects_relief_config_drift(tmp_path):
    api = _control_api()
    control_profile = PROFILE_ROOT / "control_profile.json"
    evaluation_config = PROFILE_ROOT / "evaluation_config.yaml"
    assert control_profile.is_file() and evaluation_config.is_file()

    altered = tmp_path / "dinov3_sync5"
    altered.mkdir()
    profile_data = json.loads(control_profile.read_text())
    (altered / "control_profile.json").write_text(json.dumps(profile_data))
    config_text = evaluation_config.read_text()
    config_text = config_text.replace("activation_force: 1.5", "activation_force: 10.0")
    (altered / "evaluation_config.yaml").write_text(config_text)

    with pytest.raises(ValueError, match="overforce relief differs"):
        api.load_sync5_profile(altered)
