"""Fail-closed DINO sync5 control profile for the PAC-ACT runtime."""
from __future__ import annotations

import json
from pathlib import Path


SYNC5_PROFILE_ID = "dinov3_sync5"
_EXPECTED_EFFECTIVE_PROFILE = {
    "target_force_N": 1.0,
    "total_Z_limit_m": 0.002,
    "seek_MBK": [10.0, 400.0, 10.0],
    "maintain_MBK": [10.0, 800.0, 10.0],
    "gate_m": 0.001,
    "contact_threshold_N": 0.5,
    "contact_dwell_s": 0.1,
    "stable_band_N": [0.8, 1.2],
    "stable_dwell_s": 0.2,
    "handoff_window_s": 2.0,
    "XY_acceleration_m_s2": 0.05,
    "XY_speed_m_s": 0.08,
    "yaw_speed_rad_s": 0.35,
    "yaw_acceleration_rad_s2": 0.5,
    "up_m_s": 0.02,
    "down_m_s": 0.1,
    "tracking_gain": 0.5,
    "tracking_bias_cap_m": 0.0002,
    "IK_early_exit_threshold": 1e-6,
    "force_feedback": "compensated_filtered_wrist_Fz_only",
    "Z_owner": "admittance_plus_bounded_actual_Z_follow",
    "ACT_predicted_Z_executed": False,
    "after_handoff_window": "exact_native_ACT_without_continued_acceleration_limiting",
    "maintain_force_compliance": {
        "gain_m_per_N": 5e-5,
        "offset_cap_m": 0.0005,
        "active_phase": "maintain",
        "feedback": "compensated_filtered_wrist_Fz",
    },
    "overforce_relief": {
        "activation_force": 1.5,
        "release_force": 1.1,
        "upward_rate": 0.0005,
        "return_rate": 0.0005,
        "max_offset": 0.00025,
    },
}


def _read_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as error:  # pragma: no cover - runtime dependency gate
        raise RuntimeError("PyYAML is required to verify the DINO control profile") from error
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("DINO sync5 evaluation config must be a mapping")
    return value


def _path(mapping: dict, dotted: str):
    value = mapping
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _equal(actual, expected) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(actual) - float(expected)) <= 1e-12
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                and all(_equal(a, e) for a, e in zip(actual, expected)))
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and set(actual) == set(expected)
                and all(_equal(actual[k], v) for k, v in expected.items()))
    return actual == expected


def _require_config_value(config: dict, key: str, expected, error: str):
    if not _equal(_path(config, key), expected):
        raise ValueError(error)


def _validate_evaluation_config(config: dict, profile: dict):
    effective = profile["effective_profile"]
    for key, expected in {
        "controller.desired_force": effective["target_force_N"],
        "controller.admittance.m": effective["seek_MBK"][0],
        "controller.admittance.b": effective["seek_MBK"][1],
        "controller.admittance.k": effective["seek_MBK"][2],
        "controller.delta_z_limit": effective["total_Z_limit_m"],
        "controller.delta_z_upward_rate_limit": effective["up_m_s"],
        "controller.delta_z_downward_rate_limit": effective["down_m_s"],
        "controller.overforce_relief": effective["overforce_relief"],
        "act.hybrid_admittance.seek_MBK": effective["seek_MBK"],
        "act.hybrid_admittance.maintain_MBK": effective["maintain_MBK"],
        "act.hybrid_admittance.force_compliance_m_per_N":
            effective["maintain_force_compliance"]["gain_m_per_N"],
    }.items():
        _require_config_value(config, key, expected,
                              "overforce relief differs" if key.startswith("controller.overforce_relief")
                              else f"DINO sync5 config differs at {key}")
    if _path(config, "act.hybrid_sweep_smoothing") is not None:
        raise ValueError("DINO sync5 profile must not enable continuous ACT sweep smoothing")


def load_sync5_profile(profile_root) -> dict:
    """Load and validate the immutable sync5 profile and its paired config."""
    root = Path(profile_root)
    profile_path = root / "control_profile.json"
    config_path = root / "evaluation_config.yaml"
    if not profile_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("DINO sync5 bundle requires control_profile.json and evaluation_config.yaml")
    profile = json.loads(profile_path.read_text())
    if profile.get("schema") != 1 or profile.get("profile_id") != SYNC5_PROFILE_ID:
        raise ValueError("unknown or unsupported controller profile")
    if not _equal(profile.get("effective_profile"), _EXPECTED_EFFECTIVE_PROFILE):
        raise ValueError("DINO sync5 effective control profile differs")
    _validate_evaluation_config(_read_config(config_path), profile)
    return profile


def compliant_admittance_type(base_type, profile):
    """Add the recorded, bounded maintain-phase wrist-force compliance."""
    effective = profile["effective_profile"]
    compliance = effective["maintain_force_compliance"]
    gain = float(compliance["gain_m_per_N"])
    offset_cap = float(compliance["offset_cap_m"])
    total_limit = float(effective["total_Z_limit_m"])

    class ContactCompliantAdmittance(base_type):
        def step(self, desired_force, wrist_force):
            correction = float(super().step(desired_force, wrist_force))
            if self.phase == "maintain":
                elastic = max(-offset_cap, min(offset_cap,
                    gain * (float(wrist_force) - float(desired_force))))
                correction += elastic
            return max(-total_limit, min(total_limit, correction))

    ContactCompliantAdmittance.__name__ = "DinoSync5ContactCompliantAdmittance"
    return ContactCompliantAdmittance


def reset_trial4_admittance(module):
    """Restore the sealed base before configuring each episode in this process."""
    base_type = getattr(module, "_pac_dinov3_sync5_base_admittance_type", None)
    if base_type is None:
        base_type = module.TwoStageAdmittance
        module._pac_dinov3_sync5_base_admittance_type = base_type
    base_spec = getattr(module, "_pac_dinov3_sync5_base_trial_spec", None)
    if base_spec is None:
        base_spec = module.TRIAL_SPEC
        module._pac_dinov3_sync5_base_trial_spec = base_spec
    module.TwoStageAdmittance = base_type
    module.TRIAL_SPEC = base_spec


def configure_sync5_admittance(module, cfg, profile):
    """Install sync5 gains and wrapper before the frozen contact acquisition."""
    reset_trial4_admittance(module)
    effective = profile["effective_profile"]
    values = cfg.get_path("act.hybrid_admittance")
    expected = {
        "seek_MBK": effective["seek_MBK"],
        "maintain_MBK": effective["maintain_MBK"],
        "force_compliance_m_per_N":
            effective["maintain_force_compliance"]["gain_m_per_N"],
    }
    if not isinstance(values, dict):
        values = values.to_dict() if hasattr(values, "to_dict") else values
    if not _equal(values, expected):
        raise ValueError("runtime DINO sync5 admittance differs from bundle profile")
    spec_type = type(module.TRIAL_SPEC)
    module.TRIAL_SPEC = spec_type(
        float(effective["target_force_N"]), tuple(effective["seek_MBK"]),
        tuple(effective["maintain_MBK"]))
    if hasattr(module.TRIAL_SPEC, "check_clock"):
        dt = 1.0 / float(cfg.sim.control_hz)
        module.TRIAL_SPEC.check_clock(dt)
    module.TwoStageAdmittance = compliant_admittance_type(module.TwoStageAdmittance, profile)
