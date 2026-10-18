"""Typed, complete and fail-closed settings for the approved WAM v1 experiment.

These are preregistered settings, not SAC/PAC learner defaults. Loading requires
every field and rejects unknown fields, nonfinite numbers and protocol drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from math import isfinite, pi
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import EPISODE_ARMS


@dataclass(frozen=True, slots=True)
class TimingConfig:
    history_seconds: float = 2.0
    observation_hz: int = 5
    history_frames: int = 11
    prediction_seconds: float = 1.0
    execution_hz: int = 25
    prediction_steps: int = 25
    replan_seconds: float = 0.2
    execute_steps: int = 5


@dataclass(frozen=True, slots=True)
class ModelConfig:
    object_features: int = 34
    brush_features: int = 9
    context_features: int = 15
    graph_hidden: int = 128
    message_passing_layers: int = 3
    proposal_modes: int = 4
    proposal_hidden: tuple[int, ...] = (256, 256)
    proposal_output: int = 15
    control_knots: int = 5
    future_gru_hidden: int = 128
    evaluation_hidden: tuple[int, ...] = (256, 128)
    success_outputs: int = 1
    count_outputs: int = 7
    appearance_features: int = 16
    physical_outputs: int = 9
    position_velocity_log_std_outputs: int = 4
    state_outputs: int = 3
    shape_outputs: int = 2
    visibility_outputs: int = 1
    ensemble_size: int = 1
    goal_in_physical_head: bool = False


@dataclass(frozen=True, slots=True)
class ControlConfig:
    xy_residual_ratio: float = 0.8
    yaw_residual_degrees: float = 45.0
    yaw_residual_radians: float = pi / 4
    accumulate_residual: bool = False
    modify_z: bool = False
    global_x_monotonic: bool = False
    act_local_forward_only: bool = True
    zero_xy_zero_correction: bool = True
    zero_residual_filter_preserves_act: bool = True
    warmup_frames: int = 0
    episode_seconds: float = 45.0
    episode_frames: int = 1200
    first_entry_window_seconds: float = 2.0
    reset_entry_window: bool = False
    recovery_rounds: int = 0
    exact_count_no_partial_overlap: bool = True
    preserve_native_safety: bool = True


@dataclass(frozen=True, slots=True)
class DataConfig:
    object_count: int = 6
    target_counts: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    perception_bootstrap_trajectories: int = 120
    bootstrap_perception_only: bool = True
    main_trajectories: tuple[int, ...] = (240, 120, 120)
    teacher_starts: tuple[int, ...] = (600, 300, 300)
    short_branches_per_start: int = 12
    full_continuations_per_start: int = 3
    continuation_kinds: tuple[str, ...] = ("act", "wam", "wam")
    continuation_phases: tuple[int, ...] = (-1, 0, 1)
    final_model_phase: int = 2
    min_success_starts_per_phase_per_n: int = 20
    min_success_parent_layouts_per_phase_per_n: int = 10
    all_failure_action_supervision: bool = False
    merge_old_physical_data: bool = True
    mix_unversioned_terminal_labels: bool = False
    parent_family_isolation: bool = True


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    device: str = "mps"
    dtype: str = "float32"
    seed: int = 42
    optimizer: str = "AdamW"
    gradient_clip: float = 1.0
    perception_learning_rate: float = 3e-4
    dynamics_learning_rate: float = 3e-4
    action_learning_rate: float = 1e-4
    evaluation_learning_rate: float = 1e-4
    batch_sizes: tuple[int, ...] = (4, 64, 64)
    max_epochs: tuple[int, ...] = (30, 40, 40)
    early_stopping_patience: int = 5
    confidence: str = "single_model_calibration"
    resumable_checkpoint: bool = True
    retain_curves_calibration_data_versions: bool = True


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    total_hours: float = 72.0
    throughput_pilot_hours: float = 2.0
    pilot_counts_toward_total: bool = True
    artifact_limit_gib: float = 45.0
    artifact_warning_gib: float = 40.0
    disk_min_free_gib: float = 25.0
    disk_warning_free_gib: float = 30.0
    include_temporary_and_other_process_usage: bool = True
    reserve_remaining_work: bool = True
    auto_delete_existing: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    validation_parent_layouts: int = 60
    calibration_parent_layouts: int = 60
    starts_per_validation_calibration_parent: int = 1
    preflight_layouts: int = 12
    formal_layouts: int = 120
    formal_layouts_per_n: int = 20
    arms: tuple[str, ...] = EPISODE_ARMS
    attempts_per_arm_per_layout: int = 1
    filter_on_act_success: bool = False
    min_net_successes: int = 12
    min_sample_gain: float = 0.10
    mcnemar_alternative: str = "greater"
    mcnemar_exact: bool = True
    mcnemar_p_threshold: float = 0.05
    prediction_test_requires_main_gate: bool = True
    infrastructure_error_stops_batch: bool = True
    replace_failed_layouts: bool = False


@dataclass(frozen=True, slots=True)
class FrozenConfig:
    source_root: str = "/Users/huazhi_luke/Documents/Codex/2026-09-08/zhe/exports/pac_act_bundle_resnet_mps_20261017_r8"
    backbone: str = "resnet18"
    act_training_steps: int = 100000
    camera_views: tuple[str, ...] = (
        "corner_xminus_yminus", "corner_xminus_yplus", "corner_xplus_yminus",
        "corner_xplus_yplus", "wrist",
    )
    model_sha256: str = "ee197896b8b79778ba185ad1db07ea83ff24a77a7da34f5397876e1523f3dc5f"
    preprocessing_frozen: bool = True
    native_control_frozen: bool = True
    online_geometry_source: str = "vision_only"
    online_joint_or_wrench_inputs: bool = False


def _approved_value(value: Any, expected: Any, name: str) -> Any:
    if isinstance(expected, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(expected):
            raise ValueError(f"invalid {name}")
        return tuple(_approved_value(v, e, name) for v, e in zip(value, expected))
    if type(expected) is float:
        if type(value) not in (float, int) or not isfinite(value) or value != expected:
            raise ValueError(f"{name} differs from approved v1 value {expected!r}")
        return float(value)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} differs from approved v1 value {expected!r}")
    return value


@dataclass(frozen=True, slots=True)
class WAMConfig:
    timing: TimingConfig = field(default_factory=TimingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    frozen: FrozenConfig = field(default_factory=FrozenConfig)

    def __post_init__(self) -> None:
        for section in fields(self):
            expected = section.default_factory()
            actual = getattr(self, section.name)
            if type(actual) is not type(expected):
                raise TypeError(f"invalid {section.name} section")
            for item in fields(expected):
                _approved_value(getattr(actual, item.name), getattr(expected, item.name),
                                f"{section.name}.{item.name}")

    @classmethod
    def load(cls, path: str | Path) -> WAMConfig:
        # Reject duplicate keys too: YAML's default last-key-wins is not a contract.
        class UniqueLoader(yaml.SafeLoader):
            pass

        def mapping(loader: UniqueLoader, node: yaml.MappingNode) -> dict:
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=True)
                if not isinstance(key, str) or key in result:
                    raise ValueError("config contains a duplicate or non-string key")
                result[key] = loader.construct_object(value_node, deep=True)
            return result

        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        with Path(path).open(encoding="utf-8") as source:
            document = yaml.load(source, Loader=UniqueLoader)
        sections = fields(cls)
        if not isinstance(document, Mapping) or set(document) != {s.name for s in sections}:
            raise ValueError("config must contain exactly the approved sections")
        values = {}
        for section in sections:
            expected = section.default_factory()
            raw = document[section.name]
            if not isinstance(raw, Mapping) or set(raw) != {item.name for item in fields(expected)}:
                raise ValueError(f"missing or unknown fields in {section.name}")
            normalized = {item.name: _approved_value(raw[item.name], getattr(expected, item.name),
                          f"{section.name}.{item.name}") for item in fields(expected)}
            values[section.name] = type(expected)(**normalized)
        return cls(**values)
