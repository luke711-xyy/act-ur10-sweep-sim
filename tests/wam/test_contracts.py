"""Task 1 boundaries: handwritten physical features, never simulator truth."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from wam_sweep.config import WAMConfig
from wam_sweep.contracts import (
    CandidateProgram, EpisodeResult, EpisodeSpec, HistoryBatch, InputIdentity,
    ModelVersion, ObjectBelief, OnlineObservation, ReferencePreview, RobotContext,
    RunIdentity, TaskScore, WorldPrediction, supervision,
)
from wam_sweep.history import HistoryBuffer


def belief(track_id=0, **changes):
    values = dict(
        center_xy=[0.1, 0.2], velocity_xy=[0.3, 0.4], half_extents_xy=[0.01, 0.02],
        yaw=0.0, angular_velocity=0.5, shape_probabilities=[0.25, 0.75],
        visibility=0.8, state_probabilities=[0.2, 0.3, 0.5],
        position_covariance=[[0.04, 0.01], [0.01, 0.09]],
        appearance=list(range(16)), track_id=track_id, valid=True,
    )
    return ObjectBelief(**(values | changes))


def robot(**changes):
    values = dict(
        tcp_xyz=[0.6, 0.7, 0.8], yaw=0.0, linear_velocity=[0.9, 1.0, 1.1],
        yaw_velocity=1.2, phase="sweep", residual=[0.01, 0.02, 0.03],
        residual_rate=[0.04, 0.05, 0.06], episode_time=0.0,
        contact_ready=True, reference_valid=True,
        visual_entry_age=0.0, visual_entry_valid=False,
    )
    return RobotContext(**(values | changes))


def observation(t=0.0, objects=None, **changes):
    images = {name: np.zeros((2, 3, 3), dtype=np.uint8) for name in (
        "corner_xminus_yminus", "corner_xminus_yplus", "corner_xplus_yminus",
        "corner_xplus_yplus", "wrist",
    )}
    values = dict(timestamp=t, objects=tuple([belief(i) for i in range(6)] if objects is None else objects),
                  robot=robot(episode_time=t), images=images)
    return OnlineObservation(**(values | changes))


def test_startup_history_is_masked():
    # Catches duplicated startup frames and a wrong packing width/order.
    history = HistoryBuffer()
    history.append(observation())
    batch = history.batch()
    assert batch.history_valid.sum().item() == 1
    assert tuple(batch.objects.shape) == (1, 11, 6, 34)
    assert tuple(batch.brush.shape) == (1, 11, 9)
    assert tuple(batch.context.shape) == (1, 11, 15)
    assert tuple(batch.object_valid.shape) == (1, 11, 6)
    assert batch.history_valid.tolist() == [[0.0] * 10 + [1.0]]
    assert torch.count_nonzero(batch.objects[:, :10]) == 0
    assert torch.count_nonzero(batch.object_valid[:, :10]) == 0
    expected = [0.1, 0.2, 0.3, 0.4, 0.01, 0.02, 0.0, 1.0, 0.5,
                0.25, 0.75, 0.8, 0.2, 0.3, 0.5, 0.04, 0.01, 0.09, *range(16)]
    torch.testing.assert_close(batch.objects[0, -1, 0], torch.tensor(expected))
    torch.testing.assert_close(batch.brush[0, -1], torch.tensor(
        [0.6, 0.7, 0.8, 0.0, 1.0, 0.9, 1.0, 1.1, 1.2]))
    torch.testing.assert_close(batch.context[0, -1], torch.tensor(
        [0., 0., 1., 0., .01, .02, .03, .04, .05, .06, 0., 1., 1., 0., 0.]))
    assert all(t.dtype == torch.float32 for t in (
        batch.objects, batch.brush, batch.context, batch.object_valid, batch.history_valid))


def test_two_seconds_has_eleven_frames_then_rolls_forward():
    history = HistoryBuffer()
    for step in range(11):
        history.append(observation(step / 5))
    assert history.batch().history_valid.sum().item() == 11
    history.append(observation(2.2))
    torch.testing.assert_close(history.batch().context[0, :, 10],
                               torch.tensor([.2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2., 2.2]))


def test_missing_sample_is_masked_without_fabricating_history():
    history = HistoryBuffer()
    history.append(observation())
    history.append(observation(.4))
    assert history.batch().history_valid[0, -3:].tolist() == [1., 0., 1.]
    history.append(observation(3.0))
    assert history.batch().history_valid.sum().item() == 1


@pytest.mark.parametrize("time", [-.2, 0.0, .1, float("nan"), float("inf")])
def test_invalid_history_time_is_rejected_atomically(time):
    history = HistoryBuffer()
    history.append(observation())
    before = history.batch()
    with pytest.raises((TypeError, ValueError)):
        history.append(observation(time))
    torch.testing.assert_close(history.batch().objects, before.objects)
    torch.testing.assert_close(history.batch().history_valid, before.history_valid)


@pytest.mark.parametrize("field", ["mass", "friction", "true_center", "true_radius",
    "true_type", "collision_boundary", "collected_mask", "first_entry_time",
    "joint_positions", "joint_velocities", "wrench", "true_count", "target_count"])
@pytest.mark.parametrize("constructor", [belief, robot, observation])
def test_online_contract_rejects_extra_truth_joint_wrench_and_goal_fields(field, constructor):
    with pytest.raises((TypeError, ValueError)):
        constructor(**{field: 123.0})


def test_isolated_truth_cannot_affect_packing_with_fixed_visual_inputs():
    frame = observation()
    truth = supervision.RawTruth(fields={"mass": 1., "friction": .5, "centers": [[1., 2.]],
        "collected_mask": [False], "first_entry_time": None, "wrench": [0.] * 6})
    history = HistoryBuffer()
    history.append(frame)
    before = history.batch()
    truth.fields.update(mass=1e9, friction=-123., centers=[[float("nan"), 900.]],
                        collected_mask=[True], first_entry_time=-900., wrench=[1e9] * 6)
    other = HistoryBuffer()
    other.append(frame)
    torch.testing.assert_close(other.batch().objects, before.objects, rtol=0, atol=0)
    torch.testing.assert_close(other.batch().brush, before.brush, rtol=0, atol=0)
    torch.testing.assert_close(other.batch().context, before.context, rtol=0, atol=0)
    with pytest.raises(TypeError):
        other.append(truth)


@pytest.mark.parametrize("changes", [
    {"center_xy": [float("nan"), 0.]}, {"appearance": [0.] * 15},
    {"half_extents_xy": [-1., 1.]}, {"visibility": 1.1},
    {"shape_probabilities": [.2, .2]}, {"state_probabilities": [-.1, .2, .9]},
    {"position_covariance": [[1., 2.], [2., 1.]]},
    {"position_covariance": [[1., .1], [.2, 1.]]}, {"track_id": True},
])
def test_malformed_beliefs_are_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        belief(**changes)


def test_robot_and_image_finite_shape_and_provenance_validation():
    with pytest.raises(ValueError):
        robot(yaw=float("inf"))
    with pytest.raises(ValueError):
        robot(phase="true_contact")
    with pytest.raises(ValueError):
        robot(visual_entry_age=-1.)
    with pytest.raises(ValueError):
        observation(images={"wrist": np.zeros((2, 3, 3), dtype=np.uint8)})
    with pytest.raises(ValueError):
        observation(objects=[belief(0), belief(0)])


def test_track_permutation_is_associated_and_ids_are_not_features():
    a, b = belief(10, center_xy=[1., 2.]), belief(20, center_xy=[3., 4.])
    history = HistoryBuffer()
    history.append(observation(objects=[b, a]))
    history.append(observation(.2, objects=[a, b]))
    assert history.batch().objects[0, -2:, :2, 0].tolist() == [[1., 3.], [1., 3.]]
    assert history.batch().object_valid[0, -1].tolist() == [1., 1., 0., 0., 0., 0.]
    relabeled = HistoryBuffer()
    relabeled.append(observation(objects=[replace(a, track_id=100), replace(b, track_id=200)]))
    torch.testing.assert_close(relabeled.batch().objects[0, -1], history.batch().objects[0, -1])


def test_missing_and_invalid_tracks_do_not_shift_other_slots():
    history = HistoryBuffer()
    history.append(observation(objects=[belief(10), belief(20, center_xy=[3., 4.])]))
    history.append(observation(.2, objects=[belief(20, center_xy=[5., 6.])]))
    assert history.batch().object_valid[0, -1].tolist() == [0., 1., 0., 0., 0., 0.]
    assert history.batch().objects[0, -1, 1, 0].item() == 5.
    history.append(observation(.4, objects=[belief(20, valid=False)]))
    assert history.batch().object_valid[0, -1].sum().item() == 0
    assert history.batch().objects[0, -1].count_nonzero().item() == 0


def test_rectangle_pi_and_brush_two_pi_periodicity():
    def packed(object_yaw, brush_yaw):
        history = HistoryBuffer()
        history.append(observation(objects=[belief(yaw=object_yaw)], robot=robot(yaw=brush_yaw)))
        return history.batch()
    a, b = packed(.3, .4), packed(.3 + np.pi, .4 + 2 * np.pi)
    torch.testing.assert_close(a.objects, b.objects)
    torch.testing.assert_close(a.brush, b.brush)
    c = packed(.3, .4 + np.pi)
    assert not torch.allclose(a.brush, c.brush)


def test_snapshot_restore_and_batches_are_deeply_detached_and_restore_is_atomic():
    history = HistoryBuffer()
    frame = observation()
    history.append(frame)
    frame.objects[0].center_xy[0] = 700.
    assert history.batch().objects[0, -1, 0, 0].item() == pytest.approx(.1)
    state = history.snapshot()
    restored = HistoryBuffer()
    restored.restore(state)
    state["observations"][0].objects[0].center_xy[0] = 800.
    assert restored.batch().objects[0, -1, 0, 0].item() == pytest.approx(.1)
    output = restored.batch()
    output.objects.fill_(999.)
    assert restored.batch().objects[0, -1, 0, 0].item() == pytest.approx(.1)
    bad = history.snapshot()
    bad["observations"].append(observation(0.0))
    with pytest.raises(ValueError):
        restored.restore(bad)
    assert restored.batch().history_valid.sum().item() == 1
    restored.append(observation(.2))
    assert history.batch().history_valid.sum().item() == 1


def test_empty_batch_and_mutated_input_revalidation():
    history = HistoryBuffer()
    assert history.batch().history_valid.sum().item() == 0
    frame = observation()
    frame.objects[0].center_xy[0] = float("nan")
    with pytest.raises(ValueError):
        history.append(frame)
    assert history.batch().history_valid.sum().item() == 0


@pytest.mark.parametrize("make", [
    lambda: belief(center_xy=[1e100, 0.]),
    lambda: robot(residual=[1e100, 0., 0.]),
    lambda: robot(episode_time=1e100),
])
def test_finite_numbers_that_overflow_network_fp32_are_rejected(make):
    # A float64-finite payload must not poison the buffer before FP32 packing.
    with pytest.raises(ValueError):
        make()


def test_no_detections_and_restored_rolling_history_keep_clock_and_track_slots():
    history = HistoryBuffer()
    for i in range(16):
        history.append(observation(i / 5))
    restored = HistoryBuffer()
    restored.restore(history.snapshot())
    restored.append(observation(3.2, objects=[]))
    batch = restored.batch()
    assert batch.history_valid.sum().item() == 11
    assert batch.object_valid[0, -1].sum().item() == 0
    assert batch.objects[0, -1].count_nonzero().item() == 0
    assert batch.context[0, -1, 10].item() == pytest.approx(3.2)


def test_program_preview_and_score_reject_malformed_or_future_issuance():
    CandidateProgram(knots=np.zeros((5, 3)))
    with pytest.raises(ValueError):
        CandidateProgram(knots=np.ones((5, 3)) * 1.01)
    with pytest.raises(ValueError):
        CandidateProgram(knots=np.zeros((25, 3)))
    times = np.arange(1, 26) / 25
    values = dict(targets=np.zeros((25, 4)), times=times, valid=np.ones(25),
                  issued_at=0., source_hash="native-sha")
    ReferencePreview(**values)
    for changes in ({"times": times[::-1]}, {"issued_at": .1}, {"source_hash": ""},
                    {"targets": np.full((25, 4), np.nan)}, {"valid": np.ones(25) * .5}):
        with pytest.raises(ValueError):
            ReferencePreview(**(values | changes))
    TaskScore(success_logit=torch.zeros(2, 1), final_count_logits=torch.zeros(2, 7),
              calibration_hash="calibration")
    with pytest.raises(ValueError):
        TaskScore(success_logit=torch.zeros(2, 1), final_count_logits=torch.zeros(2, 6),
                  calibration_hash="calibration")


def test_world_prediction_shapes_and_finite_values():
    values = dict(object_mean=torch.zeros(2, 25, 6, 9), object_log_std=torch.zeros(2, 25, 6, 4),
        state_logits=torch.zeros(2, 25, 6, 3), shape_logits=torch.zeros(2, 25, 6, 2),
        visibility_logits=torch.zeros(2, 25, 6, 1), brush=torch.zeros(2, 25, 9),
        brush_log_std=torch.zeros(2, 25, 9), valid=torch.ones(2, 25))
    WorldPrediction(**values)
    for changes in ({"object_mean": torch.zeros(2, 25, 6, 10)},
                    {"object_log_std": torch.full((2, 25, 6, 4), float("inf"))},
                    {"valid": torch.ones(2, 25) * .3}, {"brush": torch.zeros(2, 25, 9).double()}):
        with pytest.raises((TypeError, ValueError)):
            WorldPrediction(**(values | changes))
    with pytest.raises(ValueError):
        HistoryBatch(objects=torch.zeros(1, 11, 6, 34), brush=torch.zeros(1, 11, 9),
            context=torch.zeros(1, 11, 15), object_valid=torch.ones(1, 11, 6),
            history_valid=torch.zeros(1, 11))


def test_prediction_public_mean_matches_task9_consumer():
    prediction = WorldPrediction(
        object_mean=torch.zeros(1, 25, 6, 9), object_log_std=torch.zeros(1, 25, 6, 4),
        state_logits=torch.zeros(1, 25, 6, 3), shape_logits=torch.zeros(1, 25, 6, 2),
        visibility_logits=torch.zeros(1, 25, 6, 1), brush=torch.zeros(1, 25, 9),
        brush_log_std=torch.zeros(1, 25, 9), valid=torch.ones(1, 25),
    )
    assert prediction.object_mean.shape == (1, 25, 6, 9)


def test_canonical_episode_arms_match_task12_cli():
    path = Path(__file__).resolve().parents[2] / "configs/wam_visual_v1.yaml"
    assert WAMConfig.load(path).evaluation.arms == ("act", "wam", "no-prediction")
    for arm in ("act", "wam", "no-prediction"):
        EpisodeSpec("parent-1", 42, 3, "evaluation", 2, arm, 0)


@pytest.mark.parametrize("arm", ["no_future_prediction", "no_prediction", "unknown"])
def test_noncanonical_episode_arms_are_rejected(arm):
    with pytest.raises(ValueError):
        EpisodeSpec("parent-1", 42, 3, "evaluation", 2, arm, 0)


def test_reference_command_clock_includes_root_time_before_resulting_state():
    # Commands begin at root t; corresponding future states begin at t + 0.04.
    preview = ReferencePreview(targets=np.zeros((25, 4)), times=2. + np.arange(25) * .04,
                               valid=np.ones(25), issued_at=2., source_hash="native-sha")
    assert preview.times[0] == 2.
    assert preview.times[-1] == pytest.approx(2.96)


@pytest.mark.parametrize("kind", ["pre_issuance", "off_cadence", "off_grid"])
def test_reference_command_clock_rejects_pre_issuance_and_off_cadence(kind):
    times = 2. + np.arange(25) * .04
    if kind == "pre_issuance":
        times -= .04
    elif kind == "off_cadence":
        times += .04
        times[10] += .01
    else:
        times += .01
    with pytest.raises(ValueError):
        ReferencePreview(targets=np.zeros((25, 4)), times=times,
                         valid=np.ones(25), issued_at=2., source_hash="native-sha")


def test_identity_goal_versions_and_result_categories():
    spec = EpisodeSpec("parent-1", 42, 6, "train", 0, "act", 0)
    assert spec.target_count == 6
    for goal in (0, 7, 1.5, True):
        with pytest.raises((TypeError, ValueError)):
            replace(spec, target_count=goal)
    identity = InputIdentity("/frozen", "manifest", "model",
                             {"perception": "p", "control": "c", "preprocessing": "x", "config": "g"})
    RunIdentity("run", "spec", "plan", "config", identity, "split")
    with pytest.raises(ValueError):
        replace(identity, native_hashes={"control": ""})
    base = ModelVersion(-1, "act", "/frozen/actor", "m", "p", "d", "c", identity)
    replace(base, phase=0, kind="wam")
    with pytest.raises(ValueError):
        replace(base, phase=0)
    with pytest.raises(ValueError):
        replace(base, perception_hash="")
    for status in ("success", "task_failure", "perception_invalid", "model_invalid", "safety_stop",
                   "infrastructure_error", "incomplete"):
        result = EpisodeResult(status=status, native_reason="native-reason")
        assert result.success is (status == "success")
    with pytest.raises(ValueError):
        EpisodeResult(status="done", native_reason="native-reason")


def test_config_loads_complete_approved_defaults_and_rejects_drift(tmp_path):
    path = Path(__file__).resolve().parents[2] / "configs/wam_visual_v1.yaml"
    cfg = WAMConfig.load(path)
    assert cfg.timing.history_frames == 11
    assert cfg.model.graph_hidden == 128
    assert cfg.model.message_passing_layers == 3
    assert cfg.model.proposal_hidden == (256, 256)
    assert cfg.model.proposal_output == 15
    assert cfg.model.evaluation_hidden == (256, 128)
    assert cfg.data.main_trajectories == (240, 120, 120)
    assert cfg.data.teacher_starts == (600, 300, 300)
    assert cfg.training.batch_sizes == (4, 64, 64)
    assert cfg.training.max_epochs == (30, 40, 40)
    assert cfg.budget.total_hours == 72
    assert cfg.budget.artifact_limit_gib == 45
    assert cfg.budget.disk_min_free_gib == 25
    assert cfg.evaluation.formal_layouts * len(cfg.evaluation.arms) == 360
    assert cfg.evaluation.min_net_successes == 12
    document = yaml.safe_load(path.read_text())
    for section, key, value in (("timing", "history_frames", 10),
        ("training", "learner_warmup_frames", 50), ("model", "ensemble_size", 2),
        ("budget", "artifact_limit_gib", float("nan")), ("data", "teacher_starts", [1, 2]),
        ("training", "batch_sizes", [True, 64, 64])):
        changed = {k: dict(v) for k, v in document.items()}
        changed[section][key] = value
        candidate = tmp_path / "bad.yaml"
        candidate.write_text(yaml.safe_dump(changed))
        with pytest.raises((TypeError, ValueError)):
            WAMConfig.load(candidate)
