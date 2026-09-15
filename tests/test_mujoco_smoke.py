"""End-to-end smoke tests -- skipped automatically when MuJoCo is absent.

These are the tests that verify acceptance criteria 1, 2, 3, 4, 6 and 7 against
the real simulator.  Run them on a machine where ``pip install mujoco`` works.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco", reason="MuJoCo is not installed")

from sim.config import load_config                     # noqa: E402
from sim.controllers.state_machine import Phase        # noqa: E402
from sim.environments.episode import run_episode       # noqa: E402
from sim.environments.sweep_env import SweepEnv        # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    cfg = load_config()
    cfg.set_path("components.count", 2)
    cfg.set_path("sim.max_episode_time", 90.0)
    cfg.set_path("planner.max_strokes", 4)
    return cfg


def test_scene_compiles_and_steps(cfg):
    env = SweepEnv(cfg, seed=0)
    env.reset()
    assert env.model.nbody > 3
    assert env.decimation == 10
    z0 = env.tcp()[2]
    assert z0 == pytest.approx(float(cfg.end_effector.z_home), abs=2e-3)
    env.close()


def test_tips_contact_the_table_and_generate_a_bounded_force(cfg):
    """Acceptance criteria 2 and 3."""
    from sim.controllers.hybrid import Command

    env = SweepEnv(cfg, seed=1)
    env.reset()
    tcp = env.tcp()
    for z in np.linspace(float(tcp[2]), -0.002, 400):
        env.step_control(Command(float(tcp[0]), float(tcp[1]), float(z), 0.0))
    force = env.normal_force()
    assert force > 1.0, "pressing the closed tips into the table produced no force"
    assert force < 200.0
    env.close()


def test_single_component_episode_runs_and_is_reproducible(cfg):
    a = run_episode(cfg, seed=0, planner_name="visual_greedy",
                    perception_name="ground_truth")
    b = run_episode(cfg, seed=0, planner_name="visual_greedy",
                    perception_name="ground_truth")
    assert a.metrics.collection_rate == b.metrics.collection_rate
    assert a.metrics.n_strokes == b.metrics.n_strokes
    assert np.allclose(a.final_positions, b.final_positions, atol=1e-9)


def test_force_is_released_near_the_tray(cfg):
    """Acceptance criterion 4, in the real simulator."""
    result = run_episode(cfg, seed=2, planner_name="fixed", perception_name="ground_truth")
    line = float(cfg.controller.x_release_line)
    past = [r for r in result.trace if r.tcp_x <= line + 1e-3 and r.stroke >= 0]
    if past:
        assert max(r.force_desired for r in past) < 0.2


def test_vision_backend_produces_detections(cfg):
    env = SweepEnv(cfg, seed=3)
    env.reset()
    from sim.perception.vision import ConventionalVisionPerception

    perception = ConventionalVisionPerception(cfg)
    obs = perception.observe(env, np.random.default_rng(0))
    assert obs.rgb is not None and obs.rgb.shape[2] == 3
    assert obs.mask is not None and obs.mask.any()
    assert obs.n_detected >= 1
    truth = env.component_positions()[:, :2]
    for point in obs.points:
        assert np.min(np.linalg.norm(truth - point[None, :], axis=1)) < 0.05
    env.close()


def test_visual_planner_replans_between_strokes(cfg):
    """Acceptance criterion 6."""
    cfg2 = cfg.copy()
    cfg2.set_path("components.count", 3)
    cfg2.set_path("planner.max_strokes", 5)
    result = run_episode(cfg2, seed=5, planner_name="visual_greedy",
                         perception_name="ground_truth")
    assert len(result.observations) == len(result.strokes) or \
        len(result.observations) == len(result.strokes) + 1
    if len(result.strokes) >= 2:
        assert result.strokes[0].action.tolist() != result.strokes[1].action.tolist()


@pytest.mark.parametrize("planner", ["fixed", "global_sweep", "visual_greedy"])
def test_all_three_planners_run(cfg, planner):
    cfg2 = cfg.copy()
    cfg2.set_path("planner.max_strokes", 3)
    cfg2.set_path("sim.max_episode_time", 60.0)
    result = run_episode(cfg2, seed=11, planner_name=planner, perception_name="ground_truth")
    assert result.metrics.planner == planner
    assert result.metrics.peak_normal_force < float(cfg2.controller.safe_max_force)
    # the full state must reach the trace
    sweeping = [r for r in result.trace if r.phase == "SWEEP"]
    if sweeping:
        assert any(abs(r.fz) > 1e-6 for r in sweeping), "no wrench logged during SWEEP"


def test_clustered_layouts_run_in_the_simulator(cfg):
    cfg2 = cfg.copy()
    cfg2.set_path("components.spawn_mode", "cluster")
    cfg2.set_path("components.count", 4)
    cfg2.set_path("planner.max_strokes", 3)
    result = run_episode(cfg2, seed=4, planner_name="visual_greedy",
                         perception_name="ground_truth")
    xy = np.asarray(result.initial_positions)[:, :2]
    assert np.linalg.norm(xy - xy.mean(axis=0), axis=1).max() < 0.25


def test_dual_camera_recording_produces_a_file(cfg, tmp_path):
    from sim.video import EpisodeRecorder

    cfg2 = cfg.copy()
    cfg2.set_path("planner.max_strokes", 1)
    cfg2.set_path("sim.max_episode_time", 25.0)
    path = str(tmp_path / "episode.mp4")
    env = SweepEnv(cfg2, seed=7)
    env.reset(seed=7)
    recorder = EpisodeRecorder(cfg2, path, cameras=["scene_cam", "overhead_cam"],
                               every_n=10)
    recorder.attach(env)
    try:
        run_episode(cfg2, seed=7, planner_name="visual_greedy",
                    perception_name="ground_truth", env=env, recorder=recorder)
    finally:
        written = recorder.close()
        env.close()
    import os

    assert recorder.n_frames > 5
    assert os.path.getsize(written) > 1000


def test_unknown_camera_is_rejected(cfg, tmp_path):
    from sim.video import EpisodeRecorder

    env = SweepEnv(cfg, seed=0)
    env.reset()
    recorder = EpisodeRecorder(cfg, str(tmp_path / "x.mp4"), cameras=["not_a_camera"])
    with pytest.raises(ValueError):
        recorder.attach(env)
    env.close()


def test_rrt_transfer_mode_runs_end_to_end(cfg):
    cfg2 = cfg.copy()
    cfg2.set_path("planner.transfer.mode", "rrt")
    cfg2.set_path("planner.max_strokes", 2)
    cfg2.set_path("sim.max_episode_time", 60.0)
    result = run_episode(cfg2, seed=9, planner_name="visual_greedy",
                         perception_name="ground_truth")
    assert result.metrics.peak_normal_force < float(cfg2.controller.safe_max_force)
    # the sweep itself must still be a straight segment
    for stroke in result.strokes:
        assert stroke.action.shape == (5,)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_episodes_run_for_several_component_counts(cfg, count):
    """Acceptance criterion 7 (the 5 and 10 cases are slower; see run_experiment)."""
    cfg2 = cfg.copy()
    cfg2.set_path("components.count", count)
    cfg2.set_path("planner.max_strokes", 3)
    cfg2.set_path("sim.max_episode_time", 60.0)
    result = run_episode(cfg2, seed=count, planner_name="visual_greedy",
                         perception_name="ground_truth")
    assert result.metrics.n_components == count
    assert 0.0 <= result.metrics.collection_rate <= 1.0
    assert result.metrics.peak_normal_force < float(cfg2.controller.safe_max_force)
