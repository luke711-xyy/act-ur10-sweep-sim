"""Unit tests for metrics aggregation and reproducible layout sampling."""

import numpy as np
import pytest

from sim.config import load_config
from sim.controllers.hybrid import ControlRecord
from sim.environments.layout import (
    _sample_cluster, in_safe_workspace, in_target_region, sample_layout,
)
from sim.metrics import aggregate, compute_episode_metrics, paired_comparison


@pytest.fixture
def cfg():
    return load_config()


def synth_trace(n=200, dt=0.01, desired=3.0, measured=2.9, phase="SWEEP"):
    trace = []
    for i in range(n):
        trace.append(ControlRecord(
            t=i * dt, stroke=0, phase=phase,
            tcp_x=0.4 - 0.002 * i, tcp_y=0.1, tcp_z=0.0,
            cmd_x=0.4 - 0.002 * i, cmd_y=0.1, cmd_z=-0.001, cmd_yaw=0.0,
            z_nominal=0.0, delta_z=-0.001, force_desired=desired,
            force_raw=measured, force_filtered=measured, in_contact=measured > 0.8,
        ))
    return trace


def test_metrics_from_a_synthetic_trace():
    m = compute_episode_metrics(
        trace=synth_trace(), n_components=5, n_collected=4, n_pushed_out=1,
        n_strokes=2, sim_time=12.3, success=False, seed=7, planner="fixed",
        perception="ground_truth", geometry="hex_nut", failure_reason="budget",
    )
    assert m.collection_rate == pytest.approx(0.8)
    assert m.components_per_stroke == pytest.approx(2.0)
    assert m.path_length == pytest.approx(0.002 * 199, abs=1e-9)
    assert m.path_length_xy == pytest.approx(m.path_length)
    assert m.rms_force_error == pytest.approx(0.1, abs=1e-9)
    assert m.contact_loss_ratio == pytest.approx(0.0)
    assert m.peak_normal_force == pytest.approx(2.9)
    assert m.n_pushed_out == 1
    assert not m.success


def test_contact_loss_ratio_counts_lost_samples():
    trace = synth_trace(100, measured=2.9) + synth_trace(100, measured=0.0)
    m = compute_episode_metrics(
        trace=trace, n_components=1, n_collected=1, n_pushed_out=0, n_strokes=1,
        sim_time=2.0, success=True, seed=0, planner="fixed",
        perception="ground_truth", geometry="hex_nut",
    )
    assert m.contact_loss_ratio == pytest.approx(0.5)


def test_metrics_on_an_empty_trace_do_not_crash():
    m = compute_episode_metrics(
        trace=[], n_components=0, n_collected=0, n_pushed_out=0, n_strokes=0,
        sim_time=0.0, success=False, seed=0, planner="fixed",
        perception="ground_truth", geometry="hex_nut",
    )
    assert m.path_length == 0.0 and m.collection_rate == 0.0
    assert np.isnan(m.rms_force_error)


def test_aggregate_and_paired_comparison():
    rows = [
        {"planner": "fixed", "n_components": 3, "seed": 0, "success": True,
         "collection_rate": 1.0, "completion_time": 40.0, "n_strokes": 10,
         "path_length": 8.0, "n_pushed_out": 0, "peak_normal_force": 3.2,
         "rms_force_error": 0.1, "contact_loss_ratio": 0.0, "components_per_stroke": 0.3},
        {"planner": "visual_greedy", "n_components": 3, "seed": 0, "success": True,
         "collection_rate": 1.0, "completion_time": 18.0, "n_strokes": 3,
         "path_length": 3.0, "n_pushed_out": 0, "peak_normal_force": 3.1,
         "rms_force_error": 0.1, "contact_loss_ratio": 0.0, "components_per_stroke": 1.0},
    ]
    summary = aggregate(rows)
    assert len(summary) == 2
    assert {row["planner"] for row in summary} == {"fixed", "visual_greedy"}
    assert all(row["episodes"] == 1 and row["success_rate"] == 1.0 for row in summary)

    pairs = paired_comparison(rows, "fixed", "visual_greedy", "completion_time")
    assert len(pairs) == 1 and pairs[0]["delta"] == pytest.approx(-22.0)


# ------------------------------------------------------------------ layout
def test_layout_is_reproducible_for_a_seed(cfg):
    cfg.set_path("components.count", 5)
    a = sample_layout(cfg, np.random.default_rng(3))
    b = sample_layout(cfg, np.random.default_rng(3))
    c = sample_layout(cfg, np.random.default_rng(4))
    assert [i["x"] for i in a] == [i["x"] for i in b]
    assert [i["x"] for i in a] != [i["x"] for i in c]


@pytest.mark.parametrize("distribution,count", [("legacy", 10), ("uniform_disk", 6)])
def test_layout_never_starts_inside_the_target(cfg, distribution, count):
    # Keep the historical ten-part fixture and the formal six-part ACT task.
    cfg.set_path("components.cluster.center_distribution", distribution)
    cfg.set_path("components.count", count)
    for seed in range(25):
        layout = sample_layout(cfg, np.random.default_rng(seed))
        xy = np.array([[i["x"], i["y"]] for i in layout])
        assert not in_target_region(xy, cfg.target).any()
        assert in_safe_workspace(xy, cfg.workspace).all()


def test_default_cluster_centre_stays_in_the_area_uniform_disk(cfg):
    """Only the latent group centre is in the disk, not each part offset."""
    centres = []
    for seed in [988808167, *range(512)]:
        centre, _ = _sample_cluster(cfg, np.random.default_rng(seed))
        centres.append(centre)
        assert np.linalg.norm(centre - [0.05, 0.0]) <= 0.10 + 1e-12
    assert np.ptp(np.asarray(centres)[:, 0]) > 0.18
    assert np.asarray(centres)[:, 0].min() < -0.04


def test_default_cluster_parts_can_extend_beyond_the_cluster_centre_limit(cfg):
    """Part offsets extend beyond centre support while respecting new bounds."""
    extends_past_centre_limit = False
    for seed in [988808167, *range(128)]:
        layout = sample_layout(cfg, np.random.default_rng(seed))
        assert layout == sample_layout(cfg, np.random.default_rng(seed))
        xy = np.array([[item["x"], item["y"]] for item in layout])
        assert len(layout) == 6
        assert np.all((-0.19 <= xy[:, 0]) & (xy[:, 0] <= 0.29))
        assert np.all((cfg.components.spawn.x_min <= xy[:, 0])
                      & (xy[:, 0] <= cfg.components.spawn.x_max))
        assert np.all(np.abs(xy[:, 1]) <= 0.26)
        distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() >= 0.07 - 1e-12
        extends_past_centre_limit |= bool(np.any(xy[:, 0] > 0.15))
    assert extends_past_centre_limit


@pytest.mark.parametrize("distribution,count", [("legacy", 10), ("uniform_disk", 6)])
def test_layout_respects_minimum_separation(cfg, distribution, count):
    cfg.set_path("components.cluster.center_distribution", distribution)
    cfg.set_path("components.count", count)
    layout = sample_layout(cfg, np.random.default_rng(1))
    xy = np.array([[i["x"], i["y"]] for i in layout])
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() >= float(cfg.components.min_separation) - 1e-9


def test_layout_randomises_mass_friction_and_orientation(cfg):
    cfg.set_path("components.count", 10)
    layout = sample_layout(cfg, np.random.default_rng(2))
    masses = [i["mass"] for i in layout]
    frictions = [i["friction"] for i in layout]
    assert len(set(masses)) > 1 and len(set(frictions)) > 1
    assert all(cfg.components.mass.min <= m <= cfg.components.mass.max for m in masses)
    assert all(cfg.components.friction.min <= f <= cfg.components.friction.max
               for f in frictions)
    assert len({round(i["yaw"], 6) for i in layout}) > 1


@pytest.mark.parametrize("count", [1, 2, 3, 5, 6, 10])
def test_all_required_component_counts_can_be_placed(cfg, count):
    cfg.set_path("components.count", count)
    assert len(sample_layout(cfg, np.random.default_rng(0))) == count


@pytest.mark.parametrize("geometry", ["hex_nut", "cylinder", "screw", "bolt", "washer", "mixed"])
def test_all_geometries_can_be_sampled(cfg, geometry):
    cfg.set_path("components.geometry", geometry)
    cfg.set_path("components.count", 3)
    layout = sample_layout(cfg, np.random.default_rng(0))
    assert all(item["half_height"] > 0 for item in layout)
