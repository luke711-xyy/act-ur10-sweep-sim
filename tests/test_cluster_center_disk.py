"""Disk-centre contract, unchanged part constraints and legacy seed fixtures."""
import hashlib
import json

import numpy as np
import pytest

from sim.config import Config, load_config
from sim.environments.layout import _sample_cluster, sample_layout


@pytest.fixture
def disk_cfg():
    cfg = load_config()
    cfg.set_path("components.cluster.center_distribution", "uniform_disk")
    cfg.set_path("components.cluster.center_disk", {
        "center": [0.05, 0.0], "radius": 0.10,
    })
    return cfg


def test_disk_draw_uses_area_radius_and_preserves_spread_draw(disk_cfg):
    expected_rng = np.random.default_rng(1234)
    theta = expected_rng.uniform(0.0, 2.0 * np.pi)
    radius = 0.10 * np.sqrt(expected_rng.uniform(0.0, 1.0))
    expected = np.array([0.05, 0.0]) + radius * np.array([np.cos(theta), np.sin(theta)])
    expected_spread = expected_rng.uniform(0.065, 0.085)
    rng = np.random.default_rng(1234)
    centre, spread = _sample_cluster(disk_cfg, rng)
    np.testing.assert_array_equal(centre, expected)
    assert spread == expected_spread
    assert rng.bit_generator.state == expected_rng.bit_generator.state


def test_disk_is_uniform_in_area_and_angle_and_reaches_negative_x(disk_cfg):
    rng = np.random.default_rng(9012026)
    centres = np.array([_sample_cluster(disk_cfg, rng)[0] for _ in range(8000)])
    offsets = (centres - [0.05, 0.0]) / 0.10
    radius_squared = np.sum(offsets ** 2, axis=1)
    assert np.max(radius_squared) <= 1.0 + 1e-12
    assert radius_squared.mean() == pytest.approx(0.5, abs=0.015)
    for values in (radius_squared, np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2 * np.pi) / (2 * np.pi)):
        frequencies = np.histogram(values, bins=np.linspace(0.0, 1.0, 9))[0] / len(values)
        assert np.max(np.abs(frequencies - 0.125)) < 0.025
    np.testing.assert_allclose(centres.mean(axis=0), [0.05, 0.0], atol=0.002)
    assert centres[:, 0].min() < -0.045
    assert centres[:, 0].max() > 0.145
    assert centres[:, 1].min() < -0.095
    assert centres[:, 1].max() > 0.095


def test_disk_ignores_legacy_bounds_and_radius_reserved_safe_bounds(disk_cfg):
    # These old bounds deliberately exclude nearly the entire disk.  They may
    # neither alter draws nor shrink the new mode's support.
    alternate = disk_cfg.copy()
    alternate.set_path("components.cluster.center_bounds", {
        "x_min": 0.30, "x_max": 0.31, "y_min": 0.20, "y_max": 0.21,
    })
    alternate.set_path("components.cluster.center_mean", [9.0, 9.0])
    alternate.set_path("components.cluster.max_radius", 0.50)
    for seed in range(128):
        centre, spread = _sample_cluster(disk_cfg, np.random.default_rng(seed))
        other, other_spread = _sample_cluster(alternate, np.random.default_rng(seed))
        np.testing.assert_array_equal(centre, other)
        assert spread == other_spread


@pytest.mark.parametrize("disk", [
    {}, {"center": [0.05, 0.0]}, {"radius": 0.10},
    {"center": [0.05], "radius": 0.10},
    {"center": [[0.05, 0.0]], "radius": 0.10},
    {"center": [float("nan"), 0.0], "radius": 0.10},
    {"center": [float("inf"), 0.0], "radius": 0.10},
    {"center": [0.05, 0.0], "radius": 0.0},
    {"center": [0.05, 0.0], "radius": -0.10},
    {"center": [0.05, 0.0], "radius": float("nan")},
    {"center": [0.05, 0.0], "radius": float("inf")},
    {"center": [0.05, 0.0], "radius": None},
    {"center": [0.05, 0.0], "radius": "not-a-radius"},
])
def test_invalid_disk_configuration_fails_explicitly(disk_cfg, disk):
    disk_cfg.set_path("components.cluster.center_disk", disk)
    with pytest.raises(ValueError, match="center_disk"):
        _sample_cluster(disk_cfg, np.random.default_rng(0))


def test_unknown_centre_distribution_fails_explicitly(disk_cfg):
    disk_cfg.set_path("components.cluster.center_distribution", "typo")
    with pytest.raises(ValueError, match="center_distribution"):
        _sample_cluster(disk_cfg, np.random.default_rng(0))


@pytest.mark.parametrize("path,value", [
    ("components.spawn.x_min", -0.04),
    ("components.spawn.x_max", 0.14),
    ("components.spawn.y_min", -0.09),
    ("components.spawn.y_max", 0.09),
    ("table.half_size", [0.14, 0.40, 0.02]),
])
def test_disk_outside_actual_domain_errors_instead_of_clipping(disk_cfg, path, value):
    disk_cfg.set_path(path, value)
    with pytest.raises(ValueError, match="center_disk"):
        _sample_cluster(disk_cfg, np.random.default_rng(0))


@pytest.mark.parametrize("path,value", [
    ("components.spawn.x_min", -0.05),
    ("components.spawn.x_max", 0.15),
    ("components.spawn.y_min", -0.10),
    ("components.spawn.y_max", 0.10),
    ("table.half_size", [0.15, 0.10, 0.02]),
])
def test_disk_tangent_to_actual_domain_is_not_rejected_or_changed(disk_cfg, path, value):
    expected, expected_spread = _sample_cluster(disk_cfg, np.random.default_rng(0))
    disk_cfg.set_path(path, value)
    centre, spread = _sample_cluster(disk_cfg, np.random.default_rng(0))
    np.testing.assert_array_equal(centre, expected)
    assert spread == expected_spread


def test_disk_tolerance_does_not_accept_a_meaningfully_outside_disk(disk_cfg):
    disk_cfg.set_path("components.spawn.x_max", 0.15 - 1e-10)
    with pytest.raises(ValueError, match="center_disk"):
        _sample_cluster(disk_cfg, np.random.default_rng(0))


@pytest.mark.parametrize("seed,layout_sha,rng_state", [
    (2110000, "c43567f0a858269e4877fb54d607ae2ed9c8e06987a729f46aabe55a9720219f", 291359010984249144285658645206492692398),
    (2110081, "3b13526772a8b3ecd2a7941f8d10327712e626617356cf6a3a716675ed5a75a2", 229540370527038970933195243019799676259),
    (2110119, "ab46d13b0129147d22ba14a67a934f9b8905afc7eecfc82c4a8e203f6fc5faca", 226546037652942716590755078935638337008),
])
def test_missing_distribution_reproduces_prechange_gaussian_layout_and_rng(seed, layout_sha, rng_state):
    # Fingerprints belong to the frozen pre-disk layout, not mutable defaults.
    data = load_config(overrides=[
        "components.spawn.x_min=-0.08", "components.cluster.max_radius=0.16",
        "components.cluster.max_lateral_radius=0.16",
    ]).to_dict()
    data["components"]["cluster"].pop("center_distribution", None)
    data["components"]["cluster"].pop("center_disk", None)
    rng = np.random.default_rng(seed)
    layout = sample_layout(Config(data), rng)
    assert hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest() == layout_sha
    assert rng.bit_generator.state["state"]["state"] == rng_state


@pytest.mark.parametrize("seed,layout_sha,rng_state", [
    (3, "81df433ccace4ab79d4ce5682cd5ff5221ebcd118589af081cc9e868e8d40d67", 307113002855941470047366908114287541706),
    (4, "f6672113fe863cf75c0965b8d58bdcd0c2562dbd50a4699faa323a2bb73974c9", 198776449027563167290916423695644439988),
])
def test_explicit_legacy_reproduces_prechange_uniform_fallback(seed, layout_sha, rng_state):
    data = load_config(overrides=[
        "components.spawn.x_min=-0.08", "components.cluster.max_radius=0.16",
        "components.cluster.max_lateral_radius=0.16",
    ]).to_dict()
    data["components"]["cluster"]["center_distribution"] = "legacy"
    data["components"]["cluster"].pop("center_mean")
    rng = np.random.default_rng(seed)
    layout = sample_layout(Config(data), rng)
    assert hashlib.sha256(json.dumps(layout, sort_keys=True).encode()).hexdigest() == layout_sha
    assert rng.bit_generator.state["state"]["state"] == rng_state


def test_new_layouts_keep_configured_part_constraints_and_seed_repeatability(disk_cfg):
    for seed in range(128):
        layout = sample_layout(disk_cfg, np.random.default_rng(seed))
        assert layout == sample_layout(disk_cfg, np.random.default_rng(seed))
        xy = np.array([[part["x"], part["y"]] for part in layout])
        assert len(layout) == 6
        assert np.all((disk_cfg.components.spawn.x_min <= xy[:, 0])
                      & (xy[:, 0] <= disk_cfg.components.spawn.x_max))
        assert np.all(np.abs(xy[:, 1]) <= 0.26)
        distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() >= 0.070 - 1e-12
        assert not np.any((xy[:, 0] >= -0.47) & (xy[:, 0] <= -0.23) & (np.abs(xy[:, 1]) <= 0.23))


@pytest.mark.parametrize("target", range(1, 7))
def test_default_demonstration_config_uses_the_new_disk(target):
    from sim.act.collect_dataset import _expert_config
    cfg = _expert_config(load_config(), target)
    assert cfg.components.spawn_mode == "cluster"
    assert cfg.components.cluster.center_distribution == "uniform_disk"
    assert cfg.components.cluster.center_disk.to_dict() == {"center": [0.05, 0.0], "radius": 0.10}
    assert len(sample_layout(cfg, np.random.default_rng(37))) == 6


class _StopBeforeRollout(Exception):
    """Stop at the physical boundary, without inference/data/media side effects."""


def _check_default_consumer(cfg, seed, **kwargs):
    assert cfg.components.spawn_mode == "cluster"
    assert cfg.components.cluster.center_distribution == "uniform_disk"
    expected = sample_layout(load_config(), np.random.default_rng(seed))
    assert sample_layout(cfg, np.random.default_rng(seed)) == expected
    raise _StopBeforeRollout


@pytest.mark.parametrize("target", range(1, 7))
def test_existing_inference_cli_preserves_the_new_default(target, monkeypatch):
    from sim.act import evaluate
    monkeypatch.setattr(evaluate, "run_act_episode", _check_default_consumer)
    with pytest.raises(_StopBeforeRollout):
        evaluate._run_main(["--target-count", str(target), "--seed", "37"])


@pytest.mark.parametrize("target", range(1, 7))
def test_existing_workbench_preview_preserves_the_new_default(target, tmp_path, monkeypatch):
    from sim.web import workbench
    monkeypatch.setattr(workbench, "run_expert_episode", _check_default_consumer)
    state = workbench.WorkbenchState(load_config(), tmp_path / "dataset", tmp_path / "preview")
    with pytest.raises(_StopBeforeRollout):
        state.build_preview(seed=37, target_count=target, max_attempts=1)
    assert not (tmp_path / "preview").exists()
