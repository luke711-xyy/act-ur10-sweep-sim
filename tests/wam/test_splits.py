from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest

from wam_sweep.config import WAMConfig
from wam_sweep.contracts import InputIdentity


@pytest.fixture
def api():
    assert importlib.util.find_spec("wam_sweep.splits"), "Task2 splits module is absent"
    from wam_sweep import artifacts, splits
    return artifacts, splits


@pytest.fixture
def exclusions():
    return {"complete_known_inputs": True, "seeds": [42], "fingerprints": [],
            "reserved_seed_ranges": [[5000000, 6000000]]}


def budget(api, tmp_path):
    return api[0].BudgetLedger(tmp_path / "budget.json", owned_roots=[tmp_path],
                               config=WAMConfig().budget,
                               disk_probe=lambda _: SimpleNamespace(free=100 * 2**30))


def identity():
    return InputIdentity("/verified/snapshot", "a" * 64, "b" * 64,
                         {k: "c" * 64 for k in ("perception", "control", "preprocessing", "config", "source/sim/environments/layout.py")})


def pose(record):
    # Hand-written six settled-pose fixture, no generator/model/physics execution.
    return [[(record.seed - 90000000) / 1000, 0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0],
            [0, 2, 0, 1, 0, 0, 0], [0, 3, 0, 1, 0, 0, 0],
            [0, 4, 0, 1, 0, 0, 0], [0, 5, 0, 1, 0, 0, 0]]


def test_predetermined_852_parents_quotas_seeds_and_provisional_gate(api, exclusions):
    splits = api[1]
    manifest = splits.make_split_manifest(WAMConfig(), exclusions)
    assert len(manifest.records) == 852
    assert len({r.parent_id for r in manifest.records}) == 852
    expected = {("train", 0): (240, 90000000), ("train", 1): (120, 90001000),
                ("train", 2): (120, 90002000), ("bootstrap", -1): (120, 90010000),
                ("validation", -1): (60, 90020000), ("calibration", -1): (60, 90030000),
                ("preflight", -1): (12, 90040000), ("test", -1): (120, 90050000)}
    for key, (count, first) in expected.items():
        group = [r for r in manifest.records if (r.split, r.phase) == key]
        assert [r.seed for r in group] == list(range(first, first + count))
        assert [r.target_count for r in group] == [i % 6 + 1 for i in range(count)]
    assert Counter(r.purpose for r in manifest.records if r.split == "validation") == {"model_selection": 30, "development_calibration": 30}
    assert manifest.ready is False
    for consumer in ("collection", "training", "evaluation"):
        with pytest.raises(ValueError, match="not frozen"):
            manifest.require_ready(consumer)
    with pytest.raises(ValueError, match="unbound"):
        manifest.freeze()


@pytest.mark.parametrize("poison", ["seed", "range", "incomplete"])
def test_exclusions_fail_closed_no_seed_replacement(api, exclusions, poison):
    if poison == "seed":
        exclusions["seeds"].append(90050000)
    elif poison == "range":
        exclusions["reserved_seed_ranges"].append([90010000, 90010120])
    else:
        exclusions["complete_known_inputs"] = False
    with pytest.raises(ValueError):
        api[1].make_split_manifest(WAMConfig(), exclusions)


@pytest.mark.parametrize("derivative", ["different_n", "branch", "overlapping_window"])
def test_family_derivatives_cannot_cross_splits(api, derivative):
    root = {"parent_id": "layout-a", "seed": 90000000, "split": "train", "target_count": 1}
    child = dict(root, split="test", target_count=6, derivative=derivative)
    with pytest.raises(ValueError, match="family"):
        api[1].assert_family_isolation([root, child])
    child["split"] = "train"
    api[1].assert_family_isolation([root, child])


def test_seed_and_fingerprint_aliases_cannot_hide_same_family(api):
    for field in ("seed", "physical_fingerprint"):
        a = {"parent_id": "a", "split": "train", field: 99}
        b = {"parent_id": "b", "split": "test", field: 99}
        with pytest.raises(ValueError, match="family"):
            api[1].assert_family_isolation([a, b])


def test_real_pose_binding_freezing_and_persisted_identity(api, exclusions, tmp_path):
    splits = api[1]
    provisional = splits.make_split_manifest(WAMConfig(), exclusions)
    bound = provisional.bind(pose, budget(api, tmp_path), input_identity=identity(),
                             resolver_hash="d" * 64, seconds_per_layout=0.01)
    assert bound.ready is False
    assert all(r.physical_fingerprint for r in bound.records)
    expected = hashlib.sha256(np.asarray(pose(bound.records[0]), dtype="<f4").tobytes()).hexdigest()
    assert bound.records[0].physical_fingerprint == expected
    frozen = bound.freeze()
    assert frozen.ready is True
    assert len({r.physical_fingerprint for r in frozen.records}) == 852
    assert frozen.binding_metadata["input_identity"]["native_hashes"] == identity().native_hashes
    assert frozen.binding_metadata["config_hash"] == frozen.config_hash
    assert frozen.generator_hash and frozen.config_hash
    path = tmp_path / "split.json"
    frozen.save(path, budget(api, tmp_path))
    loaded = splits.SplitManifest.load(path)
    assert loaded.ready and loaded.manifest_hash == frozen.manifest_hash
    raw = json.loads(path.read_text())
    raw["records"][0]["seed"] += 100
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        splits.SplitManifest.load(path)


def test_partial_binding_resume_and_no_rebinding_after_outcomes(api, exclusions, tmp_path):
    manifest = api[1].make_split_manifest(WAMConfig(), exclusions)
    kwargs = dict(input_identity=identity(), resolver_hash="d" * 64, seconds_per_layout=0.01)
    bound = manifest.bind(pose, budget(api, tmp_path), parent_ids=[manifest.records[0].parent_id], **kwargs)
    assert not bound.ready
    assert sum(r.physical_fingerprint is not None for r in bound.records) == 1
    with pytest.raises(ValueError, match="bound"):
        bound.bind(pose, budget(api, tmp_path), parent_ids=[bound.records[0].parent_id], **kwargs)
    assert bound.bind(pose, budget(api, tmp_path), **kwargs).freeze().ready


def test_collision_with_sealed_fingerprint_or_other_parent_cannot_freeze(api, exclusions, tmp_path):
    manifest = api[1].make_split_manifest(WAMConfig(), exclusions)
    fp = hashlib.sha256(np.asarray(pose(manifest.records[0]), dtype="<f4").tobytes()).hexdigest()
    exclusions["fingerprints"] = [fp]
    excluded = api[1].make_split_manifest(WAMConfig(), exclusions)
    kwargs = dict(input_identity=identity(), resolver_hash="d" * 64, seconds_per_layout=0.01)
    with pytest.raises(ValueError, match="excluded"):
        excluded.bind(pose, budget(api, tmp_path), **kwargs)
    with pytest.raises(ValueError, match="family"):
        manifest.bind(lambda _: pose(manifest.records[0]), budget(api, tmp_path), **kwargs)
    assert all(r.physical_fingerprint is None for r in manifest.records)


def test_binding_admission_precedes_resolver_and_bad_poses_rejected(api, exclusions, tmp_path):
    manifest = api[1].make_split_manifest(WAMConfig(), exclusions)
    b = budget(api, tmp_path)
    b.disk_probe = lambda _: SimpleNamespace(free=24 * 2**30)
    def forbidden(_):
        pytest.fail("resolver executed before admission")
    kwargs = dict(input_identity=identity(), resolver_hash="d" * 64, seconds_per_layout=0.01)
    with pytest.raises(api[0].BudgetExceeded):
        manifest.bind(forbidden, b, **kwargs)
    for invalid in ("seed-hash", [[0] * 7] * 5, [[float("nan")] * 7] * 6):
        with pytest.raises(ValueError):
            manifest.bind(lambda _: invalid, budget(api, tmp_path), **kwargs)
