"""Tests for target-scoped demonstration names."""

import json

from sim.act.naming import next_demo_name, next_demo_serial


def _manifest(root, names):
    root.mkdir()
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps({"episode_id": name}) + "\n" for name in names),
        encoding="utf-8",
    )


def test_demo_suffixes_are_independent_per_target(tmp_path):
    previews = tmp_path / "previews"
    dataset = tmp_path / "dataset"
    _manifest(previews, [
        "preview_goal_1_0001",
        "preview_goal_1_0002",
        "preview_goal_3_0001",
    ])
    _manifest(dataset, ["preview_goal_1_0003", "preview_goal_6_0001"])

    assert next_demo_serial([previews, dataset], 1) == 4
    assert next_demo_serial([previews, dataset], 2) == 1
    assert next_demo_name([previews, dataset], 2) == "preview_goal_2_0001"
    assert next_demo_name([previews, dataset], 6) == "preview_goal_6_0002"


def test_demo_suffix_fills_a_target_scoped_hole(tmp_path):
    previews = tmp_path / "previews"
    _manifest(previews, [
        "preview_goal_1_0001",
        "preview_goal_1_0003",
    ])
    assert next_demo_serial([previews], 1) == 2
    assert next_demo_name([previews], 1) == "preview_goal_1_0002"


def test_demo_suffix_skips_orphan_episode_directory_without_manifest(tmp_path):
    previews = tmp_path / "previews"
    previews.mkdir()
    (previews / "preview_goal_6_0001").mkdir()

    assert next_demo_serial([previews], 6) == 2
    assert next_demo_name([previews], 6) == "preview_goal_6_0002"
