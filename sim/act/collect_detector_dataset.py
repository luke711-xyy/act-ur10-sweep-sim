"""Build RGB detector supervision from replayed schema-v4 expert frames.

MuJoCo segmentation is used only here, offline, to create detector labels.
The resulting manifest contains RGB images and dense labels; no segmentation
image, simulator object pose, or truth count is passed to ObjectACT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..config import load_config
from ..environments.sweep_env import SweepEnv
from ..perception.detector_training import DetectorDatasetWriter, instance_maps_from_geom_ids
from ..perception.object_tokens import CLASS_TO_INDEX, GEOMETRY_TO_CLASS


def assign_layout_splits(records: list[dict], *, val_group_modulo: int = 5) -> dict[str, str]:
    """Assign whole layouts to train/val so frames cannot cross the split."""

    modulo = int(val_group_modulo)
    if modulo < 2:
        raise ValueError("val_group_modulo must be at least 2")
    groups = sorted({str(record.get("layout_id") or record["episode_id"]) for record in records})
    result = {}
    for index, group in enumerate(groups):
        result[group] = "val" if len(groups) > 1 and index % modulo == 0 else "train"
    return {
        str(record["episode_id"]): result[str(record.get("layout_id") or record["episode_id"])]
        for record in records
    }


class CameraSegmentationRenderer:
    """Reusable offline geom-id renderer for one environment/camera pair."""

    def __init__(self, env, camera_name: str, size: tuple[int, int]):
        import mujoco

        self._mujoco = mujoco
        width, height = (int(size[0]), int(size[1]))
        self._renderer = mujoco.Renderer(env.model, height=height, width=width)
        self._renderer.enable_segmentation_rendering()
        self.camera_name = str(camera_name)

    def render(self, env) -> np.ndarray:
        self._renderer.update_scene(env.data, camera=self.camera_name)
        segmentation = self._renderer.render()
        object_id = segmentation[:, :, 0].astype(np.int32)
        object_type = segmentation[:, :, 1].astype(np.int32)
        return np.where(
            object_type == int(self._mujoco.mjtObj.mjOBJ_GEOM), object_id, -1
        )

    def close(self) -> None:
        self._renderer.close()


def render_camera_segmentation(env, camera_name: str, size: tuple[int, int]) -> np.ndarray:
    """Render geom IDs for one camera without altering the environment state."""

    renderer = CameraSegmentationRenderer(env, camera_name, size)
    try:
        return renderer.render(env)
    finally:
        renderer.close()


def set_replay_state(env: SweepEnv, joint_position: np.ndarray, object_pose: np.ndarray) -> None:
    """Put a v4 frame back into a freshly reset MuJoCo scene."""

    import mujoco

    joints = np.asarray(joint_position, dtype=float).reshape(6)
    poses = np.asarray(object_pose, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 7 or poses.shape[0] != len(env.component_body_ids):
        raise ValueError("object_pose must have shape (component_count, 7)")
    for name, value in zip(getattr(env.ee, "JOINT_NAMES", ()), joints):
        if name in getattr(env.ee, "qpos_adr", {}):
            env.data.qpos[env.ee.qpos_adr[name]] = float(value)
            env.data.qvel[env.ee.qvel_adr[name]] = 0.0
    for index, body_id in enumerate(env.component_body_ids):
        joint_id = int(env.model.body_jntadr[body_id])
        if joint_id < 0:
            continue
        qpos_address = int(env.model.jnt_qposadr[joint_id])
        qvel_address = int(env.model.jnt_dofadr[joint_id])
        if int(env.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        env.data.qpos[qpos_address:qpos_address + 7] = poses[index]
        env.data.qvel[qvel_address:qvel_address + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)
    mujoco.mj_rnePostConstraint(env.model, env.data)


def _frame_indices(frame_count: int, stride: int, maximum: int | None) -> list[int]:
    if int(stride) < 1:
        raise ValueError("frame stride must be positive")
    values = list(range(0, int(frame_count), int(stride)))
    if not values:
        return []
    if maximum is not None and int(maximum) > 0 and len(values) > int(maximum):
        values = np.linspace(0, int(frame_count) - 1, int(maximum), dtype=int).tolist()
    return sorted(set(int(value) for value in values))


def collect_detector_dataset(
    cfg,
    *,
    source_root: str | Path,
    output_root: str | Path,
    frame_stride: int = 5,
    max_frames_per_episode: int | None = 80,
    val_group_modulo: int = 5,
    max_episodes: int | None = None,
) -> dict:
    """Replay v4 RGB frames and emit detector train/validation data."""

    source_root = Path(source_root)
    output_root = Path(output_root)
    if source_root.resolve() == output_root.resolve():
        raise ValueError("detector output must be separate from the v4 source")
    manifest = source_root / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    records = [
        json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [
        record for record in records
        if bool(record.get("success", False))
        and int(record.get("schema_version", 0)) == 4
        and str(record.get("episode_kind", "")) in {"expert", "expert_preview"}
    ]
    records.sort(key=lambda item: str(item["episode_id"]))
    if max_episodes is not None:
        records = records[:max(0, int(max_episodes))]
    if not records:
        raise ValueError("source v4 manifest has no successful expert records")
    splits = assign_layout_splits(records, val_group_modulo=val_group_modulo)
    writer = DetectorDatasetWriter(output_root)
    counts = {"train": 0, "val": 0}
    for record in records:
        local_cfg = cfg.copy()
        local_cfg.set_path("task.target_count", int(record.get("target_count", 1)))
        local_cfg.set_path("components.count", int(record.get("total_count", 6)))
        env = SweepEnv(local_cfg, seed=int(record.get("seed", 0)))
        env.reset(seed=int(record.get("seed", 0)))
        classes = [
            CLASS_TO_INDEX[GEOMETRY_TO_CLASS[str(item["geometry"])]] + 1
            for item in env.layout
        ]
        arrays = np.load(source_root / record["arrays"])
        frame_count = int(record.get("frame_count", len(arrays["t"])))
        image_paths = list(record.get("overhead", []))
        indices = _frame_indices(frame_count, frame_stride, max_frames_per_episode)
        renderer = CameraSegmentationRenderer(
            env,
            "overhead_cam",
            (int(local_cfg.act.image_size[0]), int(local_cfg.act.image_size[1])),
        )
        try:
            for frame_index in indices:
                if frame_index >= len(image_paths):
                    continue
                image_path = source_root / image_paths[frame_index]
                if not image_path.exists():
                    continue
                frame_id = f"{record['episode_id']}_{frame_index:05d}"
                if (output_root / frame_id).exists():
                    continue
                set_replay_state(
                    env,
                    arrays["joint_position"][frame_index],
                    arrays["object_pose"][frame_index],
                )
                segmentation = renderer.render(env)
                instance_map, class_map = instance_maps_from_geom_ids(
                    segmentation,
                    geom_to_component=env.geom_to_component,
                    component_class_indices=classes,
                )
                from PIL import Image

                image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
                writer.add_frame(
                    frame_id,
                    image,
                    instance_map,
                    class_map,
                    split=splits[record["episode_id"]],
                    metadata={
                        "source_episode": str(record["episode_id"]),
                        "frame_index": int(frame_index),
                        "layout_id": str(record.get("layout_id", record["episode_id"])),
                    },
                )
                counts[splits[record["episode_id"]]] += 1
        finally:
            renderer.close()
            env.close()
    return {"frames": int(sum(counts.values())), "splits": counts, "episodes": len(records)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build RGB detector supervision from v4 ACT data")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default="runs/act_dataset")
    parser.add_argument("--out", default="runs/objectact_detector_dataset")
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--max-frames-per-episode", type=int, default=80)
    parser.add_argument("--val-group-modulo", type=int, default=5)
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args(argv)
    result = collect_detector_dataset(
        load_config(args.config),
        source_root=args.source,
        output_root=args.out,
        frame_stride=args.frame_stride,
        max_frames_per_episode=args.max_frames_per_episode,
        val_group_modulo=args.val_group_modulo,
        max_episodes=args.max_episodes,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
