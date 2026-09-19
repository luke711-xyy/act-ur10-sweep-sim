"""Causal RGB-derived observations and control handoff for ObjectACT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from ..perception.camera import PinholeCamera
from ..perception.object_bev import (
    BEVSpec,
    PredictedInstance,
    TrackManager,
    annotate_tray_features,
    build_bev_channels,
    fuse_projected_instances,
    pack_instance_masks,
    project_image_instance,
)
from ..perception.object_tokens import pack_object_tokens
from .interface import _resize_rgb


@dataclass(frozen=True)
class ObjectPerceptionFrame:
    object_tokens: np.ndarray
    object_valid: np.ndarray
    instance_bev: np.ndarray
    bev: np.ndarray
    visual_tracked_count: int
    visual_full_in_tray_count: int


def apply_objectact_action(
    action: np.ndarray, contact_latched: bool, admittance_z: float
) -> np.ndarray:
    """Replace only the applied Z after contact; retain ACT's four outputs in logs."""

    values = np.asarray(action, dtype=np.float32).reshape(4).copy()
    if bool(contact_latched):
        values[2] = float(admittance_z)
    return values


class ObjectACTObservationBuilder:
    """Build schema-v5 observations without simulator object bookkeeping."""

    def __init__(self, cfg, *, frontend=None):
        self.cfg = cfg
        self.frontend = frontend or RGBObjectPerceptionFrontend(cfg)
        size = tuple(int(value) for value in cfg.act.image_size)
        self.image_size = size
        self._previous_robot = None
        self._previous_task = None

    def reset(self) -> None:
        self._previous_robot = None
        self._previous_task = None
        reset = getattr(self.frontend, "reset", None)
        if callable(reset):
            reset()

    def _target_count(self, env) -> int:
        total = max(1, len(env.layout))
        value = self.cfg.get_path("task.target_count", total)
        return int(np.clip(int(value), 1, total))

    def _tray_bounds(self) -> tuple[float, float, float, float]:
        """Read tray geometry without making the observation builder test-only.

        Production configs expose ``cfg.target`` as a nested Config object;
        lightweight callers may only implement ``get_path``.  Keeping this
        lookup in one place also makes it explicit that the detector receives
        fixed scene geometry, never simulator object positions.
        """
        target = getattr(self.cfg, "target", None)
        if target is not None:
            return (
                float(target.x_min), float(target.x_max),
                float(target.y_min), float(target.y_max),
            )
        get_path = getattr(self.cfg, "get_path", None)
        if not callable(get_path):
            raise AttributeError("configuration must expose target bounds")
        defaults = {"x_min": -0.45, "x_max": -0.31, "y_min": -0.18, "y_max": 0.18}
        return tuple(
            float(get_path(f"target.{name}", defaults[name]))
            for name in ("x_min", "x_max", "y_min", "y_max")
        )

    def _robot_state(self, env, contact_latched: bool) -> np.ndarray:
        joints = np.asarray(env.ee.joint_state(), dtype=np.float32).reshape(6)
        tcp = np.asarray(env.tcp(), dtype=np.float32).reshape(3)
        yaw = float(env.ee.tcp_yaw())
        wrench = np.asarray(env.wrench(), dtype=np.float32).reshape(6)
        current = np.concatenate((
            joints, tcp, np.array([np.sin(yaw), np.cos(yaw)], dtype=np.float32),
            wrench, np.asarray([float(bool(contact_latched))], dtype=np.float32),
        )).astype(np.float32)
        previous = current if self._previous_robot is None else self._previous_robot
        self._previous_robot = current.copy()
        return np.concatenate((current, previous)).astype(np.float32)

    def observe(self, env, contact_latched: bool = False) -> dict[str, Any]:
        overhead = _resize_rgb(env.render_rgb("overhead_cam", size=self.image_size), self.image_size)
        wrist = _resize_rgb(env.render_wrist_rgb(size=self.image_size), self.image_size)
        tcp = np.asarray(env.tcp(), dtype=np.float32)
        tray_bounds = self._tray_bounds()
        frame = self.frontend.observe(
            env=env,
            overhead_rgb=overhead,
            wrist_rgb=wrist,
            brush_xy=tcp[:2],
            brush_yaw=float(env.ee.tcp_yaw()),
            timestamp=float(env.time),
            tray_bounds=tray_bounds,
        )
        total = 6.0
        target = float(self._target_count(env))
        current_task = np.asarray([
            float(frame.visual_tracked_count) / total,
            target / total,
            float(frame.visual_full_in_tray_count) / total,
        ], dtype=np.float32)
        previous_task = current_task if self._previous_task is None else self._previous_task
        self._previous_task = current_task.copy()
        return {
            "observation.images.overhead": np.transpose(overhead, (2, 0, 1)).astype(np.float32) / 255.0,
            "observation.images.wrist": np.transpose(wrist, (2, 0, 1)).astype(np.float32) / 255.0,
            "observation.robot_state": self._robot_state(env, contact_latched),
            "observation.task_state": np.concatenate((current_task, previous_task)).astype(np.float32),
            "observation.object_tokens": np.asarray(frame.object_tokens, dtype=np.float32),
            "observation.object_valid": np.asarray(frame.object_valid, dtype=bool),
            "observation.instance_bev": np.asarray(frame.instance_bev, dtype=bool),
            "observation.bev": np.asarray(frame.bev, dtype=np.float32),
            "contact_latched": bool(contact_latched),
            "t": float(env.time),
        }

    @staticmethod
    def torch_batch(observation: dict[str, Any], device: str | None = None):
        import torch

        batch = {}
        for key, value in observation.items():
            if key in {"t", "contact_latched"}:
                continue
            tensor = torch.as_tensor(value, device=device)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32)
            batch[key] = tensor.unsqueeze(0)
        return batch


class RGBObjectPerceptionFrontend:
    """Run the frozen RGB detector, projection, fusion and tracker online."""

    def __init__(self, cfg, *, detector=None, device: str = "cpu", detector_checkpoint: str | None = None):
        import torch

        from ..perception.detector import FrozenResNet18FPN
        from ..perception.detector_training import verified_detector_checkpoint

        self.device = str(device)
        self.foreground_threshold = float(
            cfg.act.get("objectact_detector_foreground_threshold", 0.60)
        )
        if not 0.0 < self.foreground_threshold < 1.0:
            raise ValueError("objectact detector foreground threshold must be in (0, 1)")
        if detector is None and not detector_checkpoint:
            raise ValueError(
                "RGBObjectPerceptionFrontend requires a verified detector checkpoint"
            )
        self.detector = detector or FrozenResNet18FPN(
            pretrained=bool(cfg.act.get("objectact_detector_pretrained", False)),
            freeze_backbone=True,
        )
        if detector_checkpoint:
            detector_checkpoint = verified_detector_checkpoint(detector_checkpoint)
            state = torch.load(detector_checkpoint, map_location="cpu", weights_only=True)
            self.detector.load_state_dict(state)
        self.detector.to(self.device).eval()
        self.tracker = TrackManager()
        self.bev_spec = BEVSpec()

    def reset(self) -> None:
        self.tracker = TrackManager()

    @staticmethod
    def _camera(env, name: str, size: tuple[int, int], fovy: float) -> PinholeCamera:
        try:
            position, rotation = env.camera_pose(name)
        except TypeError:
            position, rotation = env.camera_pose()
        return PinholeCamera(
            position=np.asarray(position, dtype=float), rotation=np.asarray(rotation, dtype=float),
            width=int(size[0]), height=int(size[1]), fovy_deg=float(fovy),
        )

    def _detect(self, image: np.ndarray):
        import torch

        from ..perception.detector import decode_detector_output

        tensor = torch.as_tensor(np.transpose(image, (2, 0, 1)), dtype=torch.float32, device=self.device)
        tensor = tensor.unsqueeze(0) / 255.0
        with torch.no_grad():
            output = self.detector(tensor)
        return decode_detector_output(
            output,
            batch_index=0,
            foreground_threshold=self.foreground_threshold,
        )

    def observe(
        self,
        *,
        env,
        overhead_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        brush_xy: np.ndarray,
        brush_yaw: float,
        timestamp: float,
        tray_bounds: Sequence[float],
    ) -> ObjectPerceptionFrame:
        size = (int(overhead_rgb.shape[1]), int(overhead_rgb.shape[0]))
        scene_fovy = float(env.cfg.perception.camera.fovy_deg)
        wrist_fovy = float(env.cfg.perception.wrist_camera.fovy_deg)
        overhead_cfg = env.cfg.get_path("video.extra_cameras.overhead_cam", None)
        overhead_fovy = float(
            overhead_cfg.fovy_deg if overhead_cfg is not None else scene_fovy
        )
        detections = []
        for image, camera_name, fovy, visibility in (
            (overhead_rgb, "overhead_cam", overhead_fovy, (1.0, 0.0)),
            (wrist_rgb, "wrist_cam", wrist_fovy, (0.0, 1.0)),
        ):
            camera = self._camera(env, camera_name, size, fovy)
            for prediction in self._detect(image):
                projected = project_image_instance(
                    prediction.mask,
                    camera=camera,
                    plane_z=float(env.table_top_z),
                    bev_spec=self.bev_spec,
                    class_probs=prediction.class_probs,
                    confidence=prediction.confidence,
                    overhead_visibility=visibility[0],
                    wrist_visibility=visibility[1],
                )
                detections.append(projected)
        fused = fuse_projected_instances(detections, spec=self.bev_spec)
        fused = [annotate_tray_features(item, spec=self.bev_spec, tray_bounds=tray_bounds) for item in fused]
        tracks = self.tracker.update(fused, timestamp=float(timestamp))
        instances = list(self.tracker.last_instances.values())
        tray_mouth = np.array(
            [(float(tray_bounds[0]) + float(tray_bounds[1])) / 2.0,
             (float(tray_bounds[2]) + float(tray_bounds[3])) / 2.0]
        )
        tokens, valid = pack_object_tokens(
            tracks, brush_xy=brush_xy, tray_mouth_xy=tray_mouth
        )
        masks, _, _ = pack_instance_masks(instances, spec=self.bev_spec)
        bev = build_bev_channels(
            instances,
            spec=self.bev_spec,
            selected_track_ids=set(),
            brush_xy=brush_xy,
            brush_size=np.array([0.12, 0.01]),
            brush_yaw=float(brush_yaw),
            tray_bounds=tray_bounds,
        )
        return ObjectPerceptionFrame(
            object_tokens=tokens,
            object_valid=valid,
            instance_bev=masks,
            bev=bev,
            visual_tracked_count=int(np.count_nonzero(valid)),
            visual_full_in_tray_count=sum(bool(track.full_inside) for track in tracks),
        )
