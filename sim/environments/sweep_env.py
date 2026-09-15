"""MuJoCo environment for the tabletop component-collection task.

The environment is deliberately thin: it builds the scene, advances physics at
the control rate, and reports measurements.  Phase logic lives in the
controller, stroke selection in the planners, and scoring in
:mod:`sim.metrics`.

MuJoCo is imported lazily so that the rest of the package (controllers,
planners, geometry, metrics) can be imported and unit-tested without it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..model.scene_builder import build_scene_xml
from .ee_interface import EndEffectorInterface, build_end_effector
from .layout import in_safe_workspace, in_target_region, sample_layout


class SweepEnv:
    def __init__(self, cfg, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.model = None
        self.data = None
        self.ee: Optional[EndEffectorInterface] = None
        self.layout: List[dict] = []
        self.xml: str = ""
        self.table_top_z: float = float(cfg.table.top_z)
        self._renderer = None
        self._seg_renderer = None
        self._depth_renderer = None
        self.decimation = max(1, int(round((1.0 / float(cfg.sim.control_hz)) / float(cfg.sim.physics_dt))))
        self.component_body_ids: List[int] = []
        self.component_geom_ids: List[List[int]] = []
        self.geom_to_component: Dict[int, int] = {}

    # ------------------------------------------------------------------ setup
    def reset(self, seed: Optional[int] = None):
        import mujoco

        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        cfg = self.cfg.copy()
        jitter = float(cfg.table.height_perturb_std)
        if jitter > 0.0:
            self.table_top_z = float(cfg.table.top_z) + float(self.rng.normal(0.0, jitter))
            cfg.set_path("table.top_z", self.table_top_z)
        else:
            self.table_top_z = float(cfg.table.top_z)
        self.episode_cfg = cfg

        self.layout = sample_layout(cfg, self.rng)
        self.xml = build_scene_xml(cfg, self.layout)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self._close_renderers()

        self.ee = build_end_effector(self.model, self.data, cfg)
        # ee_z is a world-frame slide joint, so z_home is an absolute height.
        self.ee.reset([0.42, 0.0, float(cfg.end_effector.z_home)])

        self.component_body_ids = []
        self.component_geom_ids = []
        self.geom_to_component = {}
        for index in range(len(self.layout)):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"comp_{index}")
            self.component_body_ids.append(int(bid))
            gids = [int(g) for g in range(self.model.ngeom) if int(self.model.geom_bodyid[g]) == bid]
            self.component_geom_ids.append(gids)
            for g in gids:
                self.geom_to_component[g] = index
        if hasattr(self.ee, "set_component_geoms"):
            self.ee.set_component_geoms(self.geom_to_component.keys())

        mujoco.mj_forward(self.model, self.data)
        self._settle(float(cfg.episode.settle_time))
        self.initial_positions = self.component_positions().copy()
        return self.observation()

    def _settle(self, seconds: float) -> None:
        import mujoco

        steps = int(round(seconds / float(self.cfg.sim.physics_dt)))
        for _ in range(max(0, steps)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_rnePostConstraint(self.model, self.data)

    def _close_renderers(self) -> None:
        for attr in ("_renderer", "_seg_renderer", "_depth_renderer"):
            renderer = getattr(self, attr, None)
            if renderer is not None:
                try:
                    renderer.close()
                except Exception:  # pragma: no cover - renderer teardown is best effort
                    pass
            setattr(self, attr, None)

    def close(self) -> None:
        self._close_renderers()

    # ------------------------------------------------------------------- time
    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def control_dt(self) -> float:
        return self.decimation * float(self.cfg.sim.physics_dt)

    # ------------------------------------------------------------------- step
    def step_control(self, cmd) -> None:
        """Apply one task-space command and advance physics for one control period."""
        import mujoco

        self.ee.set_command(cmd.x, cmd.y, cmd.z, cmd.yaw)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        # cfrc_ext / force sensors need the post-constraint RNE pass.
        mujoco.mj_rnePostConstraint(self.model, self.data)

    # ----------------------------------------------------------- measurements
    def tcp(self) -> np.ndarray:
        return self.ee.tcp_position()

    def tcp_velocity(self) -> np.ndarray:
        return self.ee.tcp_velocity()

    def normal_force(self) -> float:
        return float(self.ee.normal_force())

    def wrench(self) -> np.ndarray:
        return self.ee.wrench()

    def contact_breakdown(self):
        """``(wrench_parts, n_part_contacts)`` -- see EndEffectorInterface."""
        return self.ee.contact_breakdown()

    def component_positions(self) -> np.ndarray:
        return np.array([self.data.xpos[bid] for bid in self.component_body_ids], dtype=float)

    def component_quats(self) -> np.ndarray:
        return np.array([self.data.xquat[bid] for bid in self.component_body_ids], dtype=float)

    def collected_mask(self) -> np.ndarray:
        pos = self.component_positions()
        if pos.size == 0:
            return np.zeros(0, dtype=bool)
        return in_target_region(pos[:, :2], self.cfg.target)

    def lost_mask(self) -> np.ndarray:
        pos = self.component_positions()
        if pos.size == 0:
            return np.zeros(0, dtype=bool)
        inside = in_safe_workspace(pos[:, :2], self.cfg.workspace)
        fell = pos[:, 2] < (self.table_top_z - 0.03)
        return (~inside) | fell

    def collection_rate(self) -> float:
        n = len(self.component_body_ids)
        return float(self.collected_mask().sum()) / n if n else 0.0

    def observation(self) -> Dict:
        return {
            "t": self.time,
            "tcp": self.tcp(),
            "tcp_velocity": self.tcp_velocity(),
            "normal_force": self.normal_force(),
            "component_positions": self.component_positions(),
            "collected": self.collected_mask(),
            "table_top_z": self.table_top_z,
        }

    # -------------------------------------------------------------- rendering
    def render_rgb(self) -> np.ndarray:
        import mujoco

        cam = self.cfg.perception.camera
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=int(cam.height), width=int(cam.width))
        self._renderer.update_scene(self.data, camera="scene_cam")
        return self._renderer.render()

    def render_depth(self) -> np.ndarray:
        import mujoco

        cam = self.cfg.perception.camera
        if self._depth_renderer is None:
            self._depth_renderer = mujoco.Renderer(self.model, height=int(cam.height),
                                                   width=int(cam.width))
            self._depth_renderer.enable_depth_rendering()
        self._depth_renderer.update_scene(self.data, camera="scene_cam")
        return self._depth_renderer.render()

    def render_segmentation(self) -> np.ndarray:
        """``(H, W)`` array of geom ids (-1 where nothing was hit)."""
        import mujoco

        cam = self.cfg.perception.camera
        if self._seg_renderer is None:
            self._seg_renderer = mujoco.Renderer(self.model, height=int(cam.height),
                                                 width=int(cam.width))
            self._seg_renderer.enable_segmentation_rendering()
        self._seg_renderer.update_scene(self.data, camera="scene_cam")
        seg = self._seg_renderer.render()
        # channel 0 = model element id, channel 1 = object type
        obj_id = seg[:, :, 0].astype(np.int32)
        obj_type = seg[:, :, 1].astype(np.int32)
        geom_ids = np.where(obj_type == int(mujoco.mjtObj.mjOBJ_GEOM), obj_id, -1)
        return geom_ids

    def camera_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Camera position and 3x3 rotation (columns = camera x/y/z axes, world)."""
        import mujoco

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "scene_cam")
        pos = np.array(self.data.cam_xpos[cam_id], dtype=float)
        rot = np.array(self.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
        return pos, rot
