"""Sweep-task overlay on the native Rudra table/robot/camera assembly.

The old task sampler and MJCF objects are reused in metres. Only their origin
is translated to the native tabletop; the brush is rigidly held in PincOpen.
This is a scene/reset adapter, not a policy or force-controller adapter.
"""
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..config import load_config
from ..environments.layout import sample_layout
from .scene_builder import build_scene_xml

ROOT = Path(__file__).resolve().parents[2]
# Old brush +Z is its length axis; native gripper +X is its finger axis.
BRUSH_TO_GRIPPER = np.array([[0.,0.,1.], [0.,1.,0.], [-1.,0.,0.]])


def build_sweep_xml(native, cfg, task, layout):
    """Preserve configured cameras/native mounts and copy original task objects."""
    root = ET.fromstring(native.build_xml(cfg))
    old = ET.fromstring(build_scene_xml(task, layout))
    world, asset, default = (root.find(k) for k in ("worldbody","asset","default"))
    for body in list(world.findall("body")):
        if body.get("name") in {"peg","hole_circle","hole_square"}:
            world.remove(body)
    eq = root.find("equality")
    for weld in list(eq):
        if weld.get("name") == "grasp_assist":
            eq.remove(weld)
    default.append(deepcopy(old.find("default/default[@class='component']")))
    objects = [old.find(".//body[@name='target_tray']")]
    objects += [old.find(f".//body[@name='comp_{i}']") for i in range(len(layout))]
    mesh_names = {g.get("mesh") for b in objects for g in b.iter("geom") if g.get("mesh")}
    for node in old.find("asset"):
        if node.get("name") in mesh_names | {"mat_tray","mat_tip","mat_brush"}:
            asset.append(deepcopy(node))
    # Translate top-level free bodies, never nest freejoints under another body.
    translation = np.array([*cfg["table"]["center_xy"],cfg["table"]["z"]-float(task.table.top_z)])
    for body in objects:
        body = deepcopy(body)
        if body.get("name", "").startswith("comp_"):
            # Colour both visible shapes and collision proxies consistently.
            # This overlay affects appearance only, not source task physics.
            for geom in body.iter("geom"):
                geom.set("rgba", "0.02 0.02 0.02 1")
        p = np.fromstring(body.get("pos","0 0 0"),sep=" ") + translation
        body.set("pos",native.fmt(p))
        world.append(body)
    table = world.find("geom[@name='table']")
    table.set("contype","1")
    table.set("conaffinity","5")  # original parts + compliant brush sole
    table.set("friction",native.fmt(task.table.friction))

    import pincopen as PC
    grip = root.find(".//body[@name='pincopen_base']")
    anchor = np.array([PC._META["mount"]["pad_centre_x_g_mm"]*1e-3,-.0011,0.])
    anchor -= BRUSH_TO_GRIPPER @ [0.,0.,float(task.end_effector.brush_handle_center_z)]
    brush = ET.SubElement(grip,"body",name="sweep_brush",pos=native.fmt(anchor),
                          quat=native.fmt(native.mat2quat(BRUSH_TO_GRIPPER)))
    center_z = float(task.end_effector.brush_handle_center_z)+float(task.end_effector.brush_handle_length)/2
    ET.SubElement(brush,"inertial",pos=native.fmt([0,0,center_z]),mass=str(task.end_effector.brush_mass),
                  diaginertia="0.001 0.001 0.0005")
    for name in ("brush_handle","brush_head","brush_sole"):
        brush.append(deepcopy(old.find(f".//geom[@name='{name}']")))
    brush.append(deepcopy(old.find(".//site[@name='tcp_site']")))
    return ET.tostring(root,encoding="unicode")


def make_sweep_scene(native, cfg):
    """Use native rendering verbatim; specialize only task construction and reset."""
    class SweepScene(native.Scene):
        def __init__(self):
            import mujoco
            import pincopen as PC
            self.mj = mujoco
            self.cfg = deepcopy(cfg)
            source = Path(cfg["sweep_task"]["source_config"])
            self.task_config = load_config(str(source if source.is_absolute() else ROOT/source))
            self.seed = int(cfg["sweep_task"]["seed"])
            self.local_layout = sample_layout(self.task_config,np.random.default_rng(self.seed))
            self.translation = np.array([*cfg["table"]["center_xy"],cfg["table"]["z"]-float(self.task_config.table.top_z)])
            self.layout = [{**p,"x":p["x"]+self.translation[0],"y":p["y"]+self.translation[1]} for p in self.local_layout]
            self.xml = build_sweep_xml(native,cfg,self.task_config,self.local_layout)
            self.model = m = mujoco.MjModel.from_xml_string(self.xml)
            self.data = mujoco.MjData(m)
            self.renderer,self._size = None,None
            jid = lambda n: mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)
            self.arm_adr = [int(m.jnt_qposadr[jid(f"joint_{i}")]) for i in range(6)]
            self.arm_dof = [int(m.jnt_dofadr[jid(f"joint_{i}")]) for i in range(6)]
            self.bid = lambda n: mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
            self.cid = lambda n: mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_CAMERA,n)
            self.ctrl_range = m.actuator_ctrlrange[0].copy()
            thickness = float(self.task_config.end_effector.brush_handle_thickness)
            self.cfg["robot"]["gripper_frac"] = PC.frac_for_gap_mm(thickness*1e3)
            self.apply_state()

        def _pad_faces(self):
            """Inner pad faces in the native gripper frame, including linkage motion."""
            m,d = self.model,self.data
            g = self.bid("pincopen_base")
            R = d.xmat[g].reshape(3,3)
            pads = []
            for name in ("pad_L","pad_R"):
                i = self.mj.mj_name2id(m,self.mj.mjtObj.mjOBJ_GEOM,name)
                p = R.T @ (d.geom_xpos[i]-d.xpos[g])
                half_z = np.abs((R.T @ d.geom_xmat[i].reshape(3,3))[2]) @ m.geom_size[i]
                pads.append((p,half_z))
            pads.sort(key=lambda item:item[0][2])
            low,high = pads
            left,right = low[0][2]+low[1],high[0][2]-high[1]
            center = (low[0]+high[0])/2
            center[2] = (left+right)/2
            return center,float(right-left)

        def _pose_brush(self):
            from scipy.optimize import least_squares
            from scipy.spatial.transform import Rotation
            m,d,mj = self.model,self.data,self.mj
            g = self.bid("pincopen_base")
            tcp = mj.mj_name2id(m,mj.mjtObj.mjOBJ_SITE,"tcp_site")
            target = np.array(self.cfg["sweep_task"]["initial_tcp_local"],float)+self.translation
            # Vertical plate, width along table Y, finger axis pointing down.
            target_R = BRUSH_TO_GRIPPER

            def residual(q):
                d.qpos[self.arm_adr] = q
                mj.mj_forward(m,d)
                angle = Rotation.from_matrix(target_R @ d.xmat[g].reshape(3,3).T).as_rotvec()
                return np.r_[10*(d.site_xpos[tcp]-target),angle]

            q0 = np.radians(cfg["robot"]["joints_deg"])
            result = least_squares(residual,q0,bounds=(-2*np.pi,2*np.pi),max_nfev=250,
                                   ftol=1e-10,xtol=1e-10,gtol=1e-10)
            error = residual(result.x)
            if np.linalg.norm(error[:3]) > .005 or np.linalg.norm(error[3:]) > .002:
                raise RuntimeError(f"native brush initial pose IK failed: {error.tolist()}")
            self.q = result.x.copy()
            d.qpos[self.arm_adr] = self.q
            d.qvel[self.arm_dof] = 0
            mj.mj_forward(m,d)
            self.cfg["robot"].update(source="joints",joints_deg=np.degrees(self.q).tolist())

        def apply_state(self):
            import pincopen as PC
            mj,m,d = self.mj,self.model,self.data
            mj.mj_resetData(m,d)
            q = np.radians(cfg["robot"]["joints_deg"])
            self.frac = self.cfg["robot"]["gripper_frac"]
            target = float(np.clip(PC.cam_angle_for_frac(self.frac),*self.ctrl_range))
            n = int(self.cfg["render"]["settle_steps"])
            for k in range(n):
                d.ctrl[0] = target*min(1.,k/max(1,int(.65*n)))
                d.qpos[self.arm_adr] = q
                d.qvel[self.arm_dof] = 0
                mj.mj_step(m,d)
            d.qpos[self.arm_adr] = q
            d.qvel[self.arm_dof] = 0
            mj.mj_forward(m,d)
            # The upstream gap lookup differs from the compiled linkage by
            # ~2.4 mm here. Close to the measured handle width, preserving all
            # native linkage/actuator parameters and the camera installation.
            wanted = float(self.task_config.end_effector.brush_handle_thickness)
            for _ in range(12):
                _,gap = self._pad_faces()
                if abs(wanted-gap) < 0.00002:
                    break
                target = float(np.clip(target+(wanted-gap)/.05,*self.ctrl_range))
                d.ctrl[0] = target
                for _ in range(300):
                    d.qpos[self.arm_adr] = q
                    d.qvel[self.arm_dof] = 0
                    mj.mj_step(m,d)
                d.qpos[self.arm_adr] = q
                d.qvel[self.arm_dof] = 0
                mj.mj_forward(m,d)
            if abs(wanted-self._pad_faces()[1]) > .0001:
                raise RuntimeError("native gripper could not close to the brush handle width")
            order = np.argsort(PC._FM["cam_deg"])
            self.frac = float(np.interp(np.degrees(target),np.asarray(PC._FM["cam_deg"])[order],
                                       np.asarray(PC._FM["frac"])[order]))
            self.cfg["robot"]["gripper_frac"] = self.frac
            # As with the old fixed 2F85 grasp, mount the handle rigidly. Align
            # its centre with the actual settled inner pad faces, not a guess.
            center,self.jaw_gap = self._pad_faces()
            brush = self.bid("sweep_brush")
            handle_center = [0,0,float(self.task_config.end_effector.brush_handle_center_z)]
            m.body_pos[brush] = center-BRUSH_TO_GRIPPER @ handle_center
            self._pose_brush()
            actual = d.xpos[brush]+d.xmat[brush].reshape(3,3) @ handle_center
            g = self.bid("pincopen_base")
            grasp_world = d.xpos[g]+d.xmat[g].reshape(3,3) @ center
            self.grasp_error = float(np.linalg.norm(actual-grasp_world))

        def task_metadata(self):
            t,ee = self.task_config.target,self.task_config.end_effector
            bounds = {k:round(float(t[k])+self.translation[0 if k.startswith("x") else 1],9)
                      for k in ("x_min","x_max","y_min","y_max")}
            return dict(seed=self.seed,source_config=self.task_config["_source_config"],
                        component_count=len(self.layout),layout_local=deepcopy(self.local_layout),
                        layout_world=deepcopy(self.layout),translation=self.translation.tolist(),
                        generation_rules=self.task_config.components.to_dict(),target_bounds_world=bounds,
                        brush=dict(width=float(ee.brush_width),depth=float(ee.brush_depth),height=float(ee.brush_height),
                                   handle_thickness=float(ee.brush_handle_thickness),attachment="pincopen_base",
                                   mount_position_gripper=self.model.body_pos[self.bid("sweep_brush")].tolist(),
                                   measured_jaw_gap_m=self.jaw_gap,grasp_alignment_error_m=self.grasp_error))

    return SweepScene()
