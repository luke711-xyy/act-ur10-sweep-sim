"""Dynamic native Rudra scene using the existing sweep controller contracts.

Planning/proprioception use the original parallel task frame. MuJoCo and all
native cameras keep their base/world frame; their translation is recorded.
No controller gains, force processing, task geometry or sampling are replaced.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..config import load_config
from ..model.rudra_sweep import make_sweep_scene
from ..model.scene_builder import _add_rudra_ur10_actuators, _add_ur10_sensors
from .sweep_env import SweepEnv
from .ur10_interface import UR10CB3EndEffector

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT/'configs/rudra-sweep-scene.yaml'


def collection_config():
    from ..web.rudra_preview import upstream
    profile = upstream().load_config(PROFILE)
    cfg = load_config(str(ROOT/profile['sweep_task']['source_config']))
    cfg.set_path('scene.backend','rudra_native')
    cfg.set_path('scene.profile',str(PROFILE))
    cfg.set_path('act.camera_views',['top','wrist'])
    cfg.set_path('act.record_physics_contacts',True)
    cfg.set_path('planner.expert_validation_workers',1)
    cfg.set_path('sim.real_time',False)
    return cfg


class NativeRudraEndEffector(UR10CB3EndEffector):
    # A 0.1 mm IK deadband discards the small Z corrections of light contact.
    IK_CONVERGENCE_TOLERANCE = 1e-8

    def __init__(self, model, data, cfg, translation):
        super().__init__(model,data,cfg)
        self.translation = np.asarray(translation,dtype=float)

    def _solve_ik(self,target):
        world = np.asarray(target,dtype=float).copy()
        world[:3] += self.translation
        # Native grasp flips the symmetric plate by pi about the tool axis.
        # Preserve that mounted site; expose the original zero-yaw task frame.
        world[3] += np.pi
        return super()._solve_ik(world)

    def tcp_position(self):
        return super().tcp_position()-self.translation

    def tcp_yaw(self):
        angle = super().tcp_yaw()-np.pi
        return float(np.arctan2(np.sin(angle),np.cos(angle)))


class RudraSweepEnv(SweepEnv):
    def reset(self,seed=None):
        import mujoco
        from ..web.rudra_preview import upstream
        if seed is not None:
            self.seed = int(seed)
        self.close()
        native = upstream()
        profile = native.load_config(self.cfg.get_path('scene.profile',str(PROFILE)))
        profile['sweep_task']['seed'] = self.seed
        scene = make_sweep_scene(native,profile)
        self.translation = scene.translation.copy()
        root = ET.fromstring(scene.xml)
        # Carry over the measured fixed grasp established during native reset.
        root.find(".//body[@name='sweep_brush']").set(
            'pos',native.fmt(scene.model.body_pos[scene.bid('sweep_brush')]))
        sole_friction = self.cfg.get_path('end_effector.brush_sole_friction',None)
        if sole_friction is not None:
            root.find(".//geom[@name='brush_sole']").set('friction',native.fmt(sole_friction))
        flange = root.find(".//body[@name='flange_dh']")
        gripper = flange.find("body[@name='pincopen_base']")
        flange.remove(gripper)
        tool = ET.SubElement(flange,'body',name='tool')
        ET.SubElement(tool,'site',name='ft_site',size='.004',rgba='1 .2 .2 0')
        tool.append(gripper)
        for i in range(6):
            joint = root.find(f".//joint[@name='joint_{i}']")
            limit = np.pi if i==2 else 2*np.pi
            joint.set('limited','true')
            joint.set('range',native.fmt([-limit,limit]))
        if self.cfg.end_effector.get('gravity_compensation',True):
            for body in root.find(".//body[@name='ur_base']").iter('body'):
                body.set('gravcomp','1')
        _add_rudra_ur10_actuators(root.find('actuator'),self.cfg)
        _add_ur10_sensors(ET.SubElement(root,'sensor'))
        self.xml = ET.tostring(root,encoding='unicode')
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = scene.data.qpos
        self.data.qvel[:] = 0
        self.data.ctrl[0] = scene.data.ctrl[0]
        # Native assembly settling must not consume the original task budget.
        self.data.time = float(self.cfg.episode.settle_time)
        for i,value in enumerate(scene.q):
            aid = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,f'joint_{i}_act')
            self.data.ctrl[aid] = value
        self.episode_cfg = self.cfg.copy()
        self.episode_cfg.set_path('end_effector.rudra_initial_joint_positions',scene.q.tolist())
        self.layout = scene.local_layout
        self.table_top_z = float(self.cfg.table.top_z)
        self.ee = NativeRudraEndEffector(self.model,self.data,self.episode_cfg,self.translation)
        self.component_body_ids = []
        self.component_geom_ids = []
        self.geom_to_component = {}
        for i in range(len(self.layout)):
            bid = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'comp_{i}')
            gids = np.flatnonzero(self.model.geom_bodyid==bid).tolist()
            self.component_body_ids.append(int(bid))
            self.component_geom_ids.append(gids)
            self.geom_to_component.update({int(g):i for g in gids})
        self.ee.set_component_geoms(self.geom_to_component)
        self.brush_geom_ids = {int(g) for g in self.ee.tip_geom_ids}
        self.robot_geom_ids = set()
        base = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'ur_base')
        for gid,bid in enumerate(self.model.geom_bodyid):
            cursor = int(bid)
            while cursor>0 and cursor!=base:
                cursor = int(self.model.body_parentid[cursor])
            if cursor==base:
                self.robot_geom_ids.add(gid)
        self.table_geom_id = int(mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,'table'))
        self._zero_component_velocities()
        mujoco.mj_forward(self.model,self.data)
        mujoco.mj_rnePostConstraint(self.model,self.data)
        self.ee.calibrate_wrist_ft()
        self.initial_positions = self.component_positions().copy()
        self.last_raw_force_peak = self.last_raw_force_peak_time = None
        scene.model,scene.data = self.model,self.data
        scene.bid = lambda n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n)
        scene.cid = lambda n:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_CAMERA,n)
        self.native_scene = scene
        return self.observation()

    def component_positions(self):
        return super().component_positions()-self.translation

    def component_collision_x_max(self):
        return super().component_collision_x_max()-self.translation[0]

    def render_rgb(self,camera='top',size=None):
        from PIL import Image
        camera = {'scene_cam':'overview','wrist_cam':'wrist'}.get(camera,camera)
        spec = self.native_scene.cfg['render']
        # Render the complete native 4:3 field of view before policy resizing.
        image = self.native_scene.render(spec['width'],spec['height'],cams=(camera,))[camera]
        if size is not None and tuple(size)!=(image.shape[1],image.shape[0]):
            image = np.asarray(Image.fromarray(image).resize(tuple(size),Image.Resampling.BILINEAR))
        return image

    def render_wrist_rgb(self,size=None):
        return self.render_rgb('wrist',size)

    def close(self):
        if getattr(self,'native_scene',None) is not None:
            self.native_scene.close()
            self.native_scene = None
        super().close()
