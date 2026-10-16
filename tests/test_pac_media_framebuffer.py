import mujoco
import numpy as np


def test_replay_960x640_framebuffer_can_render_without_advancing_physics():
    from pac_act.media import configure_replay_framebuffer
    model=mujoco.MjModel.from_xml_string('''<mujoco>
      <visual><global offwidth="640" offheight="480"/></visual>
      <worldbody><geom type="plane" size="1 1 .1"/></worldbody>
    </mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    before=data.time
    configure_replay_framebuffer(model,(960,640))
    with mujoco.Renderer(model,width=960,height=640) as renderer:
        renderer.update_scene(data)
        frame=renderer.render()
    assert frame.shape==(640,960,3)
    assert data.time==before
