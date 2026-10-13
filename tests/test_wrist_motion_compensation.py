"""Regression coverage for payload acceleration at the MuJoCo body COM."""

import mujoco
import numpy as np
import pytest

from sim.act.evaluate import _policy_reference_substep
from sim.config import load_config
from sim.controllers.hybrid import Command
from sim.environments.sweep_env import SweepEnv
from sim.environments.ur10_interface import UR10CB3EndEffector


def _rotating_payload(angle):
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="tool">
              <joint type="hinge" axis="0 1 0"/>
              <inertial pos="0 0 0.1" mass="2"
                        diaginertia="0.02 0.03 0.04"/>
              <site name="ft_site"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    data = mujoco.MjData(model)
    data.qpos[0] = angle
    data.qvel[0] = 3.0
    # I_y about the hinge = 0.03 + 2 * 0.1**2 = 0.05 kg m**2.
    # A 0.1 N m torque therefore produces alpha_y = 2 rad/s**2.
    data.qfrc_applied[0] = 0.1
    mujoco.mj_forward(model, data)
    mujoco.mj_rnePostConstraint(model, data)

    ee = UR10CB3EndEffector.__new__(UR10CB3EndEffector)
    ee._mujoco = mujoco
    ee.model, ee.data = model, data
    ee.ft_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "ft_site")
    ee.payload_body_ids = [mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "tool")]
    return ee


@pytest.mark.parametrize("angle, expected_force", [
    (0.0, [0.4, 0.0, -1.8]),
    (np.pi / 2.0, [-1.8, 0.0, -0.4]),
])
def test_rotating_payload_force_counts_com_acceleration_once(angle, expected_force):
    # At angle 0: alpha x r = (0.2, 0, 0),
    # omega x (omega x r) = (0, 0, -0.9); multiply by mass 2.
    ee = _rotating_payload(angle)
    np.testing.assert_allclose(
        ee._payload_motion_wrench_world()[:3], expected_force, atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, np.pi / 2.0])
def test_rotating_payload_torque_transfers_com_force_to_sensor_once(angle):
    ee = _rotating_payload(angle)
    # COM rotational torque is 0.03 * 2 = 0.06 N m; r x F adds 0.04.
    np.testing.assert_allclose(
        ee._payload_motion_wrench_world()[3:], [0.0, 0.1, 0.0], atol=1e-12)


# Original policy targets preceding the erroneous contact latch in job
# c33af5058686, target 4 / seed 198446336. No checkpoint or saved run is needed.
_PRE_CONTACT_TARGETS = np.array([
    *([[0.41998398303985596, 2.3214090106193908e-05, 0.09998719394207001, 1.2071898254362168e-06]] * 5),
    [0.3992246687412262, 0.0029814497102051973, 0.10000000149011612, 0.0026834968011826277],
    [0.3933258354663849, 0.00371126108802855, 0.10000000149011612, 0.0034309392794966698],
    [0.3877358138561249, 0.004354746546596289, 0.10000000149011612, 0.004213425330817699],
    [0.3829191327095032, 0.004848044831305742, 0.10000000149011612, 0.004941333085298538],
    [0.3789042830467224, 0.005201853346079588, 0.10000000149011612, 0.005489777773618698],
    [0.37623974680900574, 0.005405798554420471, 0.09999611973762512, 0.005866605322808027],
    [0.39081114530563354, 0.0036612641997635365, 0.08533674478530884, 0.004984820261597633],
    [0.38961389660835266, 0.0037312915083020926, 0.0849127396941185, 0.006808648817241192],
    [0.38837939500808716, 0.003799000522121787, 0.08460390567779541, 0.008672895841300488],
    [0.38726574182510376, 0.003864711383357644, 0.08425302803516388, 0.01044839434325695],
    [0.38615715503692627, 0.00391614343971014, 0.08390405029058456, 0.012204470112919807],
    [0.38538268208503723, 0.003950515296310186, 0.08164308220148087, 0.013792944140732288],
    [0.38444021344184875, 0.003969233483076096, 0.0792536735534668, 0.015542726963758469],
    [0.38603243231773376, 0.00384120992384851, 0.07141206413507462, 0.012505080550909042],
    [0.3857946991920471, 0.0038289353251457214, 0.0681379958987236, 0.013212302699685097],
    [0.3854418694972992, 0.0038067286368459463, 0.0650126039981842, 0.014203586615622044],
    [0.3852120041847229, 0.003787141991779208, 0.06336104869842529, 0.015257986262440681],
    [0.38509517908096313, 0.0037619229406118393, 0.06170153617858887, 0.016167689114809036],
    [0.38502269983291626, 0.0037429577205330133, 0.060053806751966476, 0.016783423721790314],
    [0.38494279980659485, 0.0037244767881929874, 0.05839104577898979, 0.017174560576677322],
    [0.3898463249206543, 0.0031474134884774685, 0.05272533372044563, 0.023378711193799973],
    [0.3897761404514313, 0.0031336545944213867, 0.0510101243853569, 0.024458689615130424],
    [0.38973402976989746, 0.003121601417660713, 0.049349796026945114, 0.02556593529880047],
    [0.3897433876991272, 0.003111064899712801, 0.047807417809963226, 0.026630183681845665],
    [0.38974714279174805, 0.0030975390691310167, 0.04618939384818077, 0.0277956984937191],
    [0.38974907994270325, 0.0030874924268573523, 0.045260246843099594, 0.02923331782221794],
    [0.38822489976882935, 0.003516592551022768, 0.035070132464170456, 0.02741052769124508],
    [0.3888609707355499, 0.003326614387333393, 0.029430879279971123, 0.027298204600811005],
    [0.38882461190223694, 0.003315221518278122, 0.02753743901848793, 0.028592651709914207],
    [0.38877221941947937, 0.003299576696008444, 0.025669703260064125, 0.02992437593638897],
    [0.38869306445121765, 0.003284909063950181, 0.025382233783602715, 0.03140130639076233],
    [0.38858354091644287, 0.0032687082421034575, 0.025099648162722588, 0.03306208923459053],
    [0.3881673514842987, 0.0033282851800322533, 0.01442547608166933, 0.042973656207323074],
    [0.3881247937679291, 0.0033213647548109293, 0.014102453365921974, 0.04462999850511551],
], dtype=np.float32)


def test_abrupt_recorded_act_motion_does_not_double_count_payload_load():
    cfg = load_config(overrides=["end_effector.type=ur10_cb3_rudra",
                                "task.target_count=4"])
    env = SweepEnv(cfg, seed=198446336)
    residuals = []
    try:
        env.reset(seed=198446336)
        start = np.array([*env.tcp(), env.ee.tcp_yaw()], dtype=np.float32)
        for index, action in enumerate(_PRE_CONTACT_TARGETS):
            # Stop at the original first-latch substep, while still airborne.
            last_substep = 3 if index == len(_PRE_CONTACT_TARGETS) - 1 else 4
            for substep in range(1, last_substep + 1):
                reference = _policy_reference_substep(
                    start, action, substep, 4, 0.01,
                    max_xy_speed=0.35, max_z_speed=0.12, max_yaw_rate=0.35,
                    contact_latched=False,
                )
                env.step_control(Command(*[float(v) for v in reference]))
                assert not env.brush_table_contact()
                assert not env.brush_component_contact_mask().any()
                residuals.append(float(env.compensated_wrench()[2]))
            start = reference.copy()
        # These are partial-compensation bounds, NOT a claim of zero force or
        # a guarantee that the unchanged 0.05 N contact gate cannot trigger.
        assert max(abs(value) for value in residuals) < 0.65
        assert abs(residuals[-1]) < 0.20
    finally:
        env.close()
