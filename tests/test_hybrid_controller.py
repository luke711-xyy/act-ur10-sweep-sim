"""Closed-loop tests of the hybrid force/position controller.

These run the *real* controller against a deliberately simple analytic plant, so
they check the controller's own behaviour (phase sequencing, force ramping,
release guard, saturations) without needing MuJoCo.

Plant model
-----------
* the TCP tracks the command through a first-order lag (time constant ``tau``)
* the table is at ``z = 0`` and behaves as a linear spring of stiffness
  ``k_env``, so ``F = max(0, -k_env * z)`` in the pressing-positive convention
"""

import numpy as np
import pytest

from sim.config import load_config
from sim.controllers.hybrid import HybridForcePositionController
from sim.controllers.state_machine import Phase
from sim.planners.base import SweepStroke


class FakePlant:
    def __init__(self, cfg, tau=0.02, k_env=3000.0, table_z=0.0):
        self.dt = 1.0 / float(cfg.sim.control_hz)
        self.tau = tau
        self.k_env = k_env
        self.table_z = table_z
        self.tcp = np.array([0.44, -0.34, float(cfg.end_effector.z_home)])

    def step(self, cmd):
        alpha = self.dt / (self.tau + self.dt)
        target = np.array([cmd.x, cmd.y, cmd.z])
        self.tcp = self.tcp + alpha * (target - self.tcp)
        return self.tcp.copy()

    def force(self):
        return max(0.0, -self.k_env * (self.tcp[2] - self.table_z))


def run_stroke(cfg, stroke, plant=None, max_steps=20000):
    controller = HybridForcePositionController(cfg, rng=np.random.default_rng(0))
    plant = plant or FakePlant(cfg)
    controller.reset(plant.tcp)
    t = 0.0
    dt = 1.0 / float(cfg.sim.control_hz)
    controller.start_stroke(stroke, plant.tcp, t)
    steps = 0
    while not controller.stroke_done and steps < max_steps:
        cmd = controller.step(t, plant.tcp, plant.force())
        plant.step(cmd)
        t += dt
        steps += 1
    return controller, plant, steps


@pytest.fixture
def cfg():
    cfg = load_config()
    cfg.set_path("end_effector.type", "cartesian3dof")
    cfg.set_path("workspace.z_search_start", 0.008)
    return cfg


@pytest.fixture
def stroke(cfg):
    return SweepStroke(0.35, 0.05, float(cfg.planner.stroke_end_x), 0.05)


def test_full_phase_sequence_is_visited(cfg, stroke):
    controller, _, steps = run_stroke(cfg, stroke)
    assert steps < 20000, "stroke did not terminate"
    phases = [e["phase"] for e in controller.fsm.event_table()]
    for expected in ("APPROACH", "SEARCH_CONTACT", "CONTACT_DETECTED", "FORCE_RAMP",
                     "SWEEP", "FORCE_RELEASE", "RETRACT"):
        assert expected in phases, f"{expected} never entered ({phases})"
    assert phases.index("SEARCH_CONTACT") < phases.index("CONTACT_DETECTED")
    assert phases.index("FORCE_RAMP") < phases.index("SWEEP") < phases.index("FORCE_RELEASE")
    assert controller.phase is Phase.RETRACT


def test_contact_force_is_bounded_and_positive_while_sweeping(cfg, stroke):
    controller, _, _ = run_stroke(cfg, stroke)
    sweeping = [r for r in controller.trace if r.phase == "SWEEP"]
    assert len(sweeping) > 50
    forces = np.array([r.force_filtered for r in sweeping])
    target = float(cfg.controller.desired_force)
    assert forces.min() > 0.0
    assert forces.max() < float(cfg.controller.safe_max_force)
    # the loop settles: the second half of the sweep tracks the target closely
    settled = forces[len(forces) // 2:]
    assert abs(settled.mean() - target) < 0.25 * target
    assert settled.std() < 0.25 * target


def test_force_is_released_before_the_guard_line(cfg, stroke):
    """Acceptance criterion: no downward force is applied past x_release_line."""
    controller, _, _ = run_stroke(cfg, stroke)
    line = float(cfg.controller.x_release_line)
    past_line = [r for r in controller.trace if r.tcp_x <= line + 1e-3]
    assert past_line, "the stroke never reached the guard line"
    assert max(r.force_desired for r in past_line) < 0.15
    assert all(r.phase in ("FORCE_RELEASE", "RETRACT") for r in past_line)


def test_z_search_stops_instead_of_digging_when_there_is_no_table(cfg, stroke):
    plant = FakePlant(cfg, table_z=-0.5)     # table far below: contact never happens
    controller, _, steps = run_stroke(cfg, stroke, plant=plant)
    assert steps < 20000
    assert max(r.force_desired for r in controller.trace) < 1e-6
    assert min(r.cmd_z for r in controller.trace) >= float(cfg.workspace.z_search_min) - 1e-9
    phases = [e["phase"] for e in controller.fsm.event_table()]
    assert "SWEEP" not in phases
    assert phases[-1] == "RETRACT"


def test_admittance_correction_stays_inside_its_saturation(cfg, stroke):
    controller, _, _ = run_stroke(cfg, stroke)
    limit = float(cfg.controller.delta_z_limit)
    assert max(abs(r.delta_z) for r in controller.trace) <= limit + 1e-12


def test_desired_force_slew_rate_is_limited(cfg, stroke):
    controller, _, _ = run_stroke(cfg, stroke)
    dt = 1.0 / float(cfg.sim.control_hz)
    fd = np.array([r.force_desired for r in controller.trace])
    max_step = float(cfg.controller.force_ramp_rate) * dt
    assert np.abs(np.diff(fd)).max() <= max_step + 1e-9


def test_overforce_triggers_a_safe_abort(cfg, stroke):
    """A far too stiff / high table drives the force past the safety limit."""
    cfg.set_path("controller.safe_max_force", 4.0)
    plant = FakePlant(cfg, k_env=200000.0)
    controller, _, steps = run_stroke(cfg, stroke)
    controller2, _, _ = run_stroke(cfg, stroke, plant=plant)
    assert controller2.stroke_stats["aborted"] or controller2.phase is Phase.RETRACT
    if controller2.stroke_stats["aborted"]:
        phases = [e["phase"] for e in controller2.fsm.event_table()]
        assert "FORCE_RELEASE" in phases or "RETRACT" in phases


def test_admittance_state_is_reset_at_the_start_of_every_stroke(cfg, stroke):
    controller = HybridForcePositionController(cfg, rng=np.random.default_rng(0))
    plant = FakePlant(cfg)
    controller.reset(plant.tcp)
    controller.admittance.reset(delta_z=-0.005)
    controller.start_stroke(stroke, plant.tcp, 0.0)
    assert controller.admittance.delta_z == 0.0


def test_control_delay_shifts_the_command(cfg, stroke):
    cfg.set_path("controller.control_delay_steps", 5)
    controller, _, steps = run_stroke(cfg, stroke)
    assert steps < 20000
    assert controller.phase is Phase.RETRACT


def test_force_sensor_noise_does_not_destabilise_the_loop(cfg, stroke):
    cfg.set_path("controller.force_noise_std", 0.25)
    controller, _, _ = run_stroke(cfg, stroke)
    sweeping = np.array([r.force_filtered for r in controller.trace if r.phase == "SWEEP"])
    assert sweeping.size > 50
    assert sweeping.max() < float(cfg.controller.safe_max_force)
    assert abs(sweeping[len(sweeping) // 2:].mean()
               - float(cfg.controller.desired_force)) < 0.35 * float(cfg.controller.desired_force)


def test_transfer_and_retract_happen_clear_of_the_table(cfg, stroke):
    """Return/approach motions must never be in contact (Planner A requirement)."""
    controller, _, _ = run_stroke(cfg, stroke)
    travel = float(cfg.workspace.z_travel)
    approach = [r for r in controller.trace if r.phase == "APPROACH"]
    # the lateral part of the approach is flown at travel height
    lateral = [r for r in approach
               if abs(r.cmd_x - stroke.x_start) > 1e-3 or abs(r.cmd_y - stroke.y_start) > 1e-3]
    assert lateral, "no lateral transfer segment found"
    assert min(r.cmd_z for r in lateral) >= travel - 1e-6
    assert not any(r.in_contact for r in lateral)
    retract = [r for r in controller.trace if r.phase == "RETRACT"]
    assert retract and max(r.cmd_z for r in retract) >= travel - 1e-6


def test_table_height_offset_is_absorbed_by_the_contact_search(cfg, stroke):
    """A table 4 mm higher/lower must not change the regulated force."""
    forces = {}
    for table_z in (-0.004, 0.0, 0.004):
        controller, _, _ = run_stroke(cfg, stroke, plant=FakePlant(cfg, table_z=table_z))
        sweeping = np.array([r.force_filtered for r in controller.trace if r.phase == "SWEEP"])
        assert sweeping.size > 50, f"no sweep phase with table at {table_z}"
        forces[table_z] = float(sweeping[sweeping.size // 2:].mean())
    target = float(cfg.controller.desired_force)
    for value in forces.values():
        assert abs(value - target) < 0.25 * target, forces


def test_yaw_command_follows_the_stroke(cfg):
    stroke = SweepStroke(0.35, 0.25, float(cfg.planner.stroke_end_x), 0.10, yaw=0.3)
    controller, _, _ = run_stroke(cfg, stroke)
    assert all(abs(r.cmd_yaw - 0.3) < 1e-9 for r in controller.trace)
