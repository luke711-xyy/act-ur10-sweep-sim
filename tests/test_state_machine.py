"""Unit tests for the episode/stroke phase state machine."""

import pytest

from sim.controllers.state_machine import ContactStateMachine, Phase, Signals


def drive(machine, steps, **kw):
    phase = machine.phase
    for i in range(steps):
        phase = machine.update(Signals(t=float(i) * 0.01, **kw))
    return phase


def test_nominal_stroke_sequence():
    m = ContactStateMachine(contact_dwell_steps=1)
    assert m.phase is Phase.IDLE
    assert m.begin_stroke(0.0) is Phase.APPROACH

    assert m.update(Signals(t=0.1, approach_done=True)) is Phase.SEARCH_CONTACT
    assert m.update(Signals(t=0.2, contact=True)) is Phase.CONTACT_DETECTED
    assert m.update(Signals(t=0.3)) is Phase.FORCE_RAMP
    assert m.update(Signals(t=0.4, ramp_up_done=True)) is Phase.SWEEP
    assert m.update(Signals(t=1.4, trajectory_done=True)) is Phase.FORCE_RELEASE
    assert m.update(Signals(t=1.6, ramp_down_done=True)) is Phase.RETRACT
    m.update(Signals(t=1.9, retract_done=True))
    assert m.stroke_finished

    phases = [e["phase"] for e in m.event_table()]
    assert phases == ["APPROACH", "SEARCH_CONTACT", "CONTACT_DETECTED", "FORCE_RAMP",
                      "SWEEP", "FORCE_RELEASE", "RETRACT"]
    assert m.contact_time == pytest.approx(0.2)
    assert m.release_time == pytest.approx(1.6)


def test_contact_dwell_rejects_single_sample_spikes():
    m = ContactStateMachine(contact_dwell_steps=3)
    m.begin_stroke(0.0)
    m.update(Signals(t=0.0, approach_done=True))
    m.update(Signals(t=0.01, contact=True))
    m.update(Signals(t=0.02, contact=False))     # spike -> counter resets
    assert m.phase is Phase.SEARCH_CONTACT
    for i in range(3):
        m.update(Signals(t=0.03 + 0.01 * i, contact=True))
    assert m.phase is Phase.CONTACT_DETECTED


def test_release_zone_preempts_the_sweep():
    """The Z force loop must never keep searching past the tray entrance."""
    m = ContactStateMachine(contact_dwell_steps=1)
    m.begin_stroke(0.0)
    m.update(Signals(approach_done=True))
    m.update(Signals(contact=True))
    m.update(Signals())
    m.update(Signals(ramp_up_done=True))
    assert m.phase is Phase.SWEEP
    assert m.update(Signals(t=2.0, release_zone=True, trajectory_done=False)) \
        is Phase.FORCE_RELEASE


def test_release_zone_preempts_the_force_ramp_too():
    m = ContactStateMachine(contact_dwell_steps=1)
    m.begin_stroke(0.0)
    m.update(Signals(approach_done=True))
    m.update(Signals(contact=True))
    m.update(Signals())
    assert m.phase is Phase.FORCE_RAMP
    assert m.update(Signals(release_zone=True)) is Phase.FORCE_RELEASE


def test_search_without_contact_retracts_instead_of_digging():
    m = ContactStateMachine(contact_dwell_steps=1)
    m.begin_stroke(0.0)
    m.update(Signals(approach_done=True))
    assert m.update(Signals(t=1.0, search_exhausted=True)) is Phase.RETRACT


def test_overforce_abort_routes_through_force_release():
    m = ContactStateMachine(contact_dwell_steps=1)
    m.begin_stroke(0.0)
    m.update(Signals(approach_done=True))
    m.update(Signals(contact=True))
    m.update(Signals())
    m.update(Signals(ramp_up_done=True))
    assert m.phase is Phase.SWEEP
    assert m.update(Signals(t=3.0, abort=True, abort_reason="over-force")) is Phase.FORCE_RELEASE
    assert m.abort_reason == "over-force"
    # ... and then out of contact entirely
    assert m.update(Signals(t=3.2, ramp_down_done=True)) is Phase.RETRACT


def test_abort_during_approach_retracts_directly():
    m = ContactStateMachine()
    m.begin_stroke(0.0)
    assert m.update(Signals(abort=True, abort_reason="workspace")) is Phase.RETRACT


def test_multiple_strokes_and_terminal_outcome():
    m = ContactStateMachine(contact_dwell_steps=1)
    for stroke in range(3):
        m.begin_stroke(float(stroke))
        assert m.stroke_index == stroke
        m.update(Signals(approach_done=True))
        m.update(Signals(contact=True))
        m.update(Signals())
        m.update(Signals(ramp_up_done=True))
        m.update(Signals(trajectory_done=True))
        m.update(Signals(ramp_down_done=True))
        m.update(Signals(retract_done=True))
        assert m.stroke_finished
    assert m.finish(True, 10.0) is Phase.SUCCESS
    # terminal phases are absorbing
    assert m.update(Signals(t=11.0, abort=True)) is Phase.SUCCESS


def test_failure_is_terminal():
    m = ContactStateMachine()
    m.begin_stroke(0.0)
    assert m.finish(False, 5.0, note="budget") is Phase.FAILURE
    assert m.update(Signals(approach_done=True)) is Phase.FAILURE
