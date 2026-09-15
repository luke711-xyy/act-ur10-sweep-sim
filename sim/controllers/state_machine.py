"""Episode / stroke phase state machine.

Phases (as required by the task specification)::

    1 APPROACH         move above the stroke start, descend to the probe height
    2 SEARCH_CONTACT   descend slowly until the normal force crosses a threshold
    3 CONTACT_DETECTED latch z_nominal, reset the admittance state
    4 FORCE_RAMP       ramp the desired normal force up to its target
    5 SWEEP            XY position control + Z force regulation
    6 FORCE_RELEASE    ramp the desired normal force back to zero
    7 RETRACT          lift clear of the table
    8 SUCCESS | FAILURE

Phases 1-7 are executed once per sweeping stroke; SUCCESS/FAILURE are terminal
episode outcomes.  The machine is deliberately free of MuJoCo and of any
controller internals so it can be unit-tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class Phase(str, Enum):
    IDLE = "IDLE"
    APPROACH = "APPROACH"
    SEARCH_CONTACT = "SEARCH_CONTACT"
    CONTACT_DETECTED = "CONTACT_DETECTED"
    FORCE_RAMP = "FORCE_RAMP"
    SWEEP = "SWEEP"
    FORCE_RELEASE = "FORCE_RELEASE"
    RETRACT = "RETRACT"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


TERMINAL_PHASES = (Phase.SUCCESS, Phase.FAILURE)
CONTACT_PHASES = (Phase.CONTACT_DETECTED, Phase.FORCE_RAMP, Phase.SWEEP, Phase.FORCE_RELEASE)


@dataclass
class PhaseEvent:
    time: float
    phase: Phase
    stroke: int
    note: str = ""


@dataclass
class Signals:
    """Everything the state machine needs to decide on a transition.

    Keeping this as an explicit record (rather than reaching into the
    controller) is what makes the FSM independently testable.
    """

    t: float = 0.0
    approach_done: bool = False       # TCP is at the stroke start, at probe height
    search_exhausted: bool = False    # probed below the minimum search depth
    contact: bool = False             # filtered force > contact threshold
    contact_lost: bool = False        # filtered force < release threshold
    ramp_up_done: bool = False        # desired force reached its target
    ramp_down_done: bool = False      # desired force reached ~zero and force released
    trajectory_done: bool = False     # XY stroke trajectory finished
    release_zone: bool = False        # TCP entered the tray-entrance guard band
    retract_done: bool = False        # TCP back at travel height
    abort: bool = False               # safety abort (over-force, timeout, ...)
    abort_reason: str = ""


class ContactStateMachine:
    """Phase sequencer for one episode (many strokes)."""

    def __init__(self, contact_dwell_steps: int = 1):
        self.phase: Phase = Phase.IDLE
        self.stroke_index: int = -1
        self.events: List[PhaseEvent] = []
        self.abort_reason: str = ""
        self._contact_counter = 0
        self._contact_dwell_steps = max(1, int(contact_dwell_steps))
        self._retract_complete = False
        self.contact_time: Optional[float] = None
        self.release_time: Optional[float] = None
        self.stroke_contact_times: List[float] = []
        self.stroke_release_times: List[float] = []

    # -- bookkeeping ---------------------------------------------------------
    def _set(self, phase: Phase, t: float, note: str = "") -> None:
        if phase is self.phase:
            return
        self.phase = phase
        self.events.append(PhaseEvent(time=float(t), phase=phase, stroke=self.stroke_index, note=note))

    def begin_stroke(self, t: float = 0.0) -> Phase:
        self.stroke_index += 1
        self._contact_counter = 0
        self._retract_complete = False
        self.contact_time = None
        self.release_time = None
        self._set(Phase.APPROACH, t, note=f"stroke {self.stroke_index}")
        return self.phase

    @property
    def stroke_finished(self) -> bool:
        return self.phase is Phase.RETRACT and self._retract_complete

    # -- main transition -----------------------------------------------------
    def update(self, s: Signals) -> Phase:
        self._retract_complete = False
        if self.phase in TERMINAL_PHASES:
            return self.phase

        if s.abort:
            self.abort_reason = s.abort_reason or "abort"
            # An abort never leaves the tool pressing on the table: it routes
            # through FORCE_RELEASE/RETRACT rather than stopping in place.
            if self.phase in (Phase.CONTACT_DETECTED, Phase.FORCE_RAMP, Phase.SWEEP):
                self._set(Phase.FORCE_RELEASE, s.t, note=f"abort: {self.abort_reason}")
                return self.phase
            if self.phase in (Phase.APPROACH, Phase.SEARCH_CONTACT):
                self._set(Phase.RETRACT, s.t, note=f"abort: {self.abort_reason}")
                return self.phase

        if self.phase is Phase.APPROACH:
            if s.approach_done:
                self._set(Phase.SEARCH_CONTACT, s.t)

        elif self.phase is Phase.SEARCH_CONTACT:
            if s.contact:
                self._contact_counter += 1
                if self._contact_counter >= self._contact_dwell_steps:
                    self.contact_time = s.t
                    self.stroke_contact_times.append(s.t)
                    self._set(Phase.CONTACT_DETECTED, s.t, note="contact")
            else:
                self._contact_counter = 0
                if s.search_exhausted:
                    self._set(Phase.RETRACT, s.t, note="search exhausted, no contact")

        elif self.phase is Phase.CONTACT_DETECTED:
            # One-shot phase: the controller latches z_nominal and resets the
            # admittance state while we are here, then we always move on.
            self._set(Phase.FORCE_RAMP, s.t)

        elif self.phase is Phase.FORCE_RAMP:
            if s.release_zone:
                self._set(Phase.FORCE_RELEASE, s.t, note="release zone during ramp")
            elif s.ramp_up_done:
                self._set(Phase.SWEEP, s.t)

        elif self.phase is Phase.SWEEP:
            # The release-zone guard has priority over everything else: the Z
            # force loop must never keep searching for contact past the table
            # edge or into the tray entrance.
            if s.release_zone:
                self._set(Phase.FORCE_RELEASE, s.t, note="release zone")
            elif s.trajectory_done:
                self._set(Phase.FORCE_RELEASE, s.t, note="trajectory done")

        elif self.phase is Phase.FORCE_RELEASE:
            if s.ramp_down_done:
                self.release_time = s.t
                self.stroke_release_times.append(s.t)
                self._set(Phase.RETRACT, s.t, note="released")

        elif self.phase is Phase.RETRACT:
            if s.retract_done:
                self._retract_complete = True

        return self.phase

    # -- terminal outcomes ---------------------------------------------------
    def finish(self, success: bool, t: float, note: str = "") -> Phase:
        self._set(Phase.SUCCESS if success else Phase.FAILURE, t, note=note)
        return self.phase

    def event_table(self) -> List[dict]:
        return [
            {"time": e.time, "phase": e.phase.value, "stroke": e.stroke, "note": e.note}
            for e in self.events
        ]
