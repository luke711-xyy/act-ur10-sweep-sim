"""Signal-conditioning blocks used by the hybrid force/position controller."""

from __future__ import annotations

import numpy as np


class LowPassFilter:
    """First-order (exponential) low-pass filter with a cutoff in Hz.

    ``alpha = dt / (RC + dt)`` with ``RC = 1 / (2*pi*f_c)``.  A non-positive
    cutoff disables filtering (pass-through), which is convenient for tests.
    """

    def __init__(self, cutoff_hz: float, dt: float, initial: float = 0.0):
        self.dt = float(dt)
        self.cutoff_hz = float(cutoff_hz)
        if self.cutoff_hz <= 0.0:
            self.alpha = 1.0
        else:
            rc = 1.0 / (2.0 * np.pi * self.cutoff_hz)
            self.alpha = self.dt / (rc + self.dt)
        self.value = float(initial)

    def reset(self, initial: float = 0.0) -> None:
        self.value = float(initial)

    def step(self, sample: float) -> float:
        self.value += self.alpha * (float(sample) - self.value)
        return self.value


class RateLimiter:
    """Slew-rate limiter, used to ramp the desired normal force up and down."""

    def __init__(self, rate: float, dt: float, initial: float = 0.0):
        self.rate = float(rate)
        self.dt = float(dt)
        self.value = float(initial)

    def reset(self, initial: float = 0.0) -> None:
        self.value = float(initial)

    def step(self, target: float) -> float:
        max_delta = self.rate * self.dt
        delta = float(target) - self.value
        self.value += float(np.clip(delta, -max_delta, max_delta))
        return self.value

    def at(self, target: float, tol: float = 1e-3) -> bool:
        return abs(self.value - float(target)) <= tol


class DelayBuffer:
    """Fixed integer-sample delay, used to emulate actuation/communication lag."""

    def __init__(self, steps: int, initial=None):
        self.steps = max(0, int(steps))
        self._buffer = [] if initial is None else [np.array(initial, dtype=float)] * self.steps

    def reset(self, initial) -> None:
        self._buffer = [np.array(initial, dtype=float) for _ in range(self.steps)]

    def step(self, sample):
        sample = np.array(sample, dtype=float)
        if self.steps == 0:
            return sample
        if not self._buffer:
            self._buffer = [sample.copy() for _ in range(self.steps)]
        self._buffer.append(sample)
        return self._buffer.pop(0)
