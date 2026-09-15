"""Unit tests for the 1-D admittance controller."""

import numpy as np
import pytest

from sim.config import load_config
from sim.controllers.admittance import AdmittanceController1D, AdmittanceParams
from sim.controllers.filters import DelayBuffer, LowPassFilter, RateLimiter


def make(dt=0.01, **kw):
    params = AdmittanceParams(kw.pop("m", 1.0), kw.pop("b", 120.0), kw.pop("k", 60.0))
    return AdmittanceController1D(params, dt=dt, **kw)


def test_rejects_non_physical_parameters():
    with pytest.raises(ValueError):
        AdmittanceController1D(AdmittanceParams(0.0, 10.0, 10.0), dt=0.01)
    with pytest.raises(ValueError):
        AdmittanceController1D(AdmittanceParams(1.0, -1.0, 10.0), dt=0.01)
    with pytest.raises(ValueError):
        make(dt=0.0)


def test_zero_error_keeps_delta_at_zero():
    c = make()
    for _ in range(500):
        c.step(3.0, 3.0)
    assert abs(c.delta_z) < 1e-9


def test_sign_convention_under_force_deficit():
    """Measured force below target must command the tool DOWN (delta_z < 0)."""
    c = make()
    for _ in range(50):
        c.step(3.0, 0.0)
    assert c.delta_z < 0.0


def test_sign_convention_under_excess_force():
    """Measured force above target must command the tool UP (delta_z > 0)."""
    c = make()
    for _ in range(50):
        c.step(3.0, 8.0)
    assert c.delta_z > 0.0


def test_converges_to_analytic_steady_state():
    c = make(delta_limit=0.05)
    for _ in range(20000):
        c.step(3.0, 6.0)
    assert c.delta_z == pytest.approx(c.steady_state_delta(3.0, 6.0), abs=1e-6)
    assert c.delta_z == pytest.approx(3.0 / 60.0, abs=1e-6)  # (F_m - F_d) / k_d


def test_delta_saturation_is_respected():
    c = make(delta_limit=0.004)
    for _ in range(5000):
        c.step(50.0, 0.0)
    assert abs(c.delta_z) <= 0.004 + 1e-12
    assert c.saturated


def test_rate_limit_is_respected():
    dt, rate = 0.01, 0.02
    c = make(dt=dt, delta_limit=1.0, rate_limit=rate)
    previous = c.delta_z
    for _ in range(300):
        c.step(200.0, 0.0)
        assert abs(c.delta_z - previous) <= rate * dt + 1e-12
        previous = c.delta_z


def test_reset_clears_state():
    c = make()
    for _ in range(100):
        c.step(3.0, 0.0)
    assert c.delta_z != 0.0
    c.reset()
    assert c.delta_z == 0.0 and c.delta_z_rate == 0.0


def test_closed_loop_with_a_stiff_contact_is_stable_and_tracks():
    """Close the loop around a stiff plant: F = k_env * penetration.

    This is the configuration the controller actually runs in, so it is the one
    the default gains must be stable for.
    """
    cfg = load_config()
    a = cfg.controller.admittance
    dt = 1.0 / float(cfg.sim.control_hz)
    c = AdmittanceController1D(AdmittanceParams(float(a.m), float(a.b), float(a.k)), dt=dt,
                               delta_limit=float(cfg.controller.delta_z_limit),
                               rate_limit=float(cfg.controller.delta_z_rate_limit))
    k_env = float(cfg.end_effector.kp)
    f_des = float(cfg.controller.desired_force)
    z_nominal = -float(cfg.controller.contact_threshold) / k_env  # contact detection height
    history = []
    force = float(cfg.controller.contact_threshold)
    for _ in range(4000):
        delta = c.step(f_des, force)
        z_cmd = z_nominal + delta
        force = max(0.0, -k_env * z_cmd)      # table at z = 0, pressing-positive
        history.append(force)
    history = np.asarray(history)
    assert np.all(np.isfinite(history))
    assert history.max() < float(cfg.controller.safe_max_force)
    assert history[-1] == pytest.approx(f_des, rel=0.05)     # steady-state within 5 %
    assert history.max() <= f_des * 1.35                     # bounded overshoot


def test_low_pass_filter_dc_gain_and_smoothing():
    f = LowPassFilter(10.0, 0.01)
    for _ in range(2000):
        f.step(5.0)
    assert f.value == pytest.approx(5.0, abs=1e-6)
    # A 10 Hz first-order filter at 100 Hz attenuates a Nyquist-rate square wave
    # by alpha / (2 - alpha) ~ 0.24, so require at least a 3x reduction.
    f.reset(5.0)
    noisy = [5.0 + (1.0 if i % 2 else -1.0) for i in range(400)]
    out = [f.step(v) for v in noisy]
    assert np.std(out[200:]) < np.std(noisy[200:]) / 3.0
    assert abs(np.mean(out[200:]) - 5.0) < 0.05


def test_rate_limiter_slews_and_reports_arrival():
    r = RateLimiter(15.0, 0.01)
    assert not r.at(3.0)
    for _ in range(100):
        r.step(3.0)
    assert r.at(3.0, tol=1e-3)
    for _ in range(100):
        r.step(0.0)
    assert r.at(0.0, tol=1e-3)


def test_delay_buffer_shifts_by_n_steps():
    d = DelayBuffer(3)
    d.reset([0.0])
    outputs = [float(d.step([float(i)])[0]) for i in range(8)]
    assert outputs[:3] == [0.0, 0.0, 0.0]
    assert outputs[3:] == [0.0, 1.0, 2.0, 3.0, 4.0]
