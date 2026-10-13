from types import SimpleNamespace

import numpy as np
import pytest

from sim.config import load_config


@pytest.mark.parametrize("completion_frame,expected_frames,expected_success", [
    (None, 2000, False),
    (1600, 1600, True),
])
def test_inference_continues_through_warnings_until_success_or_2000_frames(
        tmp_path, monkeypatch, completion_frame, expected_frames, expected_success):
    pytest.importorskip("torch")
    import sim.act.evaluate as evaluate

    class FakeEnv:
        last_instance = None

        def __init__(self, cfg, seed=0):
            self.cfg = cfg
            self.seed = seed
            self.time = 0.0
            self.layout = [object()] * 6
            self.ee = SimpleNamespace(tcp_yaw=lambda: 0.25)
            self._tcp = np.array([0.42, 0.0, 0.10], dtype=np.float64)
            self.step_calls = 0
            self.rejected_commands = 0
            type(self).last_instance = self

        def reset(self, seed=None):
            return None

        def tcp(self):
            return self._tcp.copy()

        def normal_force(self):
            return 100.0 if self.step_calls <= 8 else 0.0

        def step_control(self, command):
            if self.rejected_commands == 0:
                self.rejected_commands += 1
                raise RuntimeError("simulated policy IK rejection")
            self.step_calls += 1
            self._tcp[:] = [self._tcp[0] + 0.01, command.y, command.z]
            self.time += 1.0 / float(self.cfg.sim.control_hz)

        def collected_mask(self):
            mask = np.zeros(6, dtype=bool)
            if completion_frame is not None and self.step_calls >= completion_frame * 4:
                mask[:] = True
            return mask

        def partial_collection_overlap_mask(self):
            return np.zeros(6, dtype=bool)

        def unsafe_robot_collision(self):
            return self.step_calls >= 20

        def lost_mask(self):
            return np.full(6, self.step_calls >= 25, dtype=bool)

        def close(self):
            return None

    class FailingPolicy:
        def eval(self):
            return self

        def reset(self):
            return None

        def predict_action_chunk(self, _batch):
            raise RuntimeError("simulated policy prediction failure")

    class FakeObservationBuilder:
        def __init__(self, *_args, **_kwargs):
            pass

        def torch_batch(self, _observation):
            return {}

        def observe(self, _env, contact_latched=False):
            return {"contact_latched": contact_latched}

        def advance_state_history(self, _env, contact_latched=False):
            return None

    class FakePolicyConfig:
        output_features = {"action": SimpleNamespace(shape=(4,))}

    monkeypatch.setattr(evaluate, "SweepEnv", FakeEnv)
    monkeypatch.setattr(evaluate, "ACTObservationBuilder", FakeObservationBuilder)
    monkeypatch.setattr(
        evaluate, "build_act_policy",
        lambda cfg, pretrained_path=None: (FailingPolicy(), FakePolicyConfig()),
    )
    monkeypatch.setattr(
        evaluate, "build_act_processors",
        lambda *args, **kwargs: (lambda value: value, lambda value: value),
    )
    cfg = load_config(overrides=[
        "sim.real_time=false", "sim.control_hz=100", "act.action_hz=25",
        "act.query_hz=5", "task.target_count=6",
        "episode.max_time=1000", "episode.max_frames=1",
        "episode.max_contact_path=0.05", "controller.safe_max_force=50",
        f"act.dataset_dir={tmp_path / 'missing'}",
    ])

    result = evaluate.run_act_episode(cfg, seed=17, model_path=None, preview=True)

    assert result.sampled_frames == expected_frames
    assert result.failure_reason == (
        "" if expected_success else "2000-frame timeout before task completion"
    )
    assert result.termination_reason == (
        "exact target count reached" if expected_success else ""
    )
    assert result.scheduler_queries > 1
    assert FakeEnv.last_instance.rejected_commands == 1
    assert result.success is expected_success


def _run_collection_schedule(tmp_path, monkeypatch, schedule, *, target=6,
                             partial_schedule=None, policy=None):
    """Exercise the real inference loop without model/rendering workloads."""
    pytest.importorskip("torch")
    import sim.act.evaluate as evaluate

    class ScheduledEnv:
        def __init__(self, cfg, seed=0):
            self.cfg = cfg
            self.time = 0.0
            self.layout = [object()] * 6
            self.ee = SimpleNamespace(tcp_yaw=lambda: 0.0)
            self.steps = 0
            self._tcp = np.array([0.42, 0.0, 0.10], dtype=np.float64)

        def reset(self, seed=None):
            pass

        def tcp(self):
            return self._tcp.copy()

        def normal_force(self):
            return 0.0

        def step_control(self, command):
            self.steps += 1
            self.time = self.steps / 100.0
            self._tcp[:] = [command.x, command.y, command.z]

        def collected_mask(self):
            return np.asarray(schedule(self.steps), dtype=bool)

        def partial_collection_overlap_mask(self):
            return (np.asarray(partial_schedule(self.steps), dtype=bool)
                    if partial_schedule else np.zeros(6, dtype=bool))

        def close(self):
            pass

    class HoldingPolicy:
        def eval(self):
            return self

        def reset(self):
            pass

        def predict_action_chunk(self, _batch):
            raise RuntimeError("no model output in collection timing fixture")

    class ObservationBuilder:
        def __init__(self, *_args, **_kwargs):
            pass

        def observe(self, _env, contact_latched=False):
            return {}

        def torch_batch(self, _observation):
            return {}

        def advance_state_history(self, _env, contact_latched=False):
            pass

    monkeypatch.setattr(evaluate, "SweepEnv", ScheduledEnv)
    monkeypatch.setattr(evaluate, "ACTObservationBuilder", ObservationBuilder)
    monkeypatch.setattr(evaluate, "build_act_policy", lambda *_args, **_kwargs: (
        policy if policy is not None else HoldingPolicy(), SimpleNamespace(
            output_features={"action": SimpleNamespace(shape=(4,))})))
    monkeypatch.setattr(evaluate, "build_act_processors", lambda *_args, **_kwargs: (
        lambda value: value, lambda value: value))
    cfg = load_config(overrides=[
        "sim.real_time=false", "sim.control_hz=100", "act.action_hz=25",
        f"task.target_count={target}", f"act.dataset_dir={tmp_path / 'missing'}",
    ])
    return evaluate.run_act_episode(cfg, preview=True)


def test_terminal_episode_drains_active_prediction_before_return(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import sim.act.evaluate as evaluate
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    timers = []
    class SlowPolicy:
        def eval(self):
            return self
        def reset(self):
            pass
        def predict_action_chunk(self, batch):
            timer = threading.Timer(0.5, release.set)
            timers.append(timer)
            timer.start()
            started.set()
            assert release.wait(3)
            finished.set()
            raise RuntimeError("test prediction ends after physical termination")
    class StartedExecutor(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            future = super().submit(*args, **kwargs)
            assert started.wait(3)
            return future
    monkeypatch.setattr(evaluate, "ThreadPoolExecutor", StartedExecutor)
    try:
        result = _run_collection_schedule(tmp_path, monkeypatch,
            lambda step: _mask(0) if step >= 1 else _mask(), target=1, policy=SlowPolicy())
        assert result.success and result.elapsed == pytest.approx(0.01)
        assert finished.is_set(), "CLI lease must outlive all active prediction work"
    finally:
        release.set()
        for timer in timers:
            timer.join()


def _mask(*indices):
    mask = np.zeros(6, dtype=bool)
    mask[list(indices)] = True
    return mask


def test_first_full_entry_sets_two_second_deadline_even_if_parts_leave(
        tmp_path, monkeypatch):
    # A 100 Hz entry between image frames must not be missed. Later entries
    # and complete loss must not restart or cancel the original deadline.
    def schedule(step):
        if 53 <= step < 200:
            return _mask(0)
        if 200 <= step < 220:
            return _mask(0, 1, 2, 3)
        return _mask()

    result = _run_collection_schedule(tmp_path, monkeypatch, schedule)

    assert result.elapsed == pytest.approx(2.53)
    assert result.sampled_frames == 64
    assert len(result.trace) == 253
    assert result.first_entry_time == pytest.approx(0.53)
    assert result.collected == result.peak_collected == 4
    assert result.final_collected == 0
    assert not result.final_collected_mask.any()
    assert not result.success
    assert result.termination_reason == "2-second collection window elapsed"
    assert "2000-frame timeout" not in result.failure_reason


def test_peak_count_is_simultaneous_not_cumulative_distinct_entries(
        tmp_path, monkeypatch):
    def schedule(step):
        if 53 <= step < 90:
            return _mask(0)
        if 120 <= step < 180:
            return _mask(1)
        return _mask()

    result = _run_collection_schedule(tmp_path, monkeypatch, schedule)

    assert result.elapsed == pytest.approx(2.53)
    assert result.collected == result.peak_collected == 1
    assert result.final_collected == 0


def test_exact_target_finishes_on_control_step_without_extra_push(
        tmp_path, monkeypatch):
    result = _run_collection_schedule(
        tmp_path, monkeypatch,
        lambda step: _mask(0, 1, 2) if step >= 53 else _mask(), target=3)

    assert result.success
    assert result.elapsed == pytest.approx(0.53)
    assert len(result.trace) == 53
    assert result.sampled_frames == 14
    assert result.collected == result.final_collected == 3
    assert result.termination_reason == "exact target count reached"


def test_excess_count_is_not_success_but_still_obeys_entry_deadline(
        tmp_path, monkeypatch):
    result = _run_collection_schedule(
        tmp_path, monkeypatch,
        lambda step: _mask(0, 1, 2, 3) if step >= 53 else _mask(), target=3)

    assert not result.success
    assert result.elapsed == pytest.approx(2.53)
    assert result.collected == result.final_collected == 4
    assert "unexpected component" in result.failure_reason


def test_partial_overlap_does_not_count_as_exact_completion(tmp_path, monkeypatch):
    result = _run_collection_schedule(
        tmp_path, monkeypatch,
        lambda step: _mask(0, 1, 2) if step >= 53 else _mask(), target=3,
        partial_schedule=lambda step: _mask(3) if step >= 53 else _mask())

    assert not result.success
    assert result.elapsed == pytest.approx(2.53)
    assert result.collected == result.final_collected == 3
    assert "partially overlaps" in result.failure_reason


def test_partial_only_entry_retains_2000_frame_cap(tmp_path, monkeypatch):
    result = _run_collection_schedule(
        tmp_path, monkeypatch, lambda step: _mask(),
        partial_schedule=lambda step: _mask(0) if step >= 53 else _mask())

    assert not result.success
    assert result.sampled_frames == 2000
    assert result.elapsed == pytest.approx(80.0)
    assert result.first_entry_time is None
    assert result.collected == result.final_collected == 0
    assert result.failure_reason == "2000-frame timeout before task completion"


def test_late_full_entry_does_not_extend_original_2000_frame_budget(
        tmp_path, monkeypatch):
    result = _run_collection_schedule(
        tmp_path, monkeypatch,
        lambda step: _mask(0) if step >= 7999 else _mask())

    assert result.sampled_frames == 2000
    assert result.elapsed == pytest.approx(80.0)
    assert result.first_entry_time == pytest.approx(79.99)
    assert result.collected == result.peak_collected == result.final_collected == 1
    assert not result.success
    assert result.failure_reason == "2000-frame timeout before task completion"
