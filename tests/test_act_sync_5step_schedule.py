import numpy as np
import pytest

from sim.act.realtime import ActionChunkScheduler
from sim.config import load_config
from scripts.sync_5step_adapter import (
    SCHEDULE_RECORD,
    SynchronousExecutor,
    adapt_config,
    install_driver_adapter,
    make_sync_scheduler,
    restore_driver_adapter,
)
from scripts.evaluate_sync_5step_episode import capture_partial_snapshot


def test_schedule_config_is_episode_local_and_preserves_model_observation_contract():
    cfg = load_config("configs/five-view-dinov3-frozen-300.yaml")
    before_views = list(cfg.act.camera_views)
    before_stats = (cfg.act.vision_encoder, cfg.act.image_size, cfg.act.ordinary_target_count_only)

    adapted = adapt_config(cfg)

    assert SCHEDULE_RECORD == {
        "prediction_steps": 25,
        "execution_steps": 5,
        "action_hz": 25.0,
        "query_hz": 5.0,
        "temporal_ensemble": False,
        "inference_advances_simulation": False,
        "deterministic": True,
    }
    assert adapted is not cfg
    assert (adapted.act.query_hz, adapted.act.action_hz) == (5.0, 25.0)
    assert (adapted.act.chunk_size, adapted.act.execute_steps) == (25, 5)
    assert adapted.act.temporal_ensemble_coeff is None
    assert adapted.act.realtime_budget_seconds == 0.0
    assert cfg.act.temporal_ensemble_coeff == 0.01
    assert cfg.act.realtime_budget_seconds == 0.2
    assert list(adapted.act.camera_views) == before_views
    assert (adapted.act.vision_encoder, adapted.act.image_size,
            adapted.act.ordinary_target_count_only) == before_stats


def test_synchronous_executor_finishes_prediction_without_advancing_simulated_time():
    clock = {"simulation_time": 4.25}
    with SynchronousExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: clock["simulation_time"])
        assert future.done()
        assert future.result() == 4.25
        assert clock["simulation_time"] == 4.25

        failed = executor.submit(lambda: (_ for _ in ()).throw(ValueError("prediction failed")))
        assert failed.done()
        with pytest.raises(ValueError, match="prediction failed"):
            failed.result()


def test_scheduler_emits_only_indices_zero_through_four_then_starts_fresh_chunk():
    cfg = adapt_config(load_config("configs/five-view-dinov3-frozen-300.yaml"))
    scheduler_type = make_sync_scheduler(ActionChunkScheduler)
    scheduler = scheduler_type(cfg, allow_late=True)
    scheduler.reset(10.0)
    values = np.arange(25 * 4, dtype=np.float32).reshape(25, 4)
    assert scheduler.accept(10.0, values, 10.0)
    assert scheduler.chunk.actions.shape == (5, 4)

    for index in range(5):
        assert np.array_equal(scheduler.action_for(10.0 + index * 0.04), values[index])
    assert scheduler.query_due(10.2)
    scheduler.mark_query(10.2)
    next_values = values + 1000
    assert scheduler.accept(10.2, next_values, 10.2)
    assert np.array_equal(scheduler.action_for(10.2), next_values[0])
    assert scheduler.temporal_ensemble is None
    assert scheduler.query_period == pytest.approx(0.2)
    assert scheduler.action_period == pytest.approx(0.04)


@pytest.mark.parametrize("values", [np.zeros((24, 4)), np.full((25, 4), np.nan)])
def test_scheduler_rejects_non_native_prediction_chunks(values):
    cfg = adapt_config(load_config("configs/five-view-dinov3-frozen-300.yaml"))
    scheduler = make_sync_scheduler(ActionChunkScheduler)(cfg)
    scheduler.reset(0.0)
    with pytest.raises(ValueError):
        scheduler.accept(0.0, values, 0.0)


def test_driver_adapter_is_process_local_and_installs_inline_prediction_executor():
    class FakeHybrid:
        @staticmethod
        def build_driver(_module, _cfg, _context):
            def driver():
                return None
            driver.__globals__["ActionChunkScheduler"] = ActionChunkScheduler
            return driver, object()

    hybrid = FakeHybrid()
    original = hybrid.build_driver
    installed = install_driver_adapter(hybrid)
    try:
        driver, _ = hybrid.build_driver(None, None, None)
        assert driver.__globals__["ThreadPoolExecutor"] is SynchronousExecutor
        assert driver.__globals__["ActionChunkScheduler"].__name__ == "SyncFiveStepScheduler"
    finally:
        restore_driver_adapter(hybrid, installed)
    assert hybrid.build_driver is original


def test_sync_scheduler_fails_closed_if_inference_advances_simulation_time():
    cfg = adapt_config(load_config("configs/five-view-dinov3-frozen-300.yaml"))
    scheduler = make_sync_scheduler(ActionChunkScheduler)(cfg)
    scheduler.reset(0.0)
    with pytest.raises(RuntimeError, match="simulation time advanced"):
        scheduler.accept(0.0, np.zeros((25, 4), dtype=np.float32), 0.001)


def test_partial_failure_snapshot_preserves_measured_count_and_keeps_unknown_null():
    class MeasuredEnv:
        time = 1.25

        def collected_mask(self):
            return np.array([True, True, False, False, False, False])

        def component_positions(self):
            return np.zeros((6, 3), dtype=np.float32)

        def component_quats(self):
            return np.tile([1., 0., 0., 0.], (6, 1))

    class UnreadableEnv(MeasuredEnv):
        def collected_mask(self):
            raise RuntimeError("measurement unavailable")

    cfg = load_config("configs/five-view-dinov3-frozen-300.yaml")
    ik_error = RuntimeError("UR10 IK target is unreachable: residual")
    measured = capture_partial_snapshot(
        ik_error, cfg, {"env": MeasuredEnv(), "trace": [{"t": 1., "collected": 1}],
                        "actions": []})
    assert measured["known_ik_failure"] is True
    assert measured["peak_collected"] == 2
    assert measured["final_collected"] == 2

    unknown = capture_partial_snapshot(
        RuntimeError("unclassified execution issue"), cfg,
        {"env": UnreadableEnv(), "trace": [], "actions": []})
    assert unknown["known_ik_failure"] is False
    assert unknown["peak_collected"] is None
    assert unknown["final_collected"] is None
