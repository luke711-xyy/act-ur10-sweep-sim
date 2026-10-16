"""Process-local ACT scheduler adapter for the synchronous 25/5 eval."""

from __future__ import annotations

from concurrent.futures import Future


SCHEDULE_RECORD = {
    "prediction_steps": 25,
    "execution_steps": 5,
    "action_hz": 25.0,
    "query_hz": 5.0,
    "temporal_ensemble": False,
    "inference_advances_simulation": False,
    "deterministic": True,
}


def adapt_config(cfg):
    """Copy a frozen eval config and change only ACT chunk scheduling fields."""
    adapted = cfg.copy()
    for key, value in (
        ("act.query_hz", 5.0),
        ("act.action_hz", 25.0),
        ("act.chunk_size", 25),
        ("act.execute_steps", 5),
        ("act.temporal_ensemble_coeff", None),
        ("act.realtime_budget_seconds", 0.0),
    ):
        adapted.set_path(key, value)
    return adapted


class SynchronousExecutor:
    """Executor-compatible inline calls: the caller cannot step physics mid-forward."""

    def __init__(self, max_workers=1, **_kwargs):
        if max_workers != 1:
            raise ValueError("synchronous ACT inference expects one worker")
        self.closed = False

    def submit(self, function, /, *args, **kwargs):
        if self.closed:
            raise RuntimeError("ACT inference executor is shut down")
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.shutdown(wait=True)
        return False

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.closed = True


def make_sync_scheduler(base_scheduler):
    """Require a fresh 25x4 prediction and expose only its first five actions."""

    class SyncFiveStepScheduler(base_scheduler):
        def __init__(self, cfg, allow_late=False):
            super().__init__(cfg, allow_late=False)
            if (abs(self.query_period - 0.2) > 1e-9
                    or abs(self.action_period - 0.04) > 1e-9
                    or self.execute_steps != 5 or abs(self.budget) > 1e-12
                    or self.temporal_ensemble is not None):
                raise ValueError("invalid synchronous five-step ACT schedule")

        def accept(self, issued_at, actions, now):
            import numpy as np

            values = np.asarray(actions, dtype=np.float32)
            if values.shape != (25, 4) or not np.isfinite(values).all():
                raise ValueError("synchronous ACT prediction must be finite with shape (25, 4)")
            if abs(float(now) - float(issued_at)) > 1e-9:
                raise RuntimeError("simulation time advanced during synchronous ACT prediction")
            accepted = super().accept(issued_at, values[:5], now)
            if not accepted:
                raise RuntimeError("fresh synchronous ACT chunk was unexpectedly rejected")
            return True

    SyncFiveStepScheduler.__name__ = "SyncFiveStepScheduler"
    return SyncFiveStepScheduler


def install_driver_adapter(hybrid_trial4_module, on_driver_error=None):
    """Patch only this Python process's next Hybrid driver construction."""
    original = hybrid_trial4_module.build_driver

    def build_sync_driver(module, cfg, context):
        driver, governor = original(module, cfg, context)
        namespace = driver.__globals__
        namespace["ActionChunkScheduler"] = make_sync_scheduler(
            namespace["ActionChunkScheduler"])
        namespace["ThreadPoolExecutor"] = SynchronousExecutor

        if on_driver_error is None:
            return driver, governor

        def guarded_driver(*args, **kwargs):
            try:
                return driver(*args, **kwargs)
            except BaseException as error:
                on_driver_error(error, cfg, context)
                raise

        return guarded_driver, governor

    hybrid_trial4_module.build_driver = build_sync_driver
    return original


def restore_driver_adapter(hybrid_trial4_module, original):
    hybrid_trial4_module.build_driver = original
