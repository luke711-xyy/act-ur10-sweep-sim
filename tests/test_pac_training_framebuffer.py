from types import SimpleNamespace

import numpy as np


def test_gpu_startup_benchmark_configures_framebuffer_before_render(monkeypatch):
    from pac_act import ppo, training
    from sim.act import interface
    from sim.environments import sweep_env

    models = []
    render_calls = []

    class FakeEnv:
        def __init__(self, _cfg, seed):
            self.seed = seed
            self.model = SimpleNamespace(vis=SimpleNamespace(global_=SimpleNamespace(
                offwidth=640, offheight=480)))
            models.append(self.model)

        def reset(self, *, seed):
            assert seed == self.seed

        def component_positions(self):
            return np.zeros((1, 3), dtype=np.float32)

        def component_quats(self):
            return np.array([[1, 0, 0, 0]], dtype=np.float32)

        def render_rgb(self, camera, size):
            render_calls.append((camera, size))
            assert self.model.vis.global_.offwidth >= size[0]
            assert self.model.vis.global_.offheight >= size[1]
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        def close(self):
            pass

    class FakeBuilder:
        def __init__(self, *_args, **_kwargs):
            pass

        def observe(self, _env, *, contact_latched):
            assert contact_latched is False
            return {"state": np.zeros(1, dtype=np.float32)}

    monkeypatch.setattr(training, "cfg_for", lambda _bundle, _goal: object())
    monkeypatch.setattr(sweep_env, "SweepEnv", FakeEnv)
    monkeypatch.setattr(interface, "ACTObservationBuilder", FakeBuilder)
    monkeypatch.setattr(ppo, "capture_rng", lambda _device: "rng")
    monkeypatch.setattr(ppo, "restore_rng", lambda _rng: None)
    monkeypatch.setattr(training, "_benchmark_backward", lambda *_args: {"passed": True})

    collection = SimpleNamespace(
        bundle="unused-bundle",
        policy=SimpleNamespace(image_keys=[], obs_keys=["state"]),
        registry=SimpleNamespace(register=lambda *_args: None),
        guard=SimpleNamespace(poll=lambda: None),
    )
    trainer = SimpleNamespace(optimizer=SimpleNamespace(zero_grad=lambda **_kwargs: None))
    result = training.benchmark(collection, trainer)

    assert result == {"passed": True}
    assert len(models) == 6
    assert render_calls == [("overhead_cam", (960, 640))] * 6
    assert all(model.vis.global_.offwidth >= 960 for model in models)
    assert all(model.vis.global_.offheight >= 640 for model in models)
