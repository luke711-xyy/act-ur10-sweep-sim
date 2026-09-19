import numpy as np


class FakeEnv:
    time = 1.25
    layout = [object()] * 6

    def tcp(self):
        return np.array([0.20, 0.01, 0.03], dtype=np.float32)

    class EE:
        @staticmethod
        def joint_state():
            return np.zeros(6, dtype=np.float32)

        @staticmethod
        def tcp_yaw():
            return 0.2

    ee = EE()

    def wrench(self):
        return np.arange(6, dtype=np.float32)

    def render_rgb(self, *_args, **_kwargs):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def render_wrist_rgb(self, *_args, **_kwargs):
        return np.zeros((8, 8, 3), dtype=np.uint8)


class FakeFrontend:
    def observe(self, **_kwargs):
        from sim.act.object_interface import ObjectPerceptionFrame

        return ObjectPerceptionFrame(
            object_tokens=np.zeros((6, 29), dtype=np.float32),
            object_valid=np.array([True, True, False, False, False, False]),
            instance_bev=np.zeros((6, 128, 160), dtype=bool),
            bev=np.zeros((6, 128, 160), dtype=np.float32),
            visual_tracked_count=2,
            visual_full_in_tray_count=1,
        )


def test_objectact_observation_builder_uses_visual_counts_not_simulator_collection_truth():
    from sim.act.object_interface import ObjectACTObservationBuilder

    class Config:
        act = type("Act", (), {"image_size": [8, 8]})()
        task = {"target_count": 3}

        def get_path(self, key, default=None):
            return self.task.get(key.split(".")[-1], default)

    observation = ObjectACTObservationBuilder(Config(), frontend=FakeFrontend()).observe(
        FakeEnv(), contact_latched=True
    )
    assert observation["observation.robot_state"].shape == (36,)
    assert observation["observation.task_state"].shape == (6,)
    np.testing.assert_allclose(observation["observation.task_state"][:3], [2 / 6, 3 / 6, 1 / 6])
    assert observation["observation.object_tokens"].shape == (6, 29)
    assert observation["observation.instance_bev"].shape == (6, 128, 160)
    assert observation["contact_latched"] is True


def test_objectact_control_masks_only_applied_z_after_contact():
    from sim.act.object_interface import apply_objectact_action

    action = np.array([0.1, -0.2, 0.03, 0.4], dtype=np.float32)
    np.testing.assert_allclose(apply_objectact_action(action, False, 0.0), action)
    applied = apply_objectact_action(action, True, -0.004)
    np.testing.assert_allclose(applied[[0, 1, 3]], action[[0, 1, 3]])
    assert applied[2] == -0.004
