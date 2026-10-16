from types import SimpleNamespace
import numpy as np
import pytest


def test_failure_measurement_survives_native_transaction_restore():
    from pac_act.hybrid import HybridEpisode
    count = [2]
    world = HybridEpisode.__new__(HybridEpisode)
    world.env = SimpleNamespace(collected_mask=lambda:np.arange(6)<count[0],time=1.2)
    world.path = None
    world.measured_count = 0
    world.capture_failure(ValueError('IK unreachable'))
    count[0] = 0  # The legacy acquisition transaction restores its last state.
    assert world.final_count() == 2


def test_failed_measurement_is_unknown_not_zero():
    from pac_act.hybrid import HybridEpisode
    def failed():
        raise ValueError('no valid measurement')
    world = HybridEpisode.__new__(HybridEpisode)
    world.env = SimpleNamespace(collected_mask=failed,time=1.2)
    world.path = None
    world.capture_failure(ValueError('physics failed'))
    assert world.final_count() is None


def test_budget_stop_inside_action_retains_executed_prefix_record(monkeypatch,tmp_path):
    from test_pac_execution import wired_episode
    from pac_act.safety import StopExperiment
    from pac_act.storage import read_rows
    world = wired_episode(monkeypatch,tmp_path,learned=True)
    def stop(spec):
        raise StopExperiment('one million')
    world.spec = dict(seed=6000000)
    world.on_action = stop
    with pytest.raises(StopExperiment):
        world.step_block(np.zeros((5,4)),learned=True)
    rows = read_rows(tmp_path/'blocks.jsonl')
    assert len(rows) == 1 and rows[0]['executed'] == 1
    assert world.active_actions == 1 and len(world.saved_actions) == 1
