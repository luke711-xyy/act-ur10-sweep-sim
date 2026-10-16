import pytest


def test_completed_validation_is_returned_without_reexecuting_layouts(tmp_path):
    from pac_act.training import evaluate_set
    from pac_act.storage import RunStore, append_json
    from pac_act.identity import seed_spec
    store = RunStore(tmp_path)
    summary = dict(arm='baseline', episodes=24, protocol_complete=True)
    append_json(tmp_path/'validations.jsonl', summary)
    class Guard:
        def poll(self):
            raise AssertionError('completed validation must not execute')
    result = evaluate_set(None, None, store, None, Guard(), arm='baseline')
    assert result == summary
    assert not list((tmp_path/'attempts').iterdir())


def test_resuming_validation_refuses_to_retry_an_interrupted_opportunity(tmp_path):
    from pac_act.training import validation_pending
    from pac_act.storage import RunStore
    from pac_act.identity import seed_spec
    store = RunStore(tmp_path)
    specs = [seed_spec('validation', i, 'baseline') for i in range(2)]
    store.claim(specs[0])
    with pytest.raises(RuntimeError, match='unresolved'):
        validation_pending(store, specs)
    store.commit(dict(**specs[0], outcome='interrupted', final_collected=None, success=None))
    pending, rows = validation_pending(store, specs)
    assert [r['seed'] for r in pending] == [7000001]
    assert rows[0]['success'] is None


def test_checkpoint_failure_does_not_publish_update_or_advance_used_count(tmp_path, monkeypatch):
    from pac_act.training import commit_update
    from pac_act.storage import Checkpoints
    import residual_rl.storage as writer
    def fail(*args, **kwargs):
        raise OSError('injected disk failure')
    monkeypatch.setattr(writer, 'save_checkpoint_payload', fail)
    cp = Checkpoints(tmp_path/'checkpoints', 'id')
    counts = dict(sampled_actions=160, used_actions=0, used_blocks=0, updates=0)
    new_counts = dict(sampled_actions=160, used_actions=160, used_blocks=32, updates=1)
    with pytest.raises(OSError, match='injected disk failure'):
        commit_update(cp, dict(counters=new_counts), counts,
                      dict(accepted=True, update=1), tmp_path)
    assert counts['used_actions'] == 0 and counts['updates'] == 0
    assert not (tmp_path/'updates.jsonl').exists()
    assert cp.index()['slots'] == []


def test_durable_update_can_recover_missing_audit_without_duplication(tmp_path):
    from pac_act.training import reconcile_update_audit
    from pac_act.storage import read_rows
    state = dict(last_update_audit=dict(update=2, accepted=True, value_loss=5.))
    reconcile_update_audit(tmp_path, state)
    reconcile_update_audit(tmp_path, state)
    rows = read_rows(tmp_path/'updates.jsonl')
    assert len(rows) == 1 and rows[0]['value_loss'] == 5.


def test_zero_execution_IK_end_seals_previous_transition_without_fake_sample():
    from pac_act.lifecycle import seal_terminal
    rows = [dict(seed=6000000, next_value=8., terminated=False)]
    seal_terminal(rows, 6000000)
    assert rows == [dict(seed=6000000, next_value=0., terminated=True)]
    rows = [dict(seed=6000000, next_value=8., terminated=False)]
    seal_terminal(rows, 6000006)
    assert rows[0]['next_value'] == 8.  # Never alter another episode's prefix.
