import json
import pytest


def test_acceptance_seed_mapping_and_no_retries(tmp_path):
    from sim.diffusion.acceptance import acceptance_cases, reserve_case
    cases = acceptance_cases()
    assert [(c["seed"],c["target_count"]) for c in cases] == [
        (81000001,1),(81000002,2),(81000003,3),
        (81000004,4),(81000005,5),(81000006,6)]
    log = tmp_path/"attempts.json"
    reserve_case(log,cases[0])
    with pytest.raises(RuntimeError,match="already attempted"):
        reserve_case(log,cases[0])
    assert json.loads(log.read_text())[0]["state"] == "started"


def test_layout_duplicate_stops_without_replacement():
    from sim.diffusion.acceptance import require_new_layout
    with pytest.raises(RuntimeError,match="duplicate"):
        require_new_layout("abc",{"abc"})
    require_new_layout("new",{"abc"})


def test_acceptance_requires_checkpoint_run_and_actual_dataset_identity():
    from sim.diffusion.acceptance import require_training_identity
    require_training_identity("a",{"identity":"a"},{"identity":"a","layout_fingerprints":["x"]})
    for checkpoint,run,audit in [
        ("b",{"identity":"a"},{"identity":"a"}),
        ("a",{"identity":"b"},{"identity":"a"}),
        ("a",{"identity":"a"},{"identity":"other"}),
    ]:
        with pytest.raises(ValueError,match="identity"):
            require_training_identity(checkpoint,run,audit)


def test_pipeline_never_accepts_stopped_failed_or_zero_update_training():
    from sim.diffusion.pipeline import ready_for_acceptance
    valid=dict(state="training_complete",pilot=dict(passed=True),
               full=dict(completed_updates=1))
    assert ready_for_acceptance(valid)
    for state in ("failed","stopped","pilot_passed","pilot_running"):
        assert not ready_for_acceptance(dict(valid,state=state))
    assert not ready_for_acceptance(dict(valid,full=dict(completed_updates=0)))


def test_acceptance_stop_flag_is_checked_before_starting_any_case():
    from sim.diffusion.acceptance import check_stop
    with pytest.raises(InterruptedError,match="stopped"):
        check_stop({"requested":True})
    check_stop({"requested":False})
