import pytest


def test_pilot_requires_loss_drop_both_updates_and_finite():
    from sim.diffusion.train import pilot_passes
    assert pilot_passes(1.,.69,{"vision":True,"unet":True})
    assert not pilot_passes(1.,.71,{"vision":True,"unet":True})
    assert not pilot_passes(1.,.1,{"vision":False,"unet":True})
    assert not pilot_passes(1.,float("nan"),{"vision":True,"unet":True})


def test_training_limits_cannot_expand_approved_budget():
    from sim.diffusion.train import training_config
    assert training_config(100)["max_steps"] == 100
    assert training_config(None)["max_steps"] == 100000
    for steps in (0,-1,100001):
        with pytest.raises(ValueError):
            training_config(steps)


def test_pilot_only_never_dispatches_full_resume_and_requires_current_sidecar():
    from sim.diffusion.train import validate_resume
    payload=dict(stage="full",step=500,seconds=2000,stage_elapsed=100)
    book=dict(budget_seconds=2000,full=dict(completed_updates=500,elapsed_seconds=100))
    with pytest.raises(ValueError,match="pilot.only"):
        validate_resume(book,payload,pilot_only=True)
    payload=dict(stage="pilot",step=25,seconds=1790,stage_elapsed=1790)
    with pytest.raises(ValueError,match="status"):
        validate_resume({},payload,pilot_only=False)
    stale=dict(budget_seconds=1500,pilot=dict(completed_updates=25,elapsed_seconds=1500))
    with pytest.raises(ValueError,match="stale"):
        validate_resume(stale,payload,pilot_only=False)
    fresh=dict(budget_seconds=1790,pilot=dict(completed_updates=25,
                elapsed_seconds=1790,initial_loss=1,initial_parameter_hashes={}))
    assert validate_resume(fresh,payload,pilot_only=False) == "pilot"


def test_a_completed_pilot_is_never_trained_again_when_starting_full():
    from sim.diffusion.train import pilot_needed
    assert pilot_needed("pilot",{}) is True
    assert pilot_needed("pilot",{"pilot":{"passed":True}}) is False
    assert pilot_needed("full",{"pilot":{"passed":True}}) is False
