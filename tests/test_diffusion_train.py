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
