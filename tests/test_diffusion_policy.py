"""Mask, output timing, provenance-compatible model and resume contracts."""
import copy
import numpy as np
import pytest
import torch


def test_masked_loss_excludes_invalid_action_and_tail():
    from sim.diffusion.policy import masked_mse, executable_actions
    pred = torch.zeros(1,28,4)
    target = torch.ones_like(pred)
    pred[:,26:] = 10000
    mask = torch.zeros(1,28,dtype=torch.bool)
    mask[:,1:26] = True
    assert masked_mse(pred,target,mask).item() == 1
    sequence = torch.arange(28).reshape(1,28,1).expand(-1,-1,4)
    assert executable_actions(sequence)[0,:,0].tolist() == list(range(1,26))
    with pytest.raises(ValueError):
        masked_mse(pred,target,torch.zeros_like(mask))


def test_official_encoder_and_unet_shapes_without_allocating_training_state():
    from sim.diffusion.policy import DiffusionPolicy
    stats = {k:dict(scale=[1.]*d,offset=[0.]*d) for k,d in [("state",36),("action",4)]}
    with torch.device("meta"):
        model = DiffusionPolicy(stats)
    assert len(model.obs_encoder.rgb_keys) == 5
    assert not any(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules())
    assert sum(p.numel() for p in model.obs_encoder.parameters()) > 50_000_000
    assert model.unet(torch.zeros(1,28,4,device="meta"),torch.tensor([2],device="meta"),
                      global_cond=torch.zeros(1,5194,device="meta")).shape == (1,28,4)


def test_resume_restores_rng_sampler_and_subsequent_update(tmp_path):
    from sim.diffusion.training import WeightedCursor, save_checkpoint, restore_checkpoint
    from sim.diffusion.vendor.model.diffusion.ema_model import EMAModel
    torch.manual_seed(42)
    model = torch.nn.Linear(3,4)
    ema = EMAModel(copy.deepcopy(model),power=.75)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-4)
    lr = torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1/(s+1))
    sampler = WeightedCursor([.35,.65],42)
    def update():
        sampler.next()
        opt.zero_grad()
        loss=model(torch.randn(1,3)).square().mean()
        loss.backward(); opt.step(); lr.step(); ema.step(model)
    update()
    path = save_checkpoint(tmp_path,model,ema,opt,lr,sampler,
                           step=1,seconds=12.5,identity="abc",stage="full",config={"seed":42})
    update()
    want = copy.deepcopy(model.state_dict())
    cursor = sampler.cursor
    restored = restore_checkpoint(path,model,ema,opt,lr,sampler,"abc",{"seed":42})
    assert restored["seconds"] == 12.5
    update()
    assert sampler.cursor == cursor
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,want[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match="identity"):
        restore_checkpoint(path,model,ema,opt,lr,sampler,"different",{"seed":42})


def test_resource_guard_reserves_checkpoint_space(tmp_path):
    from sim.diffusion.training import ResourceGuard
    guard = ResourceGuard(tmp_path,max_bytes=10,min_free_bytes=0)
    (tmp_path/"owned").write_bytes(b"0123456789")
    with pytest.raises(RuntimeError,match="artifact"):
        guard.check(reserve=1)
