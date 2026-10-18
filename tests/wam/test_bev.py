"""Hand-checked geometry, real attention and permutation invariant losses."""
from dataclasses import replace
import importlib
import numpy as np
import pytest
import torch

from wam_sweep.contracts import CAMERA_VIEWS
from wam_sweep.native import CameraCalibration, CaptureMetadata


def api():
    assert importlib.util.find_spec('wam_sweep.vision') is not None, 'Task6 vision package missing'
    return importlib.import_module('wam_sweep.vision')


def frame(timestamp=4., translation=0.):
    v = api()
    # Camera at z=1, identity rotation: MuJoCo looks down negative z,
    # positive image v is negative world y. A feature ramp encodes pixel u/v.
    k = np.array([[100.,0,159.5],[0,100.,159.5],[0,0,1.]])
    c = np.eye(4)
    c[:3,3] = [translation,0,1]
    cameras = {view:CameraCalibration(k.copy(), c.copy()) for view in CAMERA_VIEWS}
    spatial = {view:{layer:torch.ones((1,ch,size,size)) for layer,ch,size in
                        [('layer1',64,80),('layer2',128,40),('layer4',512,10)]} for view in CAMERA_VIEWS}
    packet = v.ActFeaturePacket(timestamp,spatial,'manifest','model','processor')
    return packet, CaptureMetadata(timestamp,cameras,'manifest')


def test_projection_convention_frustum_and_wrist_history():
    v = api()
    packet, calibration = frame()
    encoder = v.BEVEncoder()
    grid, mask = encoder.project_points(torch.tensor([[0.,0.,0.],[.1,.2,0.],[3.,0.,0.],[0.,0.,2.]]),
                                        calibration.cameras['wrist'])
    # grid_sample normalized pixel centers with align_corners=False.
    torch.testing.assert_close(grid[:2],torch.tensor([[0.,0.],[.0625,-.125]]),atol=1e-6,rtol=0)
    assert mask.tolist() == [True,True,False,False]
    old = calibration.cameras['wrist']
    moved = replace(old,camera_to_world=old.camera_to_world.copy())
    moved.camera_to_world[0,3] = .2
    shifted,_ = encoder.project_points(torch.tensor([[0.,0.,0.]]),moved)
    torch.testing.assert_close(shifted,torch.tensor([[-.125,0.]]),atol=1e-6,rtol=0)
    again,_ = encoder.project_points(torch.tensor([[0.,0.,0.]]),old)
    torch.testing.assert_close(again,torch.zeros(1,2),rtol=0,atol=0)
    with pytest.raises(ValueError,match='timestamp'):
        encoder(packet,replace(calibration,timestamp=4.2))
    with pytest.raises(ValueError,match='identity'):
        encoder(packet,replace(calibration,input_hash='other'))
    with pytest.raises(ValueError,match='camera'):
        encoder(packet,replace(calibration,cameras=dict(reversed(list(calibration.cameras.items())))))
    # Safe dataclass fields contain mutable mappings; consumers must revalidate.
    packet.spatial['wrist']['native36'] = torch.zeros(36)
    with pytest.raises(ValueError,match='spatial'):
        encoder(packet,calibration)


def test_fp32_bilinear_sampler_matches_independent_grid_sample_values_and_gradients():
    v = api()
    assert hasattr(v.BEVEncoder,'sample_plane'), 'MPS differentiable bilinear sampler missing'
    image = torch.arange(35,dtype=torch.float32).reshape(1,1,5,7).requires_grad_()
    grid = torch.tensor([[[[-.7,.4],[0.,0.],[1.,1.],[1.2,.9],[-1.2,-1.2]]]])
    expected = torch.nn.functional.grid_sample(image,grid,align_corners=False)
    actual = v.BEVEncoder.sample_plane(image,grid)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    expected_grad, = torch.autograd.grad(expected.sum(),image,retain_graph=True)
    actual_grad, = torch.autograd.grad(actual.sum(),image)
    torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-6,atol=1e-6)


def test_packet_rejects_native_payload_and_estimator_checkpoint_replay():
    v = api()
    from wam_sweep.contracts import InputIdentity
    import copy
    packet, calibration = frame()
    with pytest.raises((ValueError,TypeError),match='receipt'):
        replace(packet,receipt={'observation.state':torch.zeros(36)})
    identity = InputIdentity('/fixture','manifest','model',dict(perception='perception',control='control',preprocessing='processor',config='config'))
    spec = v.BEVSpec(resolution=.04)
    stored = {4.:calibration,4.2:replace(calibration,timestamp=4.2)}
    estimator = v.VisualStateEstimator(v.BEVEncoder(spec),v.SixQueryDetector(spec),identity,stored.__getitem__)
    checkpoint = estimator.checkpoint()
    other = v.VisualStateEstimator(v.BEVEncoder(spec),v.SixQueryDetector(spec),identity,stored.__getitem__)
    other.load_checkpoint(checkpoint)
    assert other.snapshot()['identity'] == estimator.snapshot()['identity']
    broken = copy.deepcopy(checkpoint)
    del broken['manifest']['input_identity']['native_hashes']['control']
    with pytest.raises(ValueError,match='identity'):
        other.load_checkpoint(broken)
    first = estimator.estimate(packet)
    state = estimator.snapshot()
    assert set(state) == {'schema','identity','timestamp','tracker'}
    future = replace(packet,timestamp=4.2)
    expected = estimator.estimate(future)
    estimator.restore(state)
    actual = estimator.estimate(future)
    assert len(actual) == len(expected)
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a.center_xy,b.center_xy)
        np.testing.assert_array_equal(a.position_covariance,b.position_covariance)
    with pytest.raises(ValueError,match='identity'):
        other.estimate(replace(packet,preprocessing_hash='poison'))


def test_multiscale_masked_bev_covers_table_with_margin_and_gradients():
    v = api()
    packet, calibration = frame(translation=10.)  # all views off table
    encoder = v.BEVEncoder()
    result = encoder(packet,calibration)
    assert result.features.shape == (1,128,168,208)
    assert result.valid.shape == (1,1,168,208)
    assert not result.valid.any() and torch.count_nonzero(result.features) == 0
    assert encoder.spec.x_bounds == (-.52,.52) and encoder.spec.y_bounds == (-.42,.42)
    packet, calibration = frame()
    output = encoder(packet,calibration)
    assert output.valid.all()
    output.features.square().mean().backward()
    assert all(encoder.projections[layer].weight.grad.abs().sum() > 0 for layer in ('layer1','layer2','layer4'))
    # The spatial raster, not a pooled camera token: shifting a feature impulse
    # through camera calibration shifts the BEV response in world coordinates.
    points = encoder.world_grid()
    torch.testing.assert_close(points[0,0],torch.tensor([-.5175,-.4175,0.]),atol=1e-6,rtol=0)
    torch.testing.assert_close(points[-1,-1],torch.tensor([.5175,.4175,0.]),atol=1e-6,rtol=0)


def test_world_raster_uses_valid_wrist_and_preserves_spatial_coordinates():
    v = api()
    packet, calibration = frame(translation=10.)
    _, centered = frame()
    calibration.cameras['wrist'] = centered.cameras['wrist']
    spatial = {view:{layer:t.clone() for layer,t in layers.items()} for view,layers in packet.spatial.items()}
    for layer,t in spatial['wrist'].items():
        side = t.shape[-1]
        # Actual ResNet stem7/2/pad3 + pool3/2/pad1 and odd-kernel
        # residual blocks: receptive centers are 0,stride,..., NOT resized
        # raster cell centers. Layer1/2/4 strides are4/8/32.
        t[:,0] = (torch.arange(side)*320/side)[None,:]
    packet = replace(packet,spatial=spatial)
    encoder = v.BEVEncoder(spec=v.BEVSpec(resolution=.04))
    with torch.no_grad():
        for projection in encoder.projections.values():
            projection.weight.zero_()
            projection.weight[0,0] = 1.
    output = encoder(packet,calibration)
    assert output.view_valid[:,4].all() and not output.view_valid[:,:4].any()
    expected = encoder.world_grid()[...,0]*100+159.5
    torch.testing.assert_close(output.features[0,0],expected,atol=4e-5,rtol=1e-6)
    updated = replace(calibration,cameras=calibration.cameras.copy())
    wrist = updated.cameras['wrist']
    matrix = wrist.camera_to_world.copy()
    matrix[0,3] = .2
    updated.cameras['wrist'] = replace(wrist,camera_to_world=matrix)
    moved = encoder(packet,updated)
    torch.testing.assert_close(moved.features[0,0],expected-20,atol=4e-5,rtol=1e-6)
    torch.testing.assert_close(encoder(packet,calibration).features,output.features,atol=0,rtol=0)


def test_appearance_is_local_to_predicted_object_not_global_fused_tokens():
    v = api()
    spec = v.BEVSpec(resolution=.04)
    detector = v.SixQueryDetector(spec)
    # Fix predicted center/shape so only image-local appearance may respond.
    with torch.no_grad():
        detector.head.weight.zero_()
        detector.head.bias.zero_()
    h,w = spec.shape
    base = torch.zeros(1,128,h,w)
    valid = torch.ones(1,1,h,w,dtype=torch.bool)
    def result(image):
        return detector(v.BEVOutput(image,valid,valid.expand(-1,5,-1,-1),0.,spec)).appearance
    original = result(base)
    local = base.clone()
    local[:,0,h//2-2:h//2+3,w//2-2:w//2+3] = 1.
    changed = result(local)
    assert not torch.allclose(original,changed)
    far = base.clone()
    far[:,:,0,0] = 100.
    torch.testing.assert_close(original,result(far),rtol=0,atol=0)


def targets(order=None):
    v = api()
    centers = torch.tensor([[-.25,-.1],[-.15,.1],[-.05,-.1],[.05,.1],[.15,-.1],[.25,.1]])
    idx = torch.arange(6) if order is None else torch.tensor(order)
    return v.DetectionTargets(center_xy=centers[idx],half_extents_xy=torch.full((6,2),.02)[idx],
                              yaw=torch.tensor([0.,.1,.2,.3,.4,.5])[idx],
                              shape_labels=torch.tensor([0,1,0,1,0,1])[idx],
                              visibility=torch.tensor([1.,1.,0.,1.,1.,1.])[idx],
                              state_labels=torch.tensor([0,0,1,2,0,0])[idx])


def test_six_query_attention_hungarian_loss_is_unordered_and_differentiable():
    v = api()
    torch.manual_seed(31)
    encoder = v.BEVEncoder(spec=v.BEVSpec(resolution=.04))
    packet, calibration = frame()
    raster = encoder(packet,calibration)
    detector = v.SixQueryDetector(spec=encoder.spec)
    prediction = detector(raster)
    assert prediction.center_xy.shape == (1,6,2) and prediction.appearance.shape == (1,6,16)
    assert len(detector.layers) == 2 and detector.layers[0].cross_attention.num_heads == 4
    losses = v.perception_loss(prediction,[targets()])
    permuted = v.perception_loss(prediction,[targets([5,2,4,0,3,1])])
    torch.testing.assert_close(losses['total'],permuted['total'],rtol=1e-6,atol=1e-6)
    assert all(torch.isfinite(value) for value in losses.values())
    losses['total'].backward()
    assert detector.queries.grad.abs().sum() > 0
    assert detector.layers[0].cross_attention.in_proj_weight.grad.abs().sum() > 0
    assert encoder.projections['layer2'].weight.grad.abs().sum() > 0
    # Circles do not have rectangle orientation supervision. Rectangles are pi periodic.
    original = targets()
    changed = replace(original,yaw=original.yaw+torch.tensor([9.,np.pi,8.,np.pi,7.,np.pi]))
    torch.testing.assert_close(v.perception_loss(prediction,[original])['yaw'],v.perception_loss(prediction,[changed])['yaw'],rtol=1e-5,atol=1e-6)
    assert torch.linalg.eigvalsh(prediction.position_covariance).min() > 0


def test_fully_unseen_bev_produces_no_nan_or_confident_detection():
    v = api()
    packet, calibration = frame(translation=10.)
    encoder = v.BEVEncoder(spec=v.BEVSpec(resolution=.04))
    prediction = v.SixQueryDetector(spec=encoder.spec)(encoder(packet,calibration))
    assert torch.isfinite(prediction.center_xy).all()
    assert torch.sigmoid(prediction.visibility_logits).max() < .5
    assert torch.linalg.eigvalsh(prediction.position_covariance).min().sqrt() > .025


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='MPS unavailable')
def test_fp32_mps_geometry_detector_loss_backward():
    v = api()
    packet, calibration = frame()
    packet = replace(packet,spatial={view:{k:t.to('mps') for k,t in layers.items()} for view,layers in packet.spatial.items()})
    encoder = v.BEVEncoder(spec=v.BEVSpec(resolution=.04)).to('mps')
    detector = v.SixQueryDetector(spec=encoder.spec).to('mps')
    output = detector(encoder(packet,calibration))
    loss = v.perception_loss(output,[targets()])['total']
    loss.backward()
    assert loss.device.type == 'mps' and loss.dtype == torch.float32
    assert detector.queries.grad.abs().sum().item() > 0
