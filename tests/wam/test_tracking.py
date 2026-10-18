"""CV Kalman/Hungarian tracking without identities or simulator truth inputs."""
import copy
import importlib
import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('wam_sweep.vision') is not None, 'Task6 vision package missing'
    return importlib.import_module('wam_sweep.vision')


def measurement(x=0.,y=0.,appearance=0,visibility=1.,std=.004):
    v = api()
    local = np.zeros(16)
    local[appearance] = 1.
    return v.VisualDetection(np.array([x,y]),np.array([.02,.015]),0.,0.,np.array([0.,1.]),
                             visibility,np.array([1.,0.,0.]),np.eye(2)*std**2,local)


def test_short_occlusion_keeps_identity():
    v = api()
    tracker = v.CausalTracker()
    assert not tracker.update(5.,[measurement()])[0].valid
    assert not tracker.update(5.2,[measurement(.002)])[0].valid
    before = tracker.update(5.4,[measurement(.004)])[0]
    assert before.valid
    missing = tracker.update(5.6,[])[0]
    assert missing.valid and missing.visibility < before.visibility
    assert missing.position_covariance.trace() > before.position_covariance.trace()
    after = tracker.update(5.8,[measurement(.008)])[0]
    assert after.track_id == before.track_id and after.valid
    timeout = tracker.update(6.40001,[])[0]
    assert not timeout.valid
    assert timeout.position_covariance.trace() > after.position_covariance.trace()


def test_crossing_reversed_query_order_preserves_appearance_identity_and_velocity():
    v = api()
    tracker = v.CausalTracker()
    for i in range(8):
        a = measurement(-.035+.01*i,appearance=0)
        b = measurement(.035-.01*i,appearance=1)
        values = [b,a] if i%2 else [a,b]
        objects = tracker.update(10.+i*.2,values)
        if i == 0:
            ids = {int(b.appearance.argmax()):b.track_id for b in objects}
    current = {int(b.appearance.argmax()):b for b in objects}
    assert all(current[a].track_id == ids[a] and current[a].valid for a in (0,1))
    assert current[0].velocity_xy[0] > .03 and current[1].velocity_xy[0] < -.03
    np.testing.assert_allclose(current[0].center_xy,[.035,0],atol=.005)


def test_unknown_visibility_uncertainty_and_gated_outlier_cannot_confirm():
    v = api()
    tracker = v.CausalTracker()
    for i in range(3):
        assert tracker.update(i*.2,[measurement(visibility=.49)]) == ()
    for i in range(3,6):
        assert tracker.update(i*.2,[measurement(std=.013)]) == ()
    for i in range(6,9):
        result = tracker.update(i*.2,[measurement(std=.012)])
    assert result[0].valid
    original_id = result[0].track_id
    result = tracker.update(1.8,[measurement(.4)])
    original = next(b for b in result if b.track_id == original_id)
    assert original.center_xy[0] < .01  # Mahalanobis gate rejects the jump
    assert not next(b for b in result if b.track_id != original_id).valid
    uncertain = v.CausalTracker()
    for i in range(3): uncertain.update(i*.2,[measurement()])
    result = uncertain.update(.9,[])
    assert not result[0].valid  # covariance can invalidate before the .6s timeout


def test_tracker_snapshot_exact_replay_and_identity_configuration_rejected():
    v = api()
    tracker = v.CausalTracker()
    for i in range(4): tracker.update(3.+i*.2,[measurement(i*.003)])
    state = tracker.snapshot()
    expected = tracker.update(3.8,[measurement(.012)])
    tracker.restore(copy.deepcopy(state))
    actual = tracker.update(3.8,[measurement(.012)])
    assert actual[0].track_id == expected[0].track_id
    np.testing.assert_array_equal(actual[0].center_xy,expected[0].center_xy)
    np.testing.assert_array_equal(actual[0].position_covariance,expected[0].position_covariance)
    with pytest.raises(ValueError,match='increas'):
        tracker.update(3.8,[])
    bad = copy.deepcopy(state)
    bad['settings']['gate_squared'] = 999.
    with pytest.raises(ValueError,match='identity|settings'):
        tracker.restore(bad)
    bad = copy.deepcopy(state)
    bad['tracks'][0]['covariance'][0,0] = -1.
    with pytest.raises(ValueError,match='covariance'):
        tracker.restore(bad)


def test_calibration_scaling_thresholds_and_checkpoint_roundtrip():
    v = api()
    with pytest.raises(ValueError,match='validation'):
        v.PerceptionCalibration.from_validation(2.,1.,split='test')
    calibration = v.PerceptionCalibration.from_validation(4.,1.,split='validation')
    tracker = v.CausalTracker(calibration=calibration)
    for i in range(3): assert tracker.update(i*.2,[measurement(std=.007)]) == ()
    assert tracker.snapshot()['calibration']['covariance_scale'] == 4.
