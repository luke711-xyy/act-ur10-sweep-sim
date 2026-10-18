"""Causal visual measurements, calibrated CV Kalman/Hungarian beliefs.

IDs exist ONLY here for association/history slots. Neural inputs never include
IDs, native state, wrench or simulator truth. This module does no training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
import hashlib
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from ..contracts import InputIdentity, ObjectBelief
from ..inputs import digest_json
from .bev import BEVEncoder
from .detector import SixQueryDetector
from .features import ActFeaturePacket


@dataclass(frozen=True)
class VisualDetection:
    center_xy: np.ndarray
    half_extents_xy: np.ndarray
    yaw: float
    angular_velocity: float
    shape_probabilities: np.ndarray
    visibility: float
    state_probabilities: np.ndarray
    position_covariance: np.ndarray
    appearance: np.ndarray

    def __post_init__(self):
        checked = ObjectBelief(self.center_xy, np.zeros(2), self.half_extents_xy, self.yaw,
                               self.angular_velocity, self.shape_probabilities, self.visibility,
                               self.state_probabilities, self.position_covariance, self.appearance, 0, False)
        if np.linalg.eigvalsh(checked.position_covariance).min() <= 0:
            raise ValueError('measurement covariance must be positive definite')
        for name in self.__dataclass_fields__:
            object.__setattr__(self,name,getattr(checked,name))


@dataclass(frozen=True)
class PerceptionCalibration:
    """Engineering identity until fitted on validation, then frozen in checkpoint.

    scale multiplies covariance (not standard deviation); temperature applies
    to visibility logits. These defaults assert no empirical calibration result.
    """
    covariance_scale: float = 1.
    visibility_temperature: float = 1.
    provenance: str = 'engineering-unfitted'

    def __post_init__(self):
        if (not math.isfinite(self.covariance_scale) or self.covariance_scale <= 0
                or not math.isfinite(self.visibility_temperature) or self.visibility_temperature <= 0
                or self.provenance not in ('engineering-unfitted','validation-frozen')):
            raise ValueError('invalid uncertainty calibration')

    @classmethod
    def from_validation(cls, covariance_scale, visibility_temperature, *, split):
        if split != 'validation':
            raise ValueError('calibration can be adjusted only on validation')
        return cls(covariance_scale,visibility_temperature,'validation-frozen')

    def apply(self, detection):
        p = detection.visibility
        if 0 < p < 1:
            logit = math.log(p/(1-p))/self.visibility_temperature
            p = .5*(1+math.tanh(logit/2))
        return replace(detection,visibility=p,position_covariance=detection.position_covariance*self.covariance_scale)


@dataclass(frozen=True)
class TrackerSettings:
    gate_squared: float = 9.21
    confirmation_observations: int = 3
    minimum_visibility: float = .5
    initial_max_std: float = .012
    invalid_max_std: float = .025
    max_unobserved_seconds: float = .6
    acceleration_std: float = .25
    appearance_weight: float = 2.

    def __post_init__(self):
        if (any(not math.isfinite(v) or v <= 0 for v in asdict(self).values())
                or type(self.confirmation_observations) is not int
                or self.initial_max_std > self.invalid_max_std or self.minimum_visibility > 1):
            raise ValueError('invalid tracker engineering settings')


def _max_std(covariance):
    return float(np.sqrt(max(0.,np.linalg.eigvalsh(covariance).max())))


class CausalTracker:
    """Six bounded episode identities; no oracle filling or future smoothing.

    Tentative tracks require three consecutive visible, <=12mm observations.
    Short misses propagate the belief; >600ms or >25mm invalidates it. Tracks
    retain IDs after invalidation for possible visual reacquisition, requiring
    confirmation again. No seventh identity is silently introduced to history.
    """
    def __init__(self, *, settings=None, calibration=None):
        self.settings = settings or TrackerSettings()
        self.calibration = calibration or PerceptionCalibration()
        self._tracks = []
        self._timestamp = None
        self._next_id = 0

    def _predict(self, track, timestamp):
        dt = timestamp-track['timestamp']
        f = np.eye(4)
        f[:2,2:] = np.eye(2)*dt
        g = np.vstack((np.eye(2)*(.5*dt*dt),np.eye(2)*dt))
        track['mean'] = f @ track['mean']
        track['covariance'] = f @ track['covariance'] @ f.T + self.settings.acceleration_std**2*(g @ g.T)
        track['yaw'] = .5*math.atan2(math.sin(2*(track['yaw']+dt*track['angular_velocity'])),
                                    math.cos(2*(track['yaw']+dt*track['angular_velocity'])))
        track['timestamp'] = timestamp
        if (timestamp-track['last_seen'] > self.settings.max_unobserved_seconds+1e-10
                or _max_std(track['covariance'][:2,:2]) > self.settings.invalid_max_std):
            track['confirmed'] = False
            track['hits'] = 0

    def _new(self, detection, timestamp, track_id):
        covariance = np.zeros((4,4))
        covariance[:2,:2] = detection.position_covariance
        covariance[2:,2:] = np.eye(2)*.1**2
        return dict(track_id=track_id,mean=np.r_[detection.center_xy,np.zeros(2)],covariance=covariance,
                    timestamp=timestamp,last_seen=timestamp,hits=1,confirmed=self.settings.confirmation_observations <= 1,
                    half_extents_xy=detection.half_extents_xy.copy(),yaw=detection.yaw,angular_velocity=0.,
                    shape_probabilities=detection.shape_probabilities.copy(),visibility=detection.visibility,
                    state_probabilities=detection.state_probabilities.copy(),appearance=detection.appearance.copy())

    def _correct(self, track, detection, timestamp):
        h = np.eye(2,4)
        p, r = track['covariance'], detection.position_covariance
        gain = np.linalg.solve(h @ p @ h.T+r,h @ p).T
        track['mean'] = track['mean']+gain @ (detection.center_xy-h @ track['mean'])
        residual = np.eye(4)-gain @ h
        # Joseph form preserves covariance PSD through short occlusion/replay.
        track['covariance'] = residual @ p @ residual.T+gain @ r @ gain.T
        track['covariance'] = .5*(track['covariance']+track['covariance'].T)
        dt = timestamp-track['last_seen']
        innovation = .5*math.atan2(math.sin(2*(detection.yaw-track['yaw'])),math.cos(2*(detection.yaw-track['yaw'])))
        track['angular_velocity'] = .5*track['angular_velocity']+.5*(innovation/max(dt,1e-6)+track['angular_velocity'])
        track['yaw'] += .7*innovation
        if detection.shape_probabilities[0] >= detection.shape_probabilities[1]:
            track['yaw'] = track['angular_velocity'] = 0.
        for name in ('half_extents_xy','shape_probabilities','appearance'):
            track[name] = .7*getattr(detection,name)+.3*track[name]
        norm = np.linalg.norm(track['appearance'])
        if norm > 1e-12:
            track['appearance'] /= norm
        track['visibility'] = detection.visibility
        track['state_probabilities'] = detection.state_probabilities.copy()
        track['last_seen'] = timestamp
        if _max_std(detection.position_covariance) <= self.settings.initial_max_std+1e-12:
            track['hits'] += 1
        elif not track['confirmed']:
            track['hits'] = 0
        if track['hits'] >= self.settings.confirmation_observations:
            track['confirmed'] = True

    def update(self, timestamp: float, detections) -> tuple[ObjectBelief,...]:
        if (not isinstance(timestamp,(int,float,np.number)) or not np.isfinite(timestamp) or timestamp < 0
                or (self._timestamp is not None and timestamp <= self._timestamp)):
            raise ValueError('tracker timestamps must strictly increase')
        if len(detections) > 6 or any(type(d) is not VisualDetection for d in detections):
            raise ValueError('at most six identity-free visual detections required')
        observations = [self.calibration.apply(replace(d)) for d in detections]
        observations = [d for d in observations if d.visibility >= self.settings.minimum_visibility
                        and _max_std(d.position_covariance) <= self.settings.invalid_max_std+1e-12]
        tracks = copy.deepcopy(self._tracks)
        for track in tracks:
            self._predict(track,timestamp)
        cost = np.full((len(tracks),len(observations)),1e9)
        for i,track in enumerate(tracks):
            for j,d in enumerate(observations):
                delta = d.center_xy-track['mean'][:2]
                s = track['covariance'][:2,:2]+d.position_covariance
                maha = float(delta @ np.linalg.solve(s,delta))
                if maha <= self.settings.gate_squared:
                    norm = np.linalg.norm(track['appearance'])*np.linalg.norm(d.appearance)
                    appearance = 1-float(track['appearance'] @ d.appearance)/norm if norm > 1e-12 else 1.
                    cost[i,j] = maha+self.settings.appearance_weight*max(0.,appearance)
        rows,cols = linear_sum_assignment(cost)
        used, observed = set(),set()
        for row,col in zip(rows,cols):
            if cost[row,col] >= 1e9:
                continue
            self._correct(tracks[row],observations[col],timestamp)
            used.add(col)
            observed.add(row)
        for i,track in enumerate(tracks):
            if i not in observed:
                if not track['confirmed']:
                    track['hits'] = 0
                dt = timestamp-(self._timestamp if self._timestamp is not None else timestamp)
                track['visibility'] *= math.exp(-dt/self.settings.max_unobserved_seconds)
                track['state_probabilities'] = np.full(3,1/3)
        next_id = self._next_id
        for j,detection in enumerate(observations):
            if j not in used and next_id < 6 and _max_std(detection.position_covariance) <= self.settings.initial_max_std+1e-12:
                tracks.append(self._new(detection,timestamp,next_id))
                next_id += 1
        beliefs = []
        for track in tracks:
            valid = (track['confirmed'] and timestamp-track['last_seen'] <= self.settings.max_unobserved_seconds+1e-10
                     and _max_std(track['covariance'][:2,:2]) <= self.settings.invalid_max_std)
            beliefs.append(ObjectBelief(track['mean'][:2],track['mean'][2:],track['half_extents_xy'],track['yaw'],
                                         track['angular_velocity'],track['shape_probabilities'],track['visibility'],
                                         track['state_probabilities'],track['covariance'][:2,:2],track['appearance'],track['track_id'],bool(valid)))
        self._tracks,self._timestamp,self._next_id = tracks,float(timestamp),next_id
        return tuple(beliefs)

    def snapshot(self):
        return copy.deepcopy(dict(schema=1,settings=asdict(self.settings),calibration=asdict(self.calibration),
                                  timestamp=self._timestamp,next_id=self._next_id,tracks=self._tracks))

    def restore(self, state):
        if (set(state) != {'schema','settings','calibration','timestamp','next_id','tracks'} or state['schema'] != 1
                or state['settings'] != asdict(self.settings) or state['calibration'] != asdict(self.calibration)):
            raise ValueError('tracker snapshot identity/settings mismatch')
        tracks = copy.deepcopy(state['tracks'])
        n, timestamp = state['next_id'],state['timestamp']
        if (type(n) is not int or not 0 <= n <= 6 or not isinstance(tracks,list) or len(tracks) != n
                or [t['track_id'] for t in tracks] != list(range(n))
                or (timestamp is None and tracks) or (timestamp is not None and (not np.isfinite(timestamp) or timestamp < 0))):
            raise ValueError('invalid tracker snapshot clock/identities')
        for track in tracks:
            p,m = track['covariance'],track['mean']
            if (not isinstance(p,np.ndarray) or p.shape != (4,4) or not np.isfinite(p).all()
                    or not np.allclose(p,p.T,atol=1e-12) or np.linalg.eigvalsh(p).min() <= 0):
                raise ValueError('invalid tracker covariance')
            if (np.asarray(m).shape != (4,) or not np.isfinite(m).all() or track['timestamp'] != timestamp
                    or not 0 <= track['last_seen'] <= timestamp or type(track['hits']) is not int or track['hits'] < 0
                    or type(track['confirmed']) is not bool):
                raise ValueError('invalid tracker causal state')
            VisualDetection(m[:2],track['half_extents_xy'],track['yaw'],track['angular_velocity'],
                            track['shape_probabilities'],track['visibility'],track['state_probabilities'],p[:2,:2],track['appearance'])
        self._tracks,self._timestamp,self._next_id = tracks,timestamp,n


def _weight_hash(states):
    digest = hashlib.sha256()
    for group,state in sorted(states.items()):
        for name,value in sorted(state.items()):
            digest.update((group+'/'+name+str(tuple(value.shape))+str(value.dtype)).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class VisualStateEstimator:
    """External two-phase consumer: estimate(packet) via exact-time metadata.

    Register as a named beliefs=False provider before preparation. Metadata
    provider must return the saved calibration for packet.timestamp, including
    historical wrist extrinsics. estimate never queries ACT or captures RGB.
    Snapshots save causal tracker/model identity only. checkpoint() separately
    supplies trainable weights AND complete dependency/calibration identities.
    """
    def __init__(self, bev: BEVEncoder, detector: SixQueryDetector, input_identity: InputIdentity,
                 capture_metadata, *, tracker=None):
        if bev.spec != detector.spec or type(input_identity) is not InputIdentity or not callable(capture_metadata):
            raise ValueError('estimator needs matching geometry, identity and capture-time provider')
        self.bev,self.detector = bev.eval(),detector.eval()
        self.input_identity = input_identity
        self.capture_metadata = capture_metadata
        self.tracker = tracker or CausalTracker()
        self._timestamp = None
        self._model_hash = _weight_hash(self._states())

    def _states(self):
        return {'bev':self.bev.state_dict(),'detector':self.detector.state_dict()}

    def _manifest(self):
        return dict(schema=1,input_identity=asdict(self.input_identity),bev_spec=asdict(self.bev.spec),
                    tracker_settings=asdict(self.tracker.settings),calibration=asdict(self.tracker.calibration),
                    model_hash=self._model_hash)

    def estimate(self, packet: ActFeaturePacket) -> tuple[ObjectBelief,...]:
        if (type(packet) is not ActFeaturePacket or packet.input_hash != self.input_identity.manifest_hash
                or packet.model_hash != self.input_identity.model_hash
                or packet.preprocessing_hash != self.input_identity.native_hashes['preprocessing']):
            raise ValueError('estimator feature/input dependency identity mismatch')
        if self._timestamp is not None and packet.timestamp <= self._timestamp:
            raise ValueError('estimator timestamps must strictly increase')
        calibration = self.capture_metadata(packet.timestamp)
        with torch.no_grad():
            output = self.detector(self.bev(packet,calibration))
        if output.center_xy.shape[0] != 1:
            raise ValueError('online causal estimator requires one frame, not an offline batch')
        arrays = {name:getattr(output,name).detach().cpu().numpy()[0] for name in
                  ('center_xy','half_extents_xy','yaw_sincos','angular_velocity','position_covariance','appearance')}
        shapes = output.shape_logits.softmax(-1).cpu().numpy()[0]
        states = output.state_logits.softmax(-1).cpu().numpy()[0]
        visible = output.visibility_logits.sigmoid().cpu().numpy()[0,:,0]
        detections = [VisualDetection(arrays['center_xy'][i],arrays['half_extents_xy'][i],
                                      .5*math.atan2(*arrays['yaw_sincos'][i]),float(arrays['angular_velocity'][i,0]),
                                      shapes[i],float(visible[i]),states[i],arrays['position_covariance'][i],arrays['appearance'][i])
                      for i in range(6)]
        result = self.tracker.update(packet.timestamp,detections)
        self._timestamp = packet.timestamp
        return result

    def snapshot(self):
        if _weight_hash(self._states()) != self._model_hash:
            raise ValueError('estimator model changed; bind a new version after training')
        return dict(schema=1,identity=digest_json(self._manifest()),timestamp=self._timestamp,tracker=self.tracker.snapshot())

    def restore(self, state):
        if (set(state) != {'schema','identity','timestamp','tracker'} or state['schema'] != 1
                or state['identity'] != digest_json(self._manifest()) or _weight_hash(self._states()) != self._model_hash):
            raise ValueError('estimator snapshot model/dependency identity mismatch')
        if state['timestamp'] != state['tracker']['timestamp']:
            raise ValueError('estimator/tracker snapshot timestamps disagree')
        self.tracker.restore(state['tracker'])
        self._timestamp = state['timestamp']

    def checkpoint(self):
        """No file writes: future trainer/store owns budgeted atomic persistence."""
        if _weight_hash(self._states()) != self._model_hash:
            raise ValueError('model changed; construct estimator after training')
        return dict(manifest=self._manifest(),weights={group:{k:v.detach().cpu().clone() for k,v in state.items()}
                                                     for group,state in self._states().items()})

    def load_checkpoint(self, checkpoint):
        manifest,weights = checkpoint['manifest'],checkpoint['weights']
        expected = self._manifest()
        if (set(checkpoint) != {'manifest','weights'} or set(manifest) != set(expected)
                or any(manifest[k] != v for k,v in expected.items() if k != 'model_hash')
                or _weight_hash(weights) != manifest['model_hash']):
            raise ValueError('checkpoint missing/mismatched dependency/calibration/model identity')
        if self._timestamp is not None:
            raise ValueError('load perception checkpoint before causal observations')
        # Validate full state on isolated module copies before touching live model.
        for name,module in (('bev',self.bev),('detector',self.detector)):
            copy.deepcopy(module).load_state_dict(weights[name],strict=True)
        self.bev.load_state_dict(weights['bev'],strict=True)
        self.detector.load_state_dict(weights['detector'],strict=True)
        self._model_hash = manifest['model_hash']
