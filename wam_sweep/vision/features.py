"""Borrowed frozen spatial features; no native imports or second encoder.

The backend owns ACT, preprocessing, query scheduling and postprocessing. Hooks
only observe the three spatial layers, once per view in the saved ACT order.
Calibration travels separately from this safe neural packet.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import copy
from typing import Mapping
import weakref

import numpy as np
import torch

from ..artifacts import RunLease
from ..contracts import CAMERA_VIEWS, OnlineObservation, ReferencePreview
from ..inputs import digest_json

LAYERS = {'layer1': (64, 80), 'layer2': (128, 40), 'layer4': (512, 10)}
IMAGE_KEYS = tuple('observation.images.' + view for view in CAMERA_VIEWS)
_OWNERS = weakref.WeakKeyDictionary()


def validate_capture_metadata(metadata, timestamp, input_hash):
    """Separate capture channel, including exact historical wrist matrices."""
    from ..native import CameraCalibration, CaptureMetadata
    if type(metadata) is not CaptureMetadata or metadata.schema != 1 or metadata.timestamp != timestamp:
        raise ValueError('calibration schema/timestamp does not match historical pixels')
    if metadata.input_hash != input_hash:
        raise ValueError('capture input identity mismatch')
    if tuple(metadata.cameras) != CAMERA_VIEWS or any(type(c) is not CameraCalibration for c in metadata.cameras.values()):
        raise ValueError('calibration camera order/types differ from feature order')


@dataclass(frozen=True)
class ActFeaturePacket:
    """Only FP32 image-space pyramids and already-known native reference data.

    No native state, target N, transformer tokens, calibration, or environment
    handle. Caller may retain one frame in memory; snapshots retain RGB instead.
    """
    timestamp: float
    spatial: Mapping[str, Mapping[str, torch.Tensor]]
    input_hash: str
    model_hash: str
    preprocessing_hash: str
    reference: ReferencePreview | None = None
    receipt: object | None = None

    def __post_init__(self):
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError('invalid feature timestamp')
        if not all(isinstance(v, str) and v for v in (self.input_hash, self.model_hash, self.preprocessing_hash)):
            raise ValueError('missing feature dependency identity')
        if tuple(self.spatial) != CAMERA_VIEWS:
            raise ValueError('feature camera order must match frozen ACT')
        devices, batches = set(), set()
        for layers in self.spatial.values():
            if tuple(layers) != tuple(LAYERS):
                raise ValueError('exactly layer1/layer2/layer4 spatial features required')
            for name, (channels, side) in LAYERS.items():
                value = layers[name]
                if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.ndim != 4
                        or value.shape[0] < 1 or tuple(value.shape[1:]) != (channels, side, side)
                        or value.requires_grad or not torch.isfinite(value).all()):
                    raise ValueError('invalid frozen spatial tensor ' + name)
                devices.add(value.device)
                batches.add(value.shape[0])
        if len(devices) != 1 or len(batches) != 1:
            raise ValueError('spatial features must share batch/device')
        if self.reference is not None and type(self.reference) is not ReferencePreview:
            raise TypeError('feature reference must be a safe ReferencePreview')
        if self.receipt is not None:
            # Importing the source-guarded facade does not import sim/PAC. Exact
            # type excludes arbitrary environment/model/native36 payloads.
            from ..native import QueryReceipt
            if type(self.receipt) is not QueryReceipt:
                raise TypeError('feature receipt must be the safe native QueryReceipt')
            values = (self.receipt.issued_at,self.receipt.received_at,self.receipt.valid_from,self.receipt.inference_wall_seconds)
            if (not all(isinstance(x,(int,float)) and np.isfinite(x) and x >= 0 for x in values)
                    or self.receipt.issued_at != self.timestamp or type(self.receipt.accepted) is not bool):
                raise ValueError('invalid feature receipt timestamp/values')


class FrozenFeatureTap:
    """Named restorable facade around backend.bind_frozen_policy().

    Active path: query_act_and_features(raw), OR backend.query_act(raw) followed
    by packet_after_query(raw, receipt). Preparation/fixed branches call ONLY
    features_only(raw). Never call from a beliefs=True callback inside observe.
    All calls require the caller's admitted activity/run lease; external RGB
    passes acquire only the distinct native inference lock.
    """
    def __init__(self, backend):
        self._backend = backend
        self.binding = backend.bind_frozen_policy()
        self.identity = backend.input_identity
        self._identity = digest_json(asdict(self.identity))
        policy = self.binding.policy
        if tuple(k.removeprefix('observation.images.') for k in policy.config.image_features) != CAMERA_VIEWS:
            raise ValueError('frozen camera order mismatch')
        if policy.training or any(p.requires_grad for p in policy.parameters()):
            raise ValueError('shared policy must already be frozen/eval')
        if policy in _OWNERS and _OWNERS[policy]() is not None:
            raise ValueError('one feature tap per shared policy')
        _OWNERS[policy] = weakref.ref(self)
        self._closed = False
        self._spatial = {view: {} for view in CAMERA_VIEWS}
        self._counts = dict.fromkeys(LAYERS, 0)
        self._mode = None
        self._last = None
        self._observation = None
        self._metadata = None
        self._reference = self._receipt = None
        self._handles = [policy.model.register_forward_pre_hook(self._act_start)]
        self._handles += [policy.model.backbone[name].register_forward_hook(self._hook(name)) for name in LAYERS]

    def _clear(self, mode=None):
        self._spatial = {view: {} for view in CAMERA_VIEWS}
        self._counts = dict.fromkeys(LAYERS, 0)
        self._mode = mode

    def _act_start(self, module, args):
        self._check_live()
        self._clear('act')

    def _hook(self, layer):
        def capture(module, args, output):
            # Passive hooks on the SAME backbone, never a second forward. A
            # bounded five-view window is replaced at every ACT/RGB start.
            if self._mode is None:
                return
            index = self._counts[layer]
            if index >= len(CAMERA_VIEWS):
                raise RuntimeError('too many frozen backbone camera passes')
            self._spatial[CAMERA_VIEWS[index]][layer] = output.detach()
            self._counts[layer] += 1
        return capture

    def _check_live(self):
        if self._closed:
            raise RuntimeError('feature tap is closed')
        if self.binding.policy.training or any(p.requires_grad for p in self.binding.policy.parameters()):
            raise RuntimeError('shared frozen policy was modified')

    def _validate(self, observation, *, replay=False):
        self._check_live()
        if type(observation) is not OnlineObservation:
            raise TypeError('feature tap accepts only safe OnlineObservation')
        if tuple(observation.images) != CAMERA_VIEWS:
            raise ValueError('image camera order mismatch')
        if any(im.dtype != np.uint8 or im.shape != (320, 320, 3) for im in observation.images.values()):
            raise ValueError('frozen RGB must be uint8 320x320')
        if not replay and self._last is not None and observation.timestamp <= self._last:
            raise ValueError('feature timestamps must increase; duplicate encoding forbidden')

    def _validate_capture_pixels(self, observation):
        # Authenticate live-entry RGB only. Never pass private native state to
        # preprocessing/the network, or require current pixels during replay.
        private = self._backend.native_act_input(observation.timestamp)
        for key, view in zip(IMAGE_KEYS, CAMERA_VIEWS):
            rgb = observation.images[view].transpose(2, 0, 1).astype(np.float32) / 255.
            if not np.array_equal(rgb, private[key]):
                raise ValueError('feature pixels differ from capture')

    def preprocess_images(self, observation):
        """Exact saved processor visual slice, filtering transition defaults."""
        self._check_live()
        image_dict = {key: torch.from_numpy(observation.images[view].transpose(2, 0, 1).copy()).float() / 255.
                      for key, view in zip(IMAGE_KEYS, CAMERA_VIEWS)}
        processed = self.binding.preprocessor(image_dict)
        return {key: processed[key] for key in IMAGE_KEYS}

    def _packet(self, observation, *, reference=None, receipt=None, metadata=None):
        if any(n != 5 for n in self._counts.values()):
            raise RuntimeError('incomplete same-forward camera pyramid; cannot re-encode to repair')
        spatial = {view: {layer: self._spatial[view][layer] for layer in LAYERS} for view in CAMERA_VIEWS}
        packet = ActFeaturePacket(observation.timestamp, spatial, self.identity.manifest_hash,
                                  self.identity.model_hash, self.identity.native_hashes['preprocessing'], reference, receipt)
        if metadata is None:
            metadata = self._backend.capture_metadata(observation.timestamp)
        validate_capture_metadata(metadata,observation.timestamp,self.identity.manifest_hash)
        # Latest raw RGB only; no pyramid in a snapshot or on disk. Backend
        # persistent snapshots externalize this OnlineObservation's images.
        self._observation = replace(observation, objects=())
        self._metadata = copy.deepcopy(metadata)
        self._last = observation.timestamp
        self._reference, self._receipt = copy.deepcopy((reference, receipt))
        self._clear()
        return packet

    def query_act_and_features(self, observation) -> ActFeaturePacket:
        self._validate(observation)
        receipt = self._backend.query_act(observation)
        return self.packet_after_query(observation, receipt)

    def packet_after_query(self, observation, receipt) -> ActFeaturePacket:
        """Finish an external backend.query_act(raw); receipt has no chunk."""
        self._validate(observation)
        if self._mode != 'act' or receipt.issued_at != observation.timestamp:
            raise ValueError('features must accompany this exact native query timestamp')
        # This also rejects a foreign/stale raw packet for an external query.
        self._validate_capture_pixels(observation)
        return self._packet(observation, reference=self._backend.reference_preview(), receipt=receipt)

    def features_only(self, observation) -> ActFeaturePacket:
        self._validate(observation)
        # Exact capture metadata is a separate producer; fail before inference.
        metadata = self._backend.capture_metadata(observation.timestamp)
        validate_capture_metadata(metadata,observation.timestamp,self.identity.manifest_hash)
        self._validate_capture_pixels(observation)
        return self._images_only(observation,metadata=metadata)

    def capture_metadata(self, timestamp):
        """Return the one retained RGB frame's separately stored calibration.

        Bind estimator to this provider for replay after backend.observe has
        replaced its own current capture. No recomputation from current FK.
        """
        if self._metadata is None or timestamp != self._metadata.timestamp:
            raise ValueError('timestamp has no retained feature capture calibration')
        return copy.deepcopy(self._metadata)

    def _images_only(self, observation, *, reference=None, receipt=None, metadata=None):
        self._clear('images')
        try:
            with RunLease(self._backend.native_lock), torch.no_grad():
                batch = self.preprocess_images(observation)
                for key in IMAGE_KEYS:
                    self.binding.policy.model.backbone(batch[key])
            return self._packet(observation, reference=reference, receipt=receipt,metadata=metadata)
        except BaseException:
            self._clear()
            raise

    def replay_features(self) -> ActFeaturePacket:
        """Explicit charged regeneration from retained historical RGB; no ACT."""
        if self._observation is None:
            raise ValueError('no retained RGB for replay')
        self._validate(self._observation, replay=True)
        return self._images_only(self._observation, reference=self._reference, receipt=self._receipt,metadata=self._metadata)

    def snapshot(self):
        if self._mode is not None:
            raise RuntimeError('finish feature packet before snapshot')
        return copy.deepcopy(dict(schema=1, identity=self._identity, timestamp=self._last,
                                  observation=self._observation, metadata=self._metadata,reference=self._reference, receipt=self._receipt))

    def restore(self, state):
        if (set(state) != {'schema', 'identity', 'timestamp', 'observation', 'metadata','reference', 'receipt'}
                or state['schema'] != 1 or state['identity'] != self._identity):
            raise ValueError('feature snapshot identity/schema mismatch')
        observation = state['observation']
        if observation is not None:
            self._validate(observation, replay=True)
            if observation.timestamp != state['timestamp']:
                raise ValueError('feature snapshot timestamp mismatch')
            validate_capture_metadata(state['metadata'],observation.timestamp,self.identity.manifest_hash)
        elif state['timestamp'] is not None or state['metadata'] is not None:
            raise ValueError('feature timestamp requires retained RGB')
        self._last, self._observation, self._metadata,self._reference, self._receipt = copy.deepcopy(
            (state['timestamp'], observation, state['metadata'],state['reference'], state['receipt']))
        self._clear()

    def close(self):
        """Detach borrowed hooks when the facade/episode owner is disposed."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._clear()
        self._closed = True
        if _OWNERS.get(self.binding.policy, lambda: None)() is self:
            del _OWNERS[self.binding.policy]
