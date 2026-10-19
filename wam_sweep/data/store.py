"""Owned, content-addressed evidence. Pickles are trusted LOCAL records only.

Callers own the canonical run lease and ONE linked active budget clock around
serialization, replay and all model/physics work. No callable/renderer is saved.
Reconstruct source-checked native runtime and matching named providers BEFORE
loading native snapshots. File hashes detect corruption, not hostile pickles.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import pickle
import re
import time
import zlib

import numpy as np

from ..artifacts import BudgetExceeded, atomic_checkpoint, safe_path
from ..contracts import OnlineObservation
from ..native import CaptureMetadata
from ..snapshots import EpisodeSnapshot, externalize_images
from ..vision.features import validate_capture_metadata

SCHEMA = 'wam-episode-store-v1'
NAMESPACES = frozenset(('captures', 'snapshots', 'models', 'evidence', 'roots',
                        'trajectories', 'perception', 'failures'))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _key(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('content SHA256 required')
    return value


def _observations(value):
    if type(value) is OnlineObservation:
        yield value
    elif type(value) is EpisodeSnapshot:
        yield from _observations(value.state)
        yield from _observations(value.participants)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _observations(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _observations(item)


def _metadata(value):
    if type(value) is CaptureMetadata:
        yield value
    elif type(value) is EpisodeSnapshot:
        yield from _metadata(value.state)
        yield from _metadata(value.participants)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _metadata(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _metadata(item)


@dataclass(frozen=True)
class StoredCapture:
    timestamp: float
    objects: tuple
    robot: object
    images: dict[str, str]
    calibration: CaptureMetadata
    schema: str = SCHEMA
    supervision_id: str | None = None


@dataclass(frozen=True)
class SnapshotArchive:
    snapshot: EpisodeSnapshot
    capture_ids: tuple[str, ...]
    schema: str = SCHEMA


@dataclass(frozen=True)
class StoredBranch:
    metadata: object
    trace_id: str
    continuation_id: str | None


@dataclass(frozen=True)
class StoredRoot:
    metadata: object
    branches: tuple[StoredBranch, ...]
    schema: str = SCHEMA


class EpisodeStore:
    """Immutable blobs + atomic named cursors, failed/partial evidence retained.

    Raw no-HUD RGB is losslessly compressed once per pixel SHA, shared by main
    captures, histories, native nested roots, branches and continuation replay.
    Calibration belongs to each capture, never to an image hash alone (a static
    image may occur at different poses/times). No feature pyramids are persisted.
    """
    def __init__(self, directory, budget, *, max_record_bytes=512 * 2**20):
        self.directory = budget.ensure_owned(directory)
        self.budget = budget
        self.max_record_bytes = max_record_bytes
        self._active()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _active(self):
        state = json.loads(self.budget.path.read_bytes())
        active = state['active']
        if active is None or active.get('reservation') not in state['reservations']:
            raise RuntimeError('store work requires caller-owned linked budget activity')
        # Same bounded inventory cadence as native._admit; read the linked
        # ownership state on every call, and atomic_checkpoint checks free disk
        # before each actual write. Deduplicated reads need no tree rescan.
        now = time.monotonic()
        if now >= getattr(self,'_next_poll',0.):
            self.budget.poll().require()
            self._next_poll = now + 1.

    def _path(self, namespace, key):
        if namespace not in NAMESPACES and namespace != 'images':
            raise ValueError('unknown evidence namespace')
        return safe_path(self.directory / namespace / _key(key))

    def _publish(self, path, data):
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('immutable content collision/corruption')
        else:
            atomic_checkpoint(path, data, self.budget)

    def put(self, namespace, value):
        self._active()
        if namespace not in NAMESPACES:
            raise ValueError('unknown evidence namespace')
        with self.budget.reserve(2 * self.max_record_bytes):
            if namespace=='roots':
                from .collection import RootRecord, ExecutionTrace
                if type(value) is not RootRecord:
                    raise TypeError('RootRecord required')
                branches = []
                for branch in value.branches:
                    trace_id = self._put_value('evidence',branch.trace)
                    continuation_id = None if branch.continuation is None else self._put_value('evidence',branch.continuation)
                    branches.append(StoredBranch(replace(branch,trace=ExecutionTrace(),continuation=None),trace_id,continuation_id))
                value = StoredRoot(replace(value,branches=()),tuple(branches))
            return self._put_value(namespace,value)

    def _put_value(self, namespace, value):
        payload = pickle.dumps((SCHEMA,namespace,value),protocol=5)
        if len(payload)>self.max_record_bytes:
            raise BudgetExceeded(f'record {len(payload)} bytes exceeds admitted serialization bound')
        data = zlib.compress(payload,level=3)
        key = _digest(data)
        self._publish(self._path(namespace,key),data)
        return key

    def get(self, namespace, key):
        value = self._read_value(namespace,key)
        if namespace=='roots' and type(value) is StoredRoot:
            if value.schema!=SCHEMA:
                raise ValueError('root archive schema mismatch')
            branches = tuple(replace(b.metadata,trace=self.get('evidence',b.trace_id),
                continuation=None if b.continuation_id is None else self.get('evidence',b.continuation_id)) for b in value.branches)
            return replace(value.metadata,branches=branches)
        return value

    def _read_value(self, namespace, key):
        self._active()
        data = self._path(namespace,key).read_bytes()
        if _digest(data) != key:
            raise ValueError('record content hash mismatch')
        schema, actual_namespace, value = pickle.loads(zlib.decompress(data))
        if schema != SCHEMA or actual_namespace != namespace:
            raise ValueError('record schema/namespace mismatch')
        return value

    def root_archive(self, key):
        value = self._read_value('roots',key)
        if type(value) is not StoredRoot or value.schema!=SCHEMA:
            raise ValueError('root archive schema mismatch')
        return value

    def write_image(self, pixel_hash, image):
        self._active()
        if (not isinstance(image,np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3):
            raise ValueError('lossless uint8 HWC RGB required')
        raw = image.tobytes()
        if _digest(str(image.shape).encode('ascii') + raw) != pixel_hash:
            raise ValueError('pixel hash mismatch')
        header = json.dumps(list(image.shape)).encode() + b'\n'
        self._publish(self._path('images',pixel_hash), header + zlib.compress(raw,level=3))
        return pixel_hash

    def read_image(self, reference):
        self._active()
        header, compressed = self._path('images',reference).read_bytes().split(b'\n',1)
        shape = tuple(json.loads(header))
        image = np.frombuffer(zlib.decompress(compressed),dtype=np.uint8).reshape(shape).copy()
        if _digest(str(image.shape).encode('ascii') + image.tobytes()) != reference:
            raise ValueError('image content hash mismatch')
        return image

    def put_capture(self, observation, calibration, *, supervision=None):
        if type(observation) is not OnlineObservation:
            raise TypeError('safe OnlineObservation required')
        validate_capture_metadata(calibration,observation.timestamp,calibration.input_hash)
        images = {}
        for view,image in observation.images.items():
            digest = _digest(str(image.shape).encode('ascii') + image.tobytes())
            images[view] = self.write_image(digest,image)
        supervision_id = None if supervision is None else self.put('evidence',supervision)
        return self.put('captures',StoredCapture(observation.timestamp,observation.objects,
                                                observation.robot,images,calibration,
                                                supervision_id=supervision_id))

    def read_capture(self, key):
        capture = self.get('captures',key)
        if type(capture) is not StoredCapture or capture.schema != SCHEMA:
            raise ValueError('capture schema mismatch')
        return (OnlineObservation(capture.timestamp,capture.objects,capture.robot,
                                  {v:self.read_image(k) for v,k in capture.images.items()}),
                capture.calibration)

    def read_supervision(self, capture_id):
        """OFFLINE only; read_capture() deliberately returns no supervision."""
        capture = self.get('captures',capture_id)
        return None if capture.supervision_id is None else self.get('evidence',capture.supervision_id)

    def put_failed_capture(self, observation, calibration, error):
        """Forensics only: unavailable/invalid calibration is retained as-is.

        No failed frame is appended to a trainable causal capture prefix.
        Pixels still use verified lossless shared references, never a raw pickle.
        """
        images = {}
        for view,pixels in observation.images.items():
            key = _digest(str(pixels.shape).encode('ascii')+pixels.tobytes())
            images[view] = self.write_image(key,pixels)
        return self.put('failures',dict(stage='capture_contract',timestamp=observation.timestamp,
            images=images,calibration=calibration,robot=observation.robot,
            exception_type=type(error).__name__,reason=str(error)))

    def save_snapshot(self, snapshot, *, captures):
        if type(snapshot) is not EpisodeSnapshot or snapshot.images_externalized:
            raise ValueError('complete unexternalized EpisodeSnapshot required')
        metadata = dict(captures)
        # Current/nested provider packets carry calibration too. Older history
        # must have been retained by collection at its original capture time.
        for item in _metadata(snapshot):
            if item.timestamp in metadata:
                if pickle.dumps(item) != pickle.dumps(metadata[item.timestamp]):
                    raise ValueError('conflicting historical capture calibration')
            metadata[item.timestamp] = item
        capture_ids = set()
        for observation in _observations(snapshot):
            if observation.timestamp not in metadata:
                raise ValueError('historical RGB missing capture-time calibration')
            capture_ids.add(self.put_capture(observation,metadata[observation.timestamp]))
        external = externalize_images(snapshot,self.write_image)
        return self.put('snapshots',SnapshotArchive(external,tuple(sorted(capture_ids))))

    def load_snapshot(self, key):
        archive = self.get('snapshots',key)
        if type(archive) is not SnapshotArchive or archive.schema != SCHEMA:
            raise ValueError('snapshot archive schema mismatch')
        return archive.snapshot

    def snapshot_captures(self, key):
        archive = self.get('snapshots',key)
        return tuple(self.read_capture(capture) for capture in archive.capture_ids)

    def publish_cursor(self, name, value):
        self._active()
        if not isinstance(name,str) or re.fullmatch('[a-zA-Z0-9_-]+',name) is None:
            raise ValueError('simple cursor name required')
        data = json.dumps(dict(schema=SCHEMA,value=value),sort_keys=True,allow_nan=False).encode()
        atomic_checkpoint(self.directory/'cursors'/(name+'.json'),data,self.budget)

    def cursor(self, name):
        if not isinstance(name,str) or re.fullmatch('[a-zA-Z0-9_-]+',name) is None:
            raise ValueError('simple cursor name required')
        record = json.loads(safe_path(self.directory/'cursors'/(name+'.json')).read_bytes())
        if record['schema'] != SCHEMA:
            raise ValueError('cursor schema mismatch')
        return record['value']
