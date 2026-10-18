"""Private physical snapshots; never an online policy input.

MuJoCo integration state includes time, warmstart, actuator, applied forces,
mocap, userdata and plugin state. A copy of MjData also retains the *accepted*
derived sensor/kinematic caches: mj_step's caches need not equal mj_forward at
the newly integrated qpos. Re-forwarding on restore can change the next force
command. Renderer handles are excluded. Persistent stores own image references.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import random
from typing import Any

import mujoco
import numpy as np
import torch

from .contracts import OnlineObservation


@dataclass(frozen=True)
class PhysicsSnapshot:
    model_signature: tuple
    integration: np.ndarray
    data: Any
    environment: dict
    end_effector: dict
    random_state: tuple


@dataclass(frozen=True)
class EpisodeSnapshot:
    identity: str
    physics: PhysicsSnapshot
    state: dict
    participants: dict
    cutpoint: str
    schema: int = 1
    images_externalized: bool = False


@dataclass(frozen=True)
class ImageReference:
    reference: str
    pixel_sha256: str
    shape: tuple


@dataclass(frozen=True)
class _StoredObservation:
    timestamp: float
    objects: tuple
    robot: Any
    images: dict


def _pixel_hash(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('invalid RGB image')
    return hashlib.sha256(str(image.shape).encode('ascii') + image.tobytes()).hexdigest()


def _walk_images(value, transform):
    if isinstance(value, (OnlineObservation, _StoredObservation)):
        return transform(value)
    if isinstance(value, EpisodeSnapshot):
        return replace(value, physics=copy.deepcopy(value.physics),
                       state=_walk_images(value.state, transform),
                       participants=_walk_images(value.participants, transform))
    if isinstance(value, dict):
        return {k: _walk_images(v, transform) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk_images(v, transform) for v in value]
    if isinstance(value, tuple):
        return tuple(_walk_images(v, transform) for v in value)
    return copy.deepcopy(value)


def externalize_images(snapshot: EpisodeSnapshot, put_image) -> EpisodeSnapshot:
    """Task8 store adapter: put_image(pixel_hash, uint8_HWC)->durable reference.

    Deduplicates across capture, complete provider histories and replay roots.
    Cross-root deduplication/disk admission belong to the content-addressed
    caller-owned store. Pixel hashes are distinct from encoded-file hashes.
    No renderer, file handle or image payload enters the resulting history.
    """
    memo = {}
    def external(observation):
        if isinstance(observation, _StoredObservation):
            return copy.deepcopy(observation)
        refs = {}
        for view, image in observation.images.items():
            digest = _pixel_hash(image)
            if digest not in memo:
                ref = put_image(digest, image.copy())
                if not isinstance(ref, str) or not ref:
                    raise ValueError('image writer must return a durable reference')
                memo[digest] = ImageReference(ref, digest, image.shape)
            refs[view] = memo[digest]
        return _StoredObservation(observation.timestamp, copy.deepcopy(observation.objects),
                                  copy.deepcopy(observation.robot), refs)
    return replace(_walk_images(snapshot, external), images_externalized=True)


def materialize_images(snapshot: EpisodeSnapshot, get_image) -> EpisodeSnapshot:
    """get_image(reference)->losslessly decoded RGB; hash/shape checked first."""
    if snapshot.images_externalized and get_image is None:
        raise ValueError('external snapshot requires an image reader')
    memo = {}
    def materialize(observation):
        if type(observation) is OnlineObservation:
            return copy.deepcopy(observation)
        images = {}
        for view, ref in observation.images.items():
            if ref not in memo:
                image = get_image(ref.reference)
                if _pixel_hash(image) != ref.pixel_sha256 or image.shape != ref.shape:
                    raise ValueError('image reference content/shape mismatch')
                memo[ref] = image.copy()
            images[view] = memo[ref]
        return OnlineObservation(observation.timestamp, observation.objects, observation.robot, images)
    return replace(_walk_images(snapshot, materialize), images_externalized=False)


_ENV_HANDLES = frozenset({'model', 'data', 'ee', 'cfg', 'episode_cfg',
                         '_renderer', '_seg_renderer', '_depth_renderer',
                         '_camera_renderers', 'step_control'})
_EE_HANDLES = frozenset({'model', 'data', 'cfg', '_mujoco'})


def _signature(model):
    return tuple(int(getattr(model, k)) for k in ('nq', 'nv', 'na', 'nu', 'nbody', 'ngeom', 'ncam'))


def _owned_attributes(obj, excluded):
    return copy.deepcopy({k: v for k, v in vars(obj).items() if k not in excluded})


def capture_physics(env) -> PhysicsSnapshot:
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    values = np.empty(mujoco.mj_stateSize(env.model, spec), dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, values, spec)
    return PhysicsSnapshot(_signature(env.model), values, copy.copy(env.data),
                           _owned_attributes(env, _ENV_HANDLES),
                           _owned_attributes(env.ee, _EE_HANDLES), capture_random_state())


def capture_random_state():
    return copy.deepcopy((random.getstate(), np.random.get_state(), torch.get_rng_state().clone(),
                          torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                          torch.mps.get_rng_state().clone() if torch.backends.mps.is_available() else None))


def restore_physics(env, snapshot: PhysicsSnapshot):
    if type(snapshot) is not PhysicsSnapshot or snapshot.model_signature != _signature(env.model):
        raise ValueError('incompatible physical snapshot')
    for obj, excluded, state in ((env, _ENV_HANDLES, snapshot.environment),
                                  (env.ee, _EE_HANDLES, snapshot.end_effector)):
        for key in set(vars(obj)) - excluded - set(state):
            delattr(obj, key)
        vars(obj).update(copy.deepcopy(state))
    # Preserve the live data object so end-effector and renderer references do
    # not point at an abandoned state. Copy all derived caches before setting
    # the explicitly versioned integration state.
    mujoco.mj_copyData(env.data, env.model, snapshot.data)
    mujoco.mj_setState(env.model, env.data, snapshot.integration,
                      mujoco.mjtState.mjSTATE_INTEGRATION)
    # Only cameras/lights are recomputed, never the force/filter pipeline.
    mujoco.mj_camlight(env.model, env.data)
    restore_random_state(snapshot.random_state)


def restore_random_state(state):
    py, np_state, cpu, cuda, mps = copy.deepcopy(state)
    random.setstate(py)
    np.random.set_state(np_state)
    torch.set_rng_state(cpu)
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    if mps is not None:
        torch.mps.set_rng_state(mps)
