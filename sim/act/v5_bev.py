"""Reconstruct the dense BEV inside the policy boundary from v5 sidecars."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..perception.object_bev import BEVSpec, PredictedInstance, build_bev_channels
from ..perception.object_tokens import OBJECT_TOKEN_SLICES


def build_bev_from_sidecar(
    *,
    object_tokens: np.ndarray,
    object_valid: np.ndarray,
    instance_bev: np.ndarray,
    robot_state: np.ndarray,
    selection_target: np.ndarray | None = None,
    tray_bounds: Sequence[float],
    forbidden_rectangles: Iterable[Sequence[float]] = (),
    brush_size: Sequence[float] = (0.12, 0.01),
    spec: BEVSpec | None = None,
) -> np.ndarray:
    """Build the six policy channels without using MuJoCo geometry state.

    The v5 sidecar stores predicted object tokens and packed instance masks,
    not a dense map.  The current brush centre/yaw comes from the 36-D robot
    observation (TCP x/y and sin/cos yaw at indices 6:11).
    """

    spec = spec or BEVSpec()
    tokens = np.asarray(object_tokens, dtype=np.float32)
    valid = np.asarray(object_valid, dtype=bool)
    masks = np.asarray(instance_bev, dtype=bool)
    state = np.asarray(robot_state, dtype=np.float32)
    if tokens.shape != (6, 29) or valid.shape != (6,) or masks.shape != (6, 128, 160):
        raise ValueError("v5 sidecar frame has incompatible object shapes")
    if state.shape != (36,):
        raise ValueError("robot_state must have shape (36,)")
    if selection_target is not None:
        selection = np.asarray(selection_target, dtype=bool)
        if selection.shape != (6,):
            raise ValueError("selection_target must have shape (6,)")
    else:
        selection = np.zeros(6, dtype=bool)
    instances = []
    for slot in range(6):
        if not valid[slot]:
            continue
        class_probs = tokens[slot, OBJECT_TOKEN_SLICES["class_probs"]]
        if float(np.sum(class_probs)) <= 1e-12:
            # Empty sidecars are useful for contract tests and are treated as
            # an uncertain round object, never as simulator truth.
            class_probs = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        instances.append(
            PredictedInstance(
                track_id=slot,
                mask_bev=masks[slot],
                class_probs=class_probs,
                confidence=float(tokens[slot, OBJECT_TOKEN_SLICES["confidence"]][0]),
                xy=tokens[slot, OBJECT_TOKEN_SLICES["xy"]],
                extent=np.maximum(tokens[slot, OBJECT_TOKEN_SLICES["extent"]], spec.resolution),
                yaw=0.0,
                tray_overlap=float(tokens[slot, OBJECT_TOKEN_SLICES["tray_overlap"]][0]),
                full_inside=bool(tokens[slot, OBJECT_TOKEN_SLICES["full_inside"]][0] > 0.5),
                overhead_visibility=float(tokens[slot, OBJECT_TOKEN_SLICES["overhead_visibility"]][0]),
                wrist_visibility=float(tokens[slot, OBJECT_TOKEN_SLICES["wrist_visibility"]][0]),
            )
        )
    brush_xy = state[6:8]
    brush_yaw = float(np.arctan2(state[9], state[10]))
    tray_mouth_xy = np.array(
        [(float(tray_bounds[0]) + float(tray_bounds[1])) / 2.0,
         (float(tray_bounds[2]) + float(tray_bounds[3])) / 2.0],
        dtype=float,
    )
    del tray_mouth_xy  # retained as an explicit geometry boundary in the caller contract
    return build_bev_channels(
        instances,
        spec=spec,
        selected_track_ids={slot for slot in range(6) if selection[slot]},
        brush_xy=brush_xy,
        brush_size=np.asarray(brush_size, dtype=float),
        brush_yaw=brush_yaw,
        tray_bounds=tray_bounds,
        forbidden_rectangles=forbidden_rectangles,
    )
