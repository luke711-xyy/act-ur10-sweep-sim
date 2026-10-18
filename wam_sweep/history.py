"""Immediate, masked two-second history on the 5 Hz observation clock."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import torch

from .contracts import HistoryBatch, OnlineObservation, PHASES


class HistoryBuffer:
    """Right-aligned time slots; no waiting, resampling, or startup replication.

    Track IDs only associate object slots, including across missing detections.
    Six physical tracks are admitted per episode; reset by creating a new buffer.
    Observations may skip whole 200 ms slots, but cannot move off the 5 Hz grid.
    """

    def __init__(self) -> None:
        self._observations: list[OnlineObservation] = []
        self._track_ids: list[int] = []
        self._origin: float | None = None

    def append(self, observation: OnlineObservation) -> None:
        if type(observation) is not OnlineObservation:
            raise TypeError("HistoryBuffer accepts only OnlineObservation")
        # Re-run nested validation before changing any state, then own all buffers.
        frame = replace(observation)
        origin = frame.timestamp if self._origin is None else self._origin
        tick = (frame.timestamp - origin) * 5
        if abs(tick - round(tick)) > 1e-6:
            raise ValueError("observation must fall on the 5 Hz clock")
        if self._observations and frame.timestamp <= self._observations[-1].timestamp:
            raise ValueError("observation timestamps must strictly increase")
        ids = self._track_ids + sorted({item.track_id for item in frame.objects} - set(self._track_ids))
        if len(ids) > 6:
            raise ValueError("more than six episode tracks; tracker identity is invalid")
        retained = [old for old in self._observations if frame.timestamp - old.timestamp <= 2. + 1e-7]
        self._observations = retained + [frame]
        self._track_ids = ids
        self._origin = origin

    def batch(self) -> HistoryBatch:
        objects = torch.zeros((1, 11, 6, 34), dtype=torch.float32)
        brush = torch.zeros((1, 11, 9), dtype=torch.float32)
        context = torch.zeros((1, 11, 15), dtype=torch.float32)
        object_valid = torch.zeros((1, 11, 6), dtype=torch.float32)
        history_valid = torch.zeros((1, 11), dtype=torch.float32)
        if self._observations:
            now = self._observations[-1].timestamp
            for frame in self._observations:
                index = 10 - round((now - frame.timestamp) * 5)
                history_valid[0, index] = 1
                for item in frame.objects:
                    if not item.valid:
                        continue
                    slot = self._track_ids.index(item.track_id)
                    cov = item.position_covariance
                    features = np.concatenate((
                        item.center_xy, item.velocity_xy, item.half_extents_xy,
                        [np.sin(2 * item.yaw), np.cos(2 * item.yaw), item.angular_velocity],
                        item.shape_probabilities, [item.visibility], item.state_probabilities,
                        [cov[0, 0], cov[0, 1], cov[1, 1]], item.appearance,
                    ))
                    objects[0, index, slot] = torch.tensor(features, dtype=torch.float32)
                    object_valid[0, index, slot] = 1
                robot = frame.robot
                brush[0, index] = torch.tensor(np.concatenate((
                    robot.tcp_xyz, [np.sin(robot.yaw), np.cos(robot.yaw)],
                    robot.linear_velocity, [robot.yaw_velocity],
                )), dtype=torch.float32)
                context[0, index] = torch.tensor(np.concatenate((
                    [float(robot.phase == phase) for phase in PHASES],
                    robot.residual, robot.residual_rate,
                    [robot.episode_time, robot.contact_ready, robot.reference_valid,
                     robot.visual_entry_age, robot.visual_entry_valid],
                )), dtype=torch.float32)
        return HistoryBatch(objects, brush, context, object_valid, history_valid)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(dict(version=1, observations=self._observations,
                             track_ids=self._track_ids, origin=self._origin))

    def restore(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or set(state) != {"version", "observations", "track_ids", "origin"}:
            raise ValueError("invalid history snapshot fields")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("unsupported history snapshot version")
        observations, ids, origin = deepcopy((state["observations"], state["track_ids"], state["origin"]))
        if (not isinstance(ids, list) or len(ids) > 6
                or any(type(item) is not int or item < 0 for item in ids) or len(set(ids)) != len(ids)):
            raise ValueError("invalid snapshot track IDs")
        if not isinstance(observations, list) or len(observations) > 11:
            raise ValueError("invalid snapshot observations")
        if origin is None:
            if observations or ids:
                raise ValueError("nonempty history requires a clock origin")
        elif (type(origin) not in (int, float) or not np.isfinite(origin) or origin < 0):
            raise ValueError("invalid snapshot clock origin")
        elif not observations:
            raise ValueError("clock origin without history")
        candidate = HistoryBuffer()
        candidate._origin = origin
        candidate._track_ids = ids
        for frame in observations:
            if type(frame) is not OnlineObservation:
                raise TypeError("snapshot contains non-online observations")
            if frame.timestamp < origin or any(item.track_id not in ids for item in frame.objects):
                raise ValueError("snapshot clock or track IDs disagree")
            candidate.append(frame)
        if len(candidate._observations) != len(observations):
            raise ValueError("snapshot contains observations outside the two-second window")
        self._observations = candidate._observations
        self._track_ids = candidate._track_ids
        self._origin = candidate._origin
