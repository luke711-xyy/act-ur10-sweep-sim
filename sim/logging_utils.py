"""Run directories, result serialisation and small console helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def config_hash(cfg_dict: Dict[str, Any]) -> str:
    payload = json.dumps(cfg_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def make_run_dir(base: str, tag: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(base, f"{tag}-{stamp}")
    os.makedirs(path, exist_ok=True)
    return path


def _jsonify(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def save_json(obj: Any, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonify(obj), handle, indent=2)
    return path


def save_csv(rows: List[dict], path: str, fieldnames: Optional[Iterable[str]] = None) -> str:
    if not rows:
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    names = list(fieldnames) if fieldnames else list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _jsonify(row.get(k)) for k in names})
    return path


def load_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value in ("", None):
                row[key] = None
                continue
            if key in ("planner", "perception", "geometry", "failure_reason", "config_hash"):
                continue
            low = str(value).strip().lower()
            if low in ("true", "false"):
                row[key] = low == "true"
                continue
            try:
                row[key] = float(value) if ("." in value or "e" in low or "nan" in low) else int(value)
            except ValueError:
                pass
    return rows


def trace_to_arrays(trace) -> Dict[str, np.ndarray]:
    """Dense per-control-step arrays, used by the dataset exporter and plots."""
    if not trace:
        return {}
    return {
        "t": np.array([r.t for r in trace], dtype=np.float32),
        "stroke_index": np.array([r.stroke for r in trace], dtype=np.int16),
        "phase": np.array([r.phase for r in trace], dtype="U18"),
        "tcp_position": np.array([[r.tcp_x, r.tcp_y, r.tcp_z] for r in trace], dtype=np.float32),
        "command": np.array([[r.cmd_x, r.cmd_y, r.cmd_z, r.cmd_yaw] for r in trace],
                            dtype=np.float32),
        "z_nominal": np.array([r.z_nominal for r in trace], dtype=np.float32),
        "delta_z": np.array([r.delta_z for r in trace], dtype=np.float32),
        "force_desired": np.array([r.force_desired for r in trace], dtype=np.float32),
        "force_raw": np.array([r.force_raw for r in trace], dtype=np.float32),
        "force_filtered": np.array([r.force_filtered for r in trace], dtype=np.float32),
        "in_contact": np.array([r.in_contact for r in trace], dtype=bool),
        "tcp_yaw": np.array([r.tcp_yaw for r in trace], dtype=np.float32),
        "tcp_velocity": np.array([[r.tcp_vx, r.tcp_vy, r.tcp_vz] for r in trace],
                                 dtype=np.float32),
        # Full external wrench on the tool, world frame: [fx, fy, fz, tx, ty, tz].
        "wrench": np.array([[r.fx, r.fy, r.fz, r.tx, r.ty, r.tz] for r in trace],
                           dtype=np.float32),
        "tangential_force": np.array([r.tangential_force for r in trace], dtype=np.float32),
        # Simulator ground truth: the component-contact share of the wrench.
        # Not available on hardware -- for validation and labelling only.
        "wrench_parts": np.array([[r.fx_p, r.fy_p, r.fz_p, r.tx_p, r.ty_p, r.tz_p]
                                  for r in trace], dtype=np.float32),
        "tangential_force_parts": np.array([r.tangential_force_parts for r in trace],
                                           dtype=np.float32),
        "n_part_contacts": np.array([r.n_part_contacts for r in trace], dtype=np.int16),
    }


def save_episode_bundle(result, out_dir: str, prefix: str = "episode") -> Dict[str, str]:
    """Save metrics, phase events, layout, config and the dense control trace."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    paths["metrics"] = save_json(result.metrics.to_dict(),
                                 os.path.join(out_dir, f"{prefix}_metrics.json"))
    paths["events"] = save_json(result.events, os.path.join(out_dir, f"{prefix}_events.json"))
    paths["layout"] = save_json(result.layout, os.path.join(out_dir, f"{prefix}_layout.json"))
    paths["config"] = save_json(result.config, os.path.join(out_dir, f"{prefix}_config.json"))
    arrays = trace_to_arrays(result.trace)
    if arrays:
        trace_path = os.path.join(out_dir, f"{prefix}_trace.npz")
        np.savez_compressed(
            trace_path,
            actions=np.array([s.action for s in result.strokes], dtype=np.float32)
            if result.strokes else np.zeros((0, 5), dtype=np.float32),
            initial_positions=np.asarray(result.initial_positions, dtype=np.float32),
            final_positions=np.asarray(result.final_positions, dtype=np.float32),
            **arrays,
        )
        paths["trace"] = trace_path
    return paths


def print_metrics(metrics) -> None:
    m = metrics.to_dict() if hasattr(metrics, "to_dict") else dict(metrics)
    order = [
        ("planner", "planner", "{}"), ("perception", "perception", "{}"),
        ("seed", "seed", "{}"), ("n_components", "components", "{}"),
        ("collection_rate", "collection rate", "{:.2%}"),
        ("success", "success", "{}"),
        ("completion_time", "completion time", "{:.1f} s"),
        ("n_strokes", "strokes", "{}"),
        ("components_per_stroke", "components / stroke", "{:.2f}"),
        ("path_length", "TCP path length", "{:.2f} m"),
        ("n_pushed_out", "pushed out of workspace", "{}"),
        ("peak_normal_force", "peak normal force", "{:.2f} N"),
        ("rms_force_error", "RMS force error", "{:.3f} N"),
        ("contact_loss_ratio", "contact-loss ratio", "{:.3f}"),
        ("wall_time", "wall time", "{:.1f} s"),
        ("failure_reason", "failure reason", "{}"),
    ]
    print("-" * 58)
    for key, label, fmt in order:
        value = m.get(key)
        if value is None or (isinstance(value, str) and not value):
            continue
        try:
            rendered = fmt.format(value)
        except (ValueError, TypeError):
            rendered = str(value)
        print(f"  {label:<26s} {rendered}")
    print("-" * 58)
