"""Episode metrics.

Everything required by the specification is computed here from (a) the
controller trace and (b) the environment's ground-truth component state, so the
same numbers are produced no matter which planner or perception backend ran.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    # identification / reproducibility
    seed: int = 0
    planner: str = ""
    perception: str = ""
    n_components: int = 0
    geometry: str = ""
    config_hash: str = ""

    # required metrics
    collection_rate: float = 0.0
    n_collected: int = 0
    success: bool = False
    completion_time: float = 0.0          # simulated seconds
    n_strokes: int = 0
    path_length: float = 0.0              # total 3-D TCP path [m]
    path_length_xy: float = 0.0
    n_pushed_out: int = 0                 # components outside the safe workspace
    peak_normal_force: float = 0.0
    rms_force_error: float = float("nan") # RMS of (F_desired - F_measured) while sweeping
    contact_loss_ratio: float = float("nan")
    components_per_stroke: float = 0.0

    # failure modes (see README "Failure modes")
    n_ride_over: int = 0                  # swept over but not moved (thin-washer mode)
    n_jam_events: int = 0                 # sustained high tangential force (jam mode)
    touchdown_overshoot_max: float = 0.0  # peak force above target at touchdown [N]
    touchdown_overshoot_mean: float = 0.0

    # diagnostics
    failure_reason: str = ""
    wall_time: float = 0.0
    aborts: int = 0
    mean_contact_force: float = float("nan")
    strokes_with_contact: int = 0
    # Wrench diagnostics (simulator ground truth -- see README "Force signal quality")
    part_contact_ratio: float = float("nan")   # share of SWEEP samples touching a part
    part_force_snr: float = float("nan")       # |F_parts,xy| / |F_total,xy| while touching
    mean_tangential_force: float = float("nan")
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["success"] = bool(self.success)
        return out


def _trace_arrays(trace) -> Dict[str, np.ndarray]:
    if not trace:
        empty = np.zeros(0)
        return {k: empty for k in
                ("t", "x", "y", "z", "fd", "ff", "ft", "ft_parts", "n_part",
                 "contact", "phase")}
    return {
        "t": np.array([r.t for r in trace], dtype=float),
        "x": np.array([r.tcp_x for r in trace], dtype=float),
        "y": np.array([r.tcp_y for r in trace], dtype=float),
        "z": np.array([r.tcp_z for r in trace], dtype=float),
        "fd": np.array([r.force_desired for r in trace], dtype=float),
        "ff": np.array([r.force_filtered for r in trace], dtype=float),
        "ft": np.array([r.tangential_force for r in trace], dtype=float),
        "ft_parts": np.array([r.tangential_force_parts for r in trace], dtype=float),
        "n_part": np.array([r.n_part_contacts for r in trace], dtype=int),
        "contact": np.array([r.in_contact for r in trace], dtype=bool),
        "phase": np.array([r.phase for r in trace], dtype=object),
    }


def compute_episode_metrics(
    *,
    trace,
    n_components: int,
    n_collected: int,
    n_pushed_out: int,
    n_strokes: int,
    sim_time: float,
    success: bool,
    seed: int,
    planner: str,
    perception: str,
    geometry: str,
    failure_reason: str = "",
    wall_time: float = 0.0,
    aborts: int = 0,
    strokes_with_contact: int = 0,
    n_ride_over: int = 0,
    n_jam_events: int = 0,
    touchdown_overshoot_max: float = 0.0,
    touchdown_overshoot_mean: float = 0.0,
    extra: Optional[dict] = None,
) -> EpisodeMetrics:
    a = _trace_arrays(trace)

    if a["x"].size >= 2:
        d3 = np.stack([np.diff(a["x"]), np.diff(a["y"]), np.diff(a["z"])], axis=1)
        path_length = float(np.sum(np.linalg.norm(d3, axis=1)))
        path_length_xy = float(np.sum(np.linalg.norm(d3[:, :2], axis=1)))
    else:
        path_length = path_length_xy = 0.0

    sweeping = a["phase"] == "SWEEP" if a["phase"].size else np.zeros(0, dtype=bool)
    if sweeping.any():
        err = a["fd"][sweeping] - a["ff"][sweeping]
        rms_force_error = float(np.sqrt(np.mean(err ** 2)))
        contact_loss_ratio = float(1.0 - a["contact"][sweeping].mean())
        mean_contact_force = float(a["ff"][sweeping].mean())
    else:
        rms_force_error = float("nan")
        contact_loss_ratio = float("nan")
        mean_contact_force = float("nan")

    if sweeping.any():
        part_contacts = a["n_part"][sweeping]
        touching = part_contacts > 0
        part_contact_ratio = float(touching.mean())
        mean_tangential_force = float(a["ft"][sweeping].mean())
        if touching.any():
            total_xy = a["ft"][sweeping][touching]
            part_xy = a["ft_parts"][sweeping][touching]
            denom = float(np.mean(total_xy))
            part_force_snr = float(np.mean(part_xy) / denom) if denom > 1e-9 else float("nan")
        else:
            part_force_snr = float("nan")
    else:
        part_contact_ratio = float("nan")
        part_force_snr = float("nan")
        mean_tangential_force = float("nan")

    peak = float(a["ff"].max()) if a["ff"].size else 0.0
    rate = float(n_collected) / n_components if n_components else 0.0

    return EpisodeMetrics(
        seed=int(seed),
        planner=planner,
        perception=perception,
        n_components=int(n_components),
        geometry=geometry,
        collection_rate=rate,
        n_collected=int(n_collected),
        success=bool(success),
        completion_time=float(sim_time),
        n_strokes=int(n_strokes),
        path_length=path_length,
        path_length_xy=path_length_xy,
        n_pushed_out=int(n_pushed_out),
        peak_normal_force=peak,
        rms_force_error=rms_force_error,
        contact_loss_ratio=contact_loss_ratio,
        components_per_stroke=(float(n_collected) / n_strokes) if n_strokes else 0.0,
        failure_reason=failure_reason,
        wall_time=float(wall_time),
        aborts=int(aborts),
        mean_contact_force=mean_contact_force,
        strokes_with_contact=int(strokes_with_contact),
        part_contact_ratio=part_contact_ratio,
        part_force_snr=part_force_snr,
        mean_tangential_force=mean_tangential_force,
        n_ride_over=int(n_ride_over),
        n_jam_events=int(n_jam_events),
        touchdown_overshoot_max=float(touchdown_overshoot_max),
        touchdown_overshoot_mean=float(touchdown_overshoot_mean),
        extra=dict(extra or {}),
    )


def count_jam_events(trace, cfg) -> int:
    """Number of jam events in a stroke.

    A jam is a sustained **tangential** force while sweeping: the pusher is
    loaded in-plane far beyond what tool-table friction alone explains, which is
    what happens when a part wedges against the tip instead of sliding.  The
    threshold must sit above the baseline ``mu * F_normal`` drag, so it is
    configured rather than derived (``metrics.jam_force_threshold``).

    Force, not stall, is the signal: the prototype's position actuators are
    strong enough that the TCP keeps tracking its command through a jam, so a
    velocity-based test would miss it.  On a real position-controlled arm the
    same threshold also precedes a protective stop.
    """
    threshold = float(cfg.get_path("metrics.jam_force_threshold", 4.0))
    min_duration = float(cfg.get_path("metrics.jam_min_duration", 0.08))
    dt = 1.0 / float(cfg.sim.control_hz)
    min_samples = max(1, int(round(min_duration / dt)))

    events, run = 0, 0
    for record in trace:
        sweeping = record.phase in ("SWEEP", "FORCE_RAMP")
        if sweeping and record.tangential_force > threshold:
            run += 1
            if run == min_samples:
                events += 1
        else:
            run = 0
    return events


def touchdown_overshoot(trace, desired_force: float, window: float = 0.5) -> float:
    """Peak normal force above the target during force build-up, in newtons.

    Measured over FORCE_RAMP plus the first ``window`` seconds of SWEEP -- the
    transient the admittance gains are actually judged on.  Returns 0 when the
    force never exceeds the target.
    """
    ramp = [r for r in trace if r.phase == "FORCE_RAMP"]
    sweep = [r for r in trace if r.phase == "SWEEP"]
    if sweep:
        t0 = sweep[0].t
        sweep = [r for r in sweep if r.t - t0 <= window]
    samples = ramp + sweep
    if not samples:
        return 0.0
    return max(0.0, max(r.force_filtered for r in samples) - float(desired_force))


def aggregate(records: List[dict], group_keys=("planner", "n_components")) -> List[dict]:
    """Mean / std / n of the numeric metrics, grouped by ``group_keys``."""
    numeric = [
        "collection_rate", "completion_time", "n_strokes", "path_length",
        "n_pushed_out", "peak_normal_force", "rms_force_error",
        "contact_loss_ratio", "components_per_stroke",
        "n_ride_over", "n_jam_events", "touchdown_overshoot_max",
        "touchdown_overshoot_mean", "part_contact_ratio", "part_force_snr",
        "mean_tangential_force",
    ]
    groups: Dict[tuple, List[dict]] = {}
    for rec in records:
        key = tuple(rec.get(k) for k in group_keys)
        groups.setdefault(key, []).append(rec)

    out = []
    for key, rows in sorted(groups.items(), key=lambda kv: [str(v) for v in kv[0]]):
        entry = {k: v for k, v in zip(group_keys, key)}
        entry["episodes"] = len(rows)
        entry["success_rate"] = float(np.mean([bool(r["success"]) for r in rows]))
        for name in numeric:
            values = np.array([r.get(name, np.nan) for r in rows], dtype=float)
            values = values[np.isfinite(values)]
            entry[f"{name}_mean"] = float(values.mean()) if values.size else float("nan")
            entry[f"{name}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        out.append(entry)
    return out


def paired_comparison(records: List[dict], planner_a: str, planner_b: str,
                      metric: str = "collection_rate") -> List[dict]:
    """Compare two planners on identical ``(seed, n_components)`` layouts."""
    index: Dict[tuple, Dict[str, dict]] = {}
    for rec in records:
        key = (rec["n_components"], rec["seed"])
        index.setdefault(key, {})[rec["planner"]] = rec

    rows = []
    for (n, seed), by_planner in sorted(index.items()):
        if planner_a in by_planner and planner_b in by_planner:
            va = float(by_planner[planner_a].get(metric, np.nan))
            vb = float(by_planner[planner_b].get(metric, np.nan))
            rows.append({"n_components": n, "seed": seed, metric + "_a": va,
                         metric + "_b": vb, "delta": vb - va})
    return rows
