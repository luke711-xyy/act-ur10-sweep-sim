"""Figures for single episodes and for batch experiments.

Chart conventions used throughout
---------------------------------
* one measure per axis -- never a twin/dual y-axis; related measures get their
  own stacked panel instead
* a fixed categorical colour order (blue / orange / aqua), assigned by *entity*
  (planner name), never recycled per figure, so a planner keeps its colour
  across every plot in a report
* recessive grid and axes, thin marks, a legend whenever two or more series are
  drawn, and direct labelling where a legend would be ambiguous
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --- design tokens ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e2"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
PLANNER_COLOR: Dict[str, str] = {"fixed": SERIES[0], "visual_greedy": SERIES[1],
                                 "global_sweep": SERIES[2]}
PLANNER_LABEL: Dict[str, str] = {"fixed": "A: fixed full-cover",
                                 "global_sweep": "C: global single sweep",
                                 "visual_greedy": "B: visual greedy (segmented)"}
PHASE_BAND = {
    "APPROACH": "#f2f1ee",
    "SEARCH_CONTACT": "#e9eef6",
    "CONTACT_DETECTED": "#dfe9f7",
    "FORCE_RAMP": "#dbe8f8",
    "SWEEP": "#eaf4ee",
    "FORCE_RELEASE": "#fbe9e0",
    "RETRACT": "#f4f2ef",
    "OBSERVE_MOVE": "#fafaf8",
}


def _style_axes(ax, xlabel: str = "", ylabel: str = "", title: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def _new_fig(width: float, height: float):
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE, dpi=140)
    return fig


def _trace_arrays(trace):
    return {
        "t": np.array([r.t for r in trace], dtype=float),
        "x": np.array([r.tcp_x for r in trace], dtype=float),
        "y": np.array([r.tcp_y for r in trace], dtype=float),
        "z": np.array([r.tcp_z for r in trace], dtype=float),
        "cz": np.array([r.cmd_z for r in trace], dtype=float),
        "zn": np.array([r.z_nominal for r in trace], dtype=float),
        "dz": np.array([r.delta_z for r in trace], dtype=float),
        "fd": np.array([r.force_desired for r in trace], dtype=float),
        "fr": np.array([r.force_raw for r in trace], dtype=float),
        "ff": np.array([r.force_filtered for r in trace], dtype=float),
        "contact": np.array([r.in_contact for r in trace], dtype=bool),
        "phase": np.array([r.phase for r in trace], dtype=object),
        "stroke": np.array([r.stroke for r in trace], dtype=int),
    }


def _shade_phases(ax, t: np.ndarray, phase: np.ndarray) -> None:
    if t.size == 0:
        return
    start = 0
    for i in range(1, t.size + 1):
        if i == t.size or phase[i] != phase[start]:
            name = str(phase[start])
            color = PHASE_BAND.get(name)
            if color:
                ax.axvspan(t[start], t[min(i, t.size - 1)], color=color, lw=0, zorder=0)
            start = i


def _event_markers(ax, events: Iterable[dict], kinds=("CONTACT_DETECTED", "FORCE_RELEASE")):
    """Mark contact and release events as thin vertical rules.

    Rules rather than top-of-axes markers: they stay readable at any zoom and
    never collide with the legend.
    """
    from matplotlib.lines import Line2D

    style = {"CONTACT_DETECTED": (SERIES[2], "contact detected", (0, (1, 2))),
             "FORCE_RELEASE": (SERIES[1], "force release", (0, (3, 2)))}
    seen = set()
    handles = []
    for ev in events:
        if ev["phase"] not in kinds:
            continue
        color, label, dashes = style[ev["phase"]]
        ax.axvline(ev["time"], color=color, lw=1.0, ls=dashes, alpha=0.85, zorder=2)
        if label not in seen:
            handles.append(Line2D([], [], color=color, lw=1.0, ls=dashes, label=label))
            seen.add(label)
    return handles


# ---------------------------------------------------------------- episode
def plot_episode(result, out_dir: str, prefix: str = "episode") -> List[str]:
    """Force tracking, Z correction, XY trajectory and contact events."""
    os.makedirs(out_dir, exist_ok=True)
    a = _trace_arrays(result.trace)
    paths: List[str] = []
    if a["t"].size == 0:
        return paths

    # --- 1. desired vs measured normal force ---
    fig = _new_fig(9.0, 3.2)
    ax = fig.add_subplot(111)
    _shade_phases(ax, a["t"], a["phase"])
    ax.plot(a["t"], a["fr"], color=INK_MUTED, lw=0.8, alpha=0.55, label="measured (raw)")
    ax.plot(a["t"], a["ff"], color=SERIES[0], lw=1.6, label="measured (filtered)")
    ax.plot(a["t"], a["fd"], color=INK, lw=1.4, ls="--", label="desired")
    _style_axes(ax, "time [s]", "normal force [N]",
                f"Normal-force tracking  ({result.metrics.planner}, seed {result.metrics.seed}, "
                f"{result.metrics.n_components} components)")
    event_handles = _event_markers(ax, result.events)
    peak = float(max(a["ff"].max(), a["fd"].max(), 1e-6))
    ax.set_ylim(min(0.0, float(a["ff"].min()) * 1.1), peak * 1.45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + event_handles,
              labels + [h.get_label() for h in event_handles],
              frameon=False, fontsize=8, labelcolor=INK_2, ncols=5,
              loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_force.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    paths.append(path)

    # --- 2. Z position correction ---
    fig = _new_fig(9.0, 4.2)
    ax1 = fig.add_subplot(211)
    _shade_phases(ax1, a["t"], a["phase"])
    ax1.plot(a["t"], a["dz"] * 1000.0, color=SERIES[0], lw=1.5, label=r"$\Delta z$ (admittance)")
    ax1.axhline(0.0, color=GRID, lw=0.8)
    _style_axes(ax1, "", "Z correction [mm]", "Admittance Z correction and commanded height")
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")

    ax2 = fig.add_subplot(212, sharex=ax1)
    _shade_phases(ax2, a["t"], a["phase"])
    ax2.plot(a["t"], a["z"] * 1000.0, color=SERIES[2], lw=1.4, label="TCP z (measured)")
    ax2.plot(a["t"], a["cz"] * 1000.0, color=INK, lw=1.0, ls="--", label="z command")
    _style_axes(ax2, "time [s]", "height above table [mm]", "")
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_z_correction.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    paths.append(path)

    # --- 3. XY trajectory ---
    fig = _new_fig(7.2, 5.2)
    ax = fig.add_subplot(111)
    cfg = result.config
    _draw_table(ax, cfg)
    contact = a["contact"]
    ax.plot(a["x"][~contact], a["y"][~contact], color=INK_MUTED, lw=0.7, alpha=0.5,
            label="transfer (no contact)")
    xs = np.where(contact, a["x"], np.nan)
    ys = np.where(contact, a["y"], np.nan)
    ax.plot(xs, ys, color=SERIES[0], lw=1.8, label="sweeping (in contact)")
    if result.initial_positions is not None and len(result.initial_positions):
        p0 = np.asarray(result.initial_positions)
        ax.scatter(p0[:, 0], p0[:, 1], s=34, facecolor="none", edgecolor=INK_2,
                   linewidths=1.0, label="components (start)", zorder=4)
    if result.final_positions is not None and len(result.final_positions):
        p1 = np.asarray(result.final_positions)
        ax.scatter(p1[:, 0], p1[:, 1], s=34, color=SERIES[1], label="components (end)", zorder=5)
    _style_axes(ax, "x [m]  (sweep direction -X)", "y [m]",
                f"TCP trajectory -- collection rate {result.metrics.collection_rate:.0%}")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_xy.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    paths.append(path)

    # --- 4. contact / release event timeline ---
    fig = _new_fig(9.0, 2.6)
    ax = fig.add_subplot(111)
    phases = ["APPROACH", "SEARCH_CONTACT", "CONTACT_DETECTED", "FORCE_RAMP",
              "SWEEP", "FORCE_RELEASE", "RETRACT"]
    y_of = {name: i for i, name in enumerate(phases)}
    for name in phases:
        sel = a["phase"] == name
        if sel.any():
            ax.plot(a["t"][sel], np.full(sel.sum(), y_of[name]), "|", color=SERIES[0],
                    markersize=8, alpha=0.85)
    contact_t = a["t"][a["contact"]]
    if contact_t.size:
        ax.plot(contact_t, np.full(contact_t.size, len(phases)), "|",
                color=SERIES[2], markersize=8, alpha=0.9)
    ax.set_yticks(list(y_of.values()) + [len(phases)])
    ax.set_yticklabels(phases + ["IN CONTACT"], fontsize=7)
    _style_axes(ax, "time [s]", "", "Phase and contact timeline")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_events.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    paths.append(path)
    return paths


def _draw_table(ax, cfg: Optional[dict]) -> None:
    if not cfg:
        return
    half = cfg["table"]["half_size"]
    ax.add_patch(plt.Rectangle((-half[0], -half[1]), 2 * half[0], 2 * half[1],
                               facecolor="#f4f3f0", edgecolor=GRID, lw=1.0, zorder=0))
    tgt = cfg["target"]
    ax.add_patch(plt.Rectangle((tgt["x_min"], tgt["y_min"]),
                               tgt["x_max"] - tgt["x_min"], tgt["y_max"] - tgt["y_min"],
                               facecolor="#dce9f8", edgecolor=SERIES[0], lw=1.2,
                               zorder=1, label=None))
    ax.text(0.5 * (tgt["x_min"] + tgt["x_max"]), tgt["y_max"] + 0.015, "target",
            color=SERIES[0], fontsize=8, ha="center")
    ws = cfg["workspace"]
    ax.add_patch(plt.Rectangle((ws["x_min"], ws["y_min"]),
                               ws["x_max"] - ws["x_min"], ws["y_max"] - ws["y_min"],
                               facecolor="none", edgecolor=INK_MUTED, lw=0.8, ls=":", zorder=1))


def plot_observation(obs: dict, cfg: dict, out_path: str) -> Optional[str]:
    """RGB frame, image-space mask and the derived table-plane occupancy grid."""
    if obs.get("rgb") is None and obs.get("mask") is None:
        return None
    panels = [p for p in ("rgb", "mask") if obs.get(p) is not None] + ["occupancy"]
    fig = _new_fig(3.6 * len(panels), 3.2)
    for i, name in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1)
        if name == "rgb":
            ax.imshow(obs["rgb"])
            ax.set_title("fixed camera RGB", color=INK, fontsize=9, loc="left")
            ax.axis("off")
        elif name == "mask":
            ax.imshow(obs["mask"], cmap="gray")
            ax.set_title("component mask", color=INK, fontsize=9, loc="left")
            ax.axis("off")
        else:
            ws = cfg["workspace"]
            ax.imshow(obs["occupancy"], origin="lower", cmap="Blues",
                      extent=[ws["x_min"], ws["x_max"], ws["y_min"], ws["y_max"]])
            pts = np.asarray(obs["points"])
            if pts.size:
                ax.scatter(pts[:, 0], pts[:, 1], s=26, color=SERIES[1],
                           label="cluster centroids")
                ax.legend(frameon=False, fontsize=7, labelcolor=INK_2)
            _style_axes(ax, "x [m]", "y [m]", "table-plane occupancy")
            ax.set_aspect("equal")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------- experiments
_EXP_PANELS: Sequence = (
    ("collection_rate", "collection rate", "mean collection rate", (0.0, 1.05)),
    ("components_per_stroke", "components / stroke", "components collected per stroke", None),
    ("n_strokes", "strokes", "number of sweeping strokes", None),
    ("completion_time", "time [s]", "episode completion time", None),
    ("path_length", "path [m]", "total TCP path length", None),
    ("n_pushed_out", "components", "components pushed out of the workspace", None),
)


def plot_experiment(summary: List[dict], out_dir: str, prefix: str = "experiment") -> List[str]:
    """Grouped bars (mean +/- s.d.) per planner, one panel per metric."""
    os.makedirs(out_dir, exist_ok=True)
    planners = sorted({row["planner"] for row in summary})
    counts = sorted({int(row["n_components"]) for row in summary})
    if not planners or not counts:
        return []

    fig = _new_fig(12.0, 6.6)
    width = min(0.8 / max(1, len(planners)), 0.32)
    bar_handles, bar_labels = [], []
    for i, (metric, ylabel, title, ylim) in enumerate(_EXP_PANELS):
        ax = fig.add_subplot(2, 3, i + 1)
        for j, planner in enumerate(planners):
            means, stds = [], []
            for n in counts:
                row = next((r for r in summary
                            if r["planner"] == planner and int(r["n_components"]) == n), None)
                means.append(row[f"{metric}_mean"] if row else np.nan)
                stds.append(row[f"{metric}_std"] if row else 0.0)
            xs = np.arange(len(counts)) + (j - (len(planners) - 1) / 2) * width
            bars = ax.bar(xs, means, width=width * 0.92, yerr=stds, capsize=2,
                          color=PLANNER_COLOR.get(planner, SERIES[j % len(SERIES)]),
                          edgecolor=SURFACE, linewidth=1.2,
                          error_kw={"elinewidth": 0.9, "ecolor": INK_2},
                          label=PLANNER_LABEL.get(planner, planner), zorder=3)
            if i == 0:
                bar_handles.append(bars)
                bar_labels.append(PLANNER_LABEL.get(planner, planner))
        ax.set_xticks(np.arange(len(counts)))
        ax.set_xticklabels([str(n) for n in counts])
        _style_axes(ax, "components per episode", ylabel, title)
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle("Paired planner comparison -- identical layouts, identical low-level controller",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.legend(bar_handles, bar_labels, frameon=False, fontsize=9, labelcolor=INK_2,
               ncols=len(bar_labels), loc="upper right", bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, f"{prefix}_summary.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return [path]


def plot_force_quality(summary: List[dict], out_dir: str,
                       prefix: str = "experiment") -> List[str]:
    """Control-quality check: the low-level loop must be planner-independent."""
    os.makedirs(out_dir, exist_ok=True)
    planners = sorted({row["planner"] for row in summary})
    counts = sorted({int(row["n_components"]) for row in summary})
    panels = (("peak_normal_force", "peak normal force [N]"),
              ("rms_force_error", "RMS force tracking error [N]"),
              ("contact_loss_ratio", "contact-loss ratio"))
    fig = _new_fig(11.0, 3.4)
    width = min(0.8 / max(1, len(planners)), 0.32)
    handles, labels = [], []
    for i, (metric, ylabel) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1)
        for j, planner in enumerate(planners):
            means = [next((r[f"{metric}_mean"] for r in summary
                           if r["planner"] == planner and int(r["n_components"]) == n), np.nan)
                     for n in counts]
            stds = [next((r[f"{metric}_std"] for r in summary
                          if r["planner"] == planner and int(r["n_components"]) == n), 0.0)
                    for n in counts]
            xs = np.arange(len(counts)) + (j - (len(planners) - 1) / 2) * width
            bars = ax.bar(xs, means, width=width * 0.92, yerr=stds, capsize=2,
                          color=PLANNER_COLOR.get(planner, SERIES[j % len(SERIES)]),
                          edgecolor=SURFACE, linewidth=1.2,
                          error_kw={"elinewidth": 0.9, "ecolor": INK_2},
                          label=PLANNER_LABEL.get(planner, planner), zorder=3)
            if i == 0:
                handles.append(bars)
                labels.append(PLANNER_LABEL.get(planner, planner))
        ax.set_xticks(np.arange(len(counts)))
        ax.set_xticklabels([str(n) for n in counts])
        _style_axes(ax, "components per episode", ylabel, ylabel)
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_2,
               ncols=len(labels), loc="upper right", bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = os.path.join(out_dir, f"{prefix}_force_quality.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return [path]


def plot_paired_delta(rows: List[dict], metric: str, out_dir: str,
                      prefix: str = "experiment") -> Optional[str]:
    """Per-seed difference (Planner B - Planner A) on identical layouts."""
    if not rows:
        return None
    os.makedirs(out_dir, exist_ok=True)
    counts = sorted({int(r["n_components"]) for r in rows})
    fig = _new_fig(7.0, 3.4)
    ax = fig.add_subplot(111)
    data = [[r["delta"] for r in rows if int(r["n_components"]) == n] for n in counts]
    parts = ax.boxplot(data, positions=np.arange(len(counts)), widths=0.55,
                       patch_artist=True, medianprops={"color": INK, "linewidth": 1.4},
                       flierprops={"marker": "o", "markersize": 3,
                                   "markerfacecolor": INK_MUTED, "markeredgecolor": "none"})
    for patch in parts["boxes"]:
        patch.set_facecolor("#dce9f8")
        patch.set_edgecolor(SERIES[0])
    ax.axhline(0.0, color=INK_MUTED, lw=1.0, ls="--")
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels([str(n) for n in counts])
    _style_axes(ax, "components per episode", f"delta {metric}",
                f"Paired difference in {metric}: visual greedy - fixed (same seeds)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_paired_{metric}.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


# ------------------------------------------------------------- force sweep
#: (metric, panel title, y-axis unit label, y limits)
_SWEEP_PANELS: Sequence = (
    ("collection_rate", "collection rate", "rate", (0.0, 1.05)),
    ("n_ride_over", "ride-over  (force too low)", "events / episode", None),
    ("n_jam_events", "jams  (force too high)", "events / episode", None),
    ("touchdown_overshoot_max", "touchdown force overshoot", "[N]", None),
    ("n_pushed_out", "components ejected", "components", None),
    ("peak_normal_force", "peak normal force", "[N]", None),
)


def plot_force_sweep(summary: List[dict], out_dir: str,
                     prefix: str = "force_sweep") -> List[str]:
    """Failure modes vs the desired normal force, one line per geometry.

    Up to three geometries share a panel (the categorical palette is validated
    all-pairs at three slots).  Beyond that the figure becomes small multiples
    rather than cycling hues, which would put two indistinguishable colours on
    the same axes.
    """
    os.makedirs(out_dir, exist_ok=True)
    geometries = sorted({row["geometry"] for row in summary})
    forces = sorted({float(row["desired_force"]) for row in summary})
    if not geometries or not forces:
        return []
    if len(geometries) > 3:
        return _plot_force_sweep_facets(summary, geometries, forces, out_dir, prefix)

    fig = _new_fig(12.0, 6.6)
    handles, labels = [], []
    for i, (metric, title, ylabel, ylim) in enumerate(_SWEEP_PANELS):
        ax = fig.add_subplot(2, 3, i + 1)
        for j, geometry in enumerate(geometries):
            ys = [next((r[f"{metric}_mean"] for r in summary
                        if r["geometry"] == geometry
                        and float(r["desired_force"]) == f), np.nan) for f in forces]
            line, = ax.plot(forces, ys, marker="o", markersize=5, lw=2.0,
                            color=SERIES[j % 3], markeredgecolor=SURFACE,
                            markeredgewidth=1.0, label=geometry, zorder=3)
            if i == 0:
                handles.append(line)
                labels.append(geometry)
        _style_axes(ax, "desired normal force [N]", ylabel, title)
        if ylim:
            ax.set_ylim(*ylim)
    fig.suptitle("Force sweep -- where a single fixed F_z* stops working",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK_2,
               ncols=len(labels), loc="upper right", bbox_to_anchor=(0.995, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, f"{prefix}_summary.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return [path]


def _plot_force_sweep_facets(summary, geometries, forces, out_dir, prefix) -> List[str]:
    metrics = [("collection_rate", "collection rate"),
               ("n_ride_over", "ride-over"),
               ("n_jam_events", "jams")]
    rows, cols = len(metrics), len(geometries)
    fig = _new_fig(2.9 * cols, 2.6 * rows)
    for r, (metric, ylabel) in enumerate(metrics):
        for c, geometry in enumerate(geometries):
            ax = fig.add_subplot(rows, cols, r * cols + c + 1)
            ys = [next((row[f"{metric}_mean"] for row in summary
                        if row["geometry"] == geometry
                        and float(row["desired_force"]) == f), np.nan) for f in forces]
            ax.plot(forces, ys, marker="o", markersize=5, lw=2.0, color=SERIES[0],
                    markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
            _style_axes(ax, "F_z* [N]" if r == rows - 1 else "",
                        ylabel if c == 0 else "", geometry if r == 0 else "")
            if metric == "collection_rate":
                ax.set_ylim(0.0, 1.05)
    fig.suptitle("Force sweep by geometry", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(out_dir, f"{prefix}_facets.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return [path]
