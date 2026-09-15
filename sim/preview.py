"""2-D animated preview of an episode, rendered with Matplotlib.

This is the **no-GL** companion to :mod:`sim.video`. It draws what the planner
and the controller did -- stroke geometry, phase timing, component motion, the
force trace -- straight from the episode record, so it works on any machine, in
CI, and over SSH, with no MuJoCo renderer and no OpenGL context.

What it is not
--------------
It is a plot, not a camera. It shows the task from directly above as flat shapes.
Use :mod:`sim.video` when you want the actual rendered scene.

Every frame carries a provenance banner naming the data source, so a file that
travels on its own can never be mistaken for rendered physics.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import Config  # noqa: E402
from .planners.geometry_utils import pusher_width  # noqa: E402
from .plotting import GRID, INK, INK_2, INK_MUTED, SERIES, SURFACE, _style_axes  # noqa: E402

PHASE_COLOR = {
    "APPROACH": INK_MUTED, "OBSERVE_MOVE": INK_MUTED, "SEARCH_CONTACT": SERIES[3],
    "CONTACT_DETECTED": SERIES[2], "FORCE_RAMP": SERIES[2], "SWEEP": SERIES[0],
    "FORCE_RELEASE": SERIES[1], "RETRACT": INK_MUTED,
}


def _tool_polygon(x: float, y: float, yaw: float, half_x: float, half_y: float) -> np.ndarray:
    corners = np.array([[-half_x, -half_y], [half_x, -half_y],
                        [half_x, half_y], [-half_x, half_y]])
    c, s = np.cos(yaw), np.sin(yaw)
    return corners @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def animate_episode(
    result,
    out_path: str,
    fps: int = 25,
    speed: float = 1.0,
    force_window: float = 8.0,
    source: str = "kinematic preview -- NOT MuJoCo physics",
    dpi: int = 110,
) -> Optional[str]:
    """Write an animated top-down preview of ``result``.

    Parameters
    ----------
    result
        An :class:`~sim.environments.episode.EpisodeResult`.  Run the episode with
        ``track_every > 0`` so component motion is recorded; without it the parts
        are drawn at their start positions only.
    speed
        Playback speed relative to simulated time (2.0 = twice as fast).
    source
        Provenance string drawn into every frame.  Pass ``"MuJoCo"`` when the
        result came from the real simulator.
    """
    trace = result.trace
    if not trace:
        return None
    cfg = result.config
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    t = np.array([r.t for r in trace], dtype=float)
    x = np.array([r.tcp_x for r in trace], dtype=float)
    y = np.array([r.tcp_y for r in trace], dtype=float)
    yaw = np.array([r.cmd_yaw for r in trace], dtype=float)
    fd = np.array([r.force_desired for r in trace], dtype=float)
    ff = np.array([r.force_filtered for r in trace], dtype=float)
    contact = np.array([r.in_contact for r in trace], dtype=bool)
    phase = np.array([r.phase for r in trace], dtype=object)
    stroke = np.array([r.stroke for r in trace], dtype=int)

    frame_times = np.arange(t[0], t[-1], max(speed / float(fps), 1e-6))
    index_of = np.searchsorted(t, frame_times).clip(0, t.size - 1)

    tracks = getattr(result, "component_tracks", None)
    track_times = getattr(result, "track_times", None)
    if tracks is not None and track_times is not None and len(track_times):
        track_index = np.searchsorted(track_times, frame_times).clip(0, len(track_times) - 1)
    else:
        tracks, track_index = None, None
    start_positions = np.asarray(result.initial_positions)[:, :2]

    # result.config is a plain dict; reuse the single definition of pusher width
    half = pusher_width(Config(cfg)) / 2.0
    tip_half_x = float(cfg["end_effector"]["tip_half"][0])

    fig = plt.figure(figsize=(11.0, 5.0), facecolor=SURFACE, dpi=dpi)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.24,
                            left=0.055, right=0.985, top=0.86, bottom=0.14)
    ax_scene = fig.add_subplot(grid[0, 0])
    ax_force = fig.add_subplot(grid[0, 1])

    _draw_static_scene(ax_scene, cfg)
    ax_scene.scatter(start_positions[:, 0], start_positions[:, 1], s=30, facecolor="none",
                     edgecolor=INK_MUTED, linewidths=1.0, zorder=3, label="start")
    (path_line,) = ax_scene.plot([], [], color=INK_MUTED, lw=0.8, alpha=0.5, zorder=4)
    (contact_line,) = ax_scene.plot([], [], color=SERIES[0], lw=2.0, zorder=5)
    parts = ax_scene.scatter(start_positions[:, 0], start_positions[:, 1], s=46,
                             color=SERIES[1], zorder=6, label="components")
    tool = plt.Polygon(_tool_polygon(x[0], y[0], yaw[0], tip_half_x, half),
                       closed=True, facecolor="#2b2b30", edgecolor=INK, lw=1.0, zorder=7)
    ax_scene.add_patch(tool)
    _style_axes(ax_scene, "x [m]   (sweep direction -X)", "y [m]", "top view")
    ax_scene.set_aspect("equal")
    ax_scene.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")

    f_top = max(float(np.max(ff)) * 1.25, float(np.max(fd)) * 1.6, 1.0)
    ax_force.set_ylim(0.0, f_top)
    (line_meas,) = ax_force.plot([], [], color=SERIES[0], lw=1.8, label="measured")
    (line_des,) = ax_force.plot([], [], color=INK, lw=1.4, ls="--", label="desired")
    _style_axes(ax_force, "time [s]", "normal force [N]", "hybrid controller: Z axis")
    ax_force.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")

    banner = fig.text(0.055, 0.945, "", color=INK, fontsize=11, ha="left", va="center")
    subtitle = fig.text(0.985, 0.945, "", color=INK_MUTED, fontsize=9, ha="right", va="center")
    readout = fig.text(0.055, 0.035, "", color=INK_2, fontsize=9, ha="left", va="center")

    metrics = result.metrics
    total = len(result.layout)

    def update(frame: int):
        i = int(index_of[frame])
        now = float(frame_times[frame])
        path_line.set_data(x[: i + 1], y[: i + 1])
        masked_x = np.where(contact[: i + 1], x[: i + 1], np.nan)
        masked_y = np.where(contact[: i + 1], y[: i + 1], np.nan)
        contact_line.set_data(masked_x, masked_y)
        tool.set_xy(_tool_polygon(x[i], y[i], yaw[i], tip_half_x, half))
        tool.set_edgecolor(PHASE_COLOR.get(str(phase[i]), INK))
        tool.set_linewidth(2.2 if contact[i] else 1.0)

        if tracks is not None:
            parts.set_offsets(tracks[int(track_index[frame])])

        lo = max(0, int(np.searchsorted(t, now - force_window)))
        line_meas.set_data(t[lo : i + 1], ff[lo : i + 1])
        line_des.set_data(t[lo : i + 1], fd[lo : i + 1])
        ax_force.set_xlim(max(0.0, now - force_window), max(now, force_window * 0.25))

        banner.set_text(f"{metrics.planner}   ·   {total} components   ·   seed {metrics.seed}")
        subtitle.set_text(source)
        readout.set_text(
            f"t = {now:6.2f} s      stroke {int(stroke[i]):>2d}      {str(phase[i]):<15s}"
            f"      F = {ff[i]:5.2f} / {fd[i]:4.2f} N"
            f"      {'IN CONTACT' if contact[i] else '          '}"
        )
        return path_line, contact_line, tool, parts, line_meas, line_des

    anim = animation.FuncAnimation(fig, update, frames=len(frame_times), blit=False)
    writer = _writer(out_path, fps)
    anim.save(out_path, writer=writer, savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    return out_path


def ensure_ffmpeg() -> bool:
    """Make Matplotlib's ffmpeg writer usable, borrowing imageio-ffmpeg's binary.

    ``pip install imageio-ffmpeg`` ships an ffmpeg executable but does **not** put
    it on PATH, so Matplotlib's FFMpegWriter reports itself unavailable on a
    clean Windows install even though a perfectly good ffmpeg is sitting in
    site-packages.  Point Matplotlib at it instead of silently degrading to GIF.
    """
    if animation.FFMpegWriter.isAvailable():
        return True
    try:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                   # pragma: no cover
        return False
    return animation.FFMpegWriter.isAvailable()


def _writer(out_path: str, fps: int):
    if out_path.lower().endswith(".gif"):
        return animation.PillowWriter(fps=fps)
    if ensure_ffmpeg():
        return animation.FFMpegWriter(fps=fps, bitrate=2400,
                                      extra_args=["-pix_fmt", "yuv420p"])
    print("  preview: ffmpeg not available, falling back to a GIF writer "
          "(install imageio-ffmpeg for MP4)")
    return animation.PillowWriter(fps=fps)


def _draw_static_scene(ax, cfg) -> None:
    half = cfg["table"]["half_size"]
    ax.add_patch(plt.Rectangle((-half[0], -half[1]), 2 * half[0], 2 * half[1],
                               facecolor="#f4f3f0", edgecolor=GRID, lw=1.0, zorder=0))
    tgt = cfg["target"]
    ax.add_patch(plt.Rectangle((tgt["x_min"], tgt["y_min"]),
                               tgt["x_max"] - tgt["x_min"], tgt["y_max"] - tgt["y_min"],
                               facecolor="#dce9f8", edgecolor=SERIES[0], lw=1.4, zorder=1))
    ax.text(0.5 * (tgt["x_min"] + tgt["x_max"]), tgt["y_max"] + 0.018, "target tray",
            color=SERIES[0], fontsize=8, ha="center")
    ws = cfg["workspace"]
    ax.add_patch(plt.Rectangle((ws["x_min"], ws["y_min"]),
                               ws["x_max"] - ws["x_min"], ws["y_max"] - ws["y_min"],
                               facecolor="none", edgecolor=INK_MUTED, lw=0.8, ls=":", zorder=1))
    ax.axvline(cfg["controller"]["x_release_line"], color=SERIES[1], lw=0.9, ls="--",
               alpha=0.7, zorder=1)
    ax.text(cfg["controller"]["x_release_line"], ws["y_min"] - 0.035, "force release",
            color=SERIES[1], fontsize=7, ha="center")


def animate_comparison(
    results: Sequence,
    out_path: str,
    fps: int = 25,
    speed: float = 6.0,
    labels: Optional[Sequence[str]] = None,
    source: str = "kinematic preview -- NOT MuJoCo physics",
    dpi: int = 110,
) -> Optional[str]:
    """Play several planners on the *same* layout side by side, on one clock.

    This is the experimental claim as a picture: identical initial scene,
    identical low-level controller, different stroke planner.  Each pane freezes
    when its episode ends, so the difference in episode length is visible
    directly rather than having to be read off a bar chart.
    """
    results = [r for r in results if r.trace]
    if not results:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    labels = list(labels or [r.metrics.planner for r in results])
    cfg = results[0].config
    half = pusher_width(Config(cfg)) / 2.0
    tip_half_x = float(cfg["end_effector"]["tip_half"][0])

    per = []
    for result in results:
        trace = result.trace
        per.append({
            "t": np.array([r.t for r in trace]),
            "x": np.array([r.tcp_x for r in trace]),
            "y": np.array([r.tcp_y for r in trace]),
            "yaw": np.array([r.cmd_yaw for r in trace]),
            "contact": np.array([r.in_contact for r in trace], dtype=bool),
            "stroke": np.array([r.stroke for r in trace], dtype=int),
            "tracks": getattr(result, "component_tracks", None),
            "track_times": getattr(result, "track_times", None),
            "start": np.asarray(result.initial_positions)[:, :2],
            "metrics": result.metrics,
        })

    duration = max(float(d["t"][-1]) for d in per)
    frame_times = np.arange(0.0, duration + 1e-9, max(speed / float(fps), 1e-6))

    n = len(per)
    fig = plt.figure(figsize=(5.0 * n, 5.2), facecolor=SURFACE, dpi=dpi)
    grid = fig.add_gridspec(1, n, wspace=0.18, left=0.04, right=0.985, top=0.84, bottom=0.12)

    artists = []
    for i, data in enumerate(per):
        ax = fig.add_subplot(grid[0, i])
        _draw_static_scene(ax, cfg)
        ax.scatter(data["start"][:, 0], data["start"][:, 1], s=26, facecolor="none",
                   edgecolor=INK_MUTED, linewidths=1.0, zorder=3)
        (path_line,) = ax.plot([], [], color=INK_MUTED, lw=0.7, alpha=0.5, zorder=4)
        (contact_line,) = ax.plot([], [], color=SERIES[0], lw=1.8, zorder=5)
        parts = ax.scatter(data["start"][:, 0], data["start"][:, 1], s=42,
                           color=SERIES[1], zorder=6)
        tool = plt.Polygon(_tool_polygon(data["x"][0], data["y"][0], data["yaw"][0],
                                         tip_half_x, half),
                           closed=True, facecolor="#2b2b30", edgecolor=INK, lw=1.0, zorder=7)
        ax.add_patch(tool)
        _style_axes(ax, "x [m]", "y [m]" if i == 0 else "", labels[i])
        ax.set_aspect("equal")
        caption = ax.text(0.02, 0.02, "", transform=ax.transAxes, color=INK_2, fontsize=9,
                          ha="left", va="bottom")
        artists.append((path_line, contact_line, parts, tool, caption))

    banner = fig.text(0.04, 0.935, "", color=INK, fontsize=12, ha="left", va="center")
    subtitle = fig.text(0.985, 0.935, "", color=INK_MUTED, fontsize=9, ha="right", va="center")
    clock = fig.text(0.04, 0.035, "", color=INK_2, fontsize=10, ha="left", va="center")

    total = len(results[0].layout)
    seed = results[0].metrics.seed

    def update(frame: int):
        now = float(frame_times[frame])
        for data, (path_line, contact_line, parts, tool, caption) in zip(per, artists):
            t = data["t"]
            done = now >= t[-1]
            i = int(np.searchsorted(t, min(now, t[-1])).clip(0, t.size - 1))
            path_line.set_data(data["x"][: i + 1], data["y"][: i + 1])
            contact_line.set_data(
                np.where(data["contact"][: i + 1], data["x"][: i + 1], np.nan),
                np.where(data["contact"][: i + 1], data["y"][: i + 1], np.nan),
            )
            tool.set_xy(_tool_polygon(data["x"][i], data["y"][i], data["yaw"][i],
                                      tip_half_x, half))
            tool.set_linewidth(2.2 if data["contact"][i] else 1.0)
            if data["tracks"] is not None and data["track_times"] is not None \
                    and len(data["track_times"]):
                k = int(np.searchsorted(data["track_times"], min(now, t[-1]))
                        .clip(0, len(data["track_times"]) - 1))
                parts.set_offsets(data["tracks"][k])
            m = data["metrics"]
            caption.set_text(
                (f"FINISHED  {m.n_strokes} strokes  {m.completion_time:.0f} s  "
                 f"{m.path_length:.1f} m  {m.collection_rate:.0%}")
                if done else
                f"stroke {int(data['stroke'][i]):>2d}   running"
            )
            caption.set_color(SERIES[2] if done else INK_2)
        banner.set_text(f"same layout · {total} components · seed {seed} · "
                        f"identical hybrid force controller")
        subtitle.set_text(source)
        clock.set_text(f"simulated time  {now:6.1f} s      (playback {speed:.0f}x)")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frame_times), blit=False)
    anim.save(out_path, writer=_writer(out_path, fps),
              savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    return out_path
