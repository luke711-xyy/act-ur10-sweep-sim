"""Multi-camera episode recording.

Renders one pane per camera, stacks them side by side and draws a HUD strip
underneath showing what the controller is doing at that instant: phase, stroke
index, desired vs measured normal force, contact state and collection progress.

The extra cameras exist **only** here.  Perception looks up ``scene_cam`` by
name and nothing else, so adding view angles can never leak information into a
planner or into an exported dataset.

Frames are streamed to the writer rather than buffered, so a 200 s episode costs
a few megabytes of RAM instead of a few gigabytes.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np

HUD_HEIGHT = 72
PANE_LABEL_HEIGHT = 22

_INK = (250, 250, 248)
_INK_DIM = (168, 168, 160)
_BG = (24, 24, 26)
_ACCENT = (42, 120, 214)      # blue   -- measured force
_WARN = (235, 104, 52)        # orange -- force above target
_GOOD = (27, 175, 122)        # aqua   -- in contact


def _font(size: int):
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # Pillow < 10.1
        return ImageFont.load_default()


class EpisodeRecorder:
    """Streams a multi-camera video of one episode."""

    def __init__(self, cfg, path: str, cameras: Optional[Sequence[str]] = None,
                 fps: Optional[int] = None, hud: Optional[bool] = None,
                 every_n: Optional[int] = None, size=None):
        self.cfg = cfg
        vcfg = cfg.get_path("video", None)
        self.cameras: List[str] = list(cameras or (vcfg.cameras if vcfg else ["scene_cam"]))
        self.fps = int(fps or (vcfg.fps if vcfg else 25))
        self.every_n = int(every_n or (vcfg.every_n_control_steps if vcfg else 4))
        self.hud = bool(cfg.get_path("video.hud", True) if hud is None else hud)
        self.width = int(size[0] if size else cfg.get_path("video.width", 640))
        self.height = int(size[1] if size else cfg.get_path("video.height", 480))
        self.path = path
        self._renderer = None
        self._writer = None
        self._counter = 0
        self.n_frames = 0
        self._fmt = os.path.splitext(path)[1].lower().lstrip(".") or "mp4"
        # The gauge is scaled to a few times the setpoint, not to safe_max_force:
        # at a 3 N target on a 25 N axis the bar would sit at 12 % and show nothing.
        gauge = cfg.get_path("video.force_gauge_max", None)
        self.force_gauge_max = float(gauge) if gauge else max(
            2.5 * float(cfg.controller.desired_force), 1.0)
        self._label_font = _font(14)
        self._hud_font = _font(15)
        self._hud_font_small = _font(12)

    # ------------------------------------------------------------------ setup
    def attach(self, env) -> "EpisodeRecorder":
        import mujoco

        available = {
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(env.model.ncam)
        }
        missing = [c for c in self.cameras if c not in available]
        if missing:
            raise ValueError(
                f"camera(s) {missing} are not in the scene; available: {sorted(available)}. "
                "Add them under video.extra_cameras in the config."
            )
        self._renderer = mujoco.Renderer(env.model, height=self.height, width=self.width)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or ".", exist_ok=True)
        self._writer = self._open_writer()
        return self

    def _open_writer(self):
        import imageio.v2 as imageio

        if self._fmt == "gif":
            return imageio.get_writer(self.path, mode="I", fps=self.fps, loop=0)
        try:
            return imageio.get_writer(
                self.path, fps=self.fps, codec="libx264",
                quality=int(self.cfg.get_path("video.quality", 7)),
                macro_block_size=1, ffmpeg_log_level="error",
            )
        except Exception as exc:                       # pragma: no cover - env dependent
            gif = os.path.splitext(self.path)[0] + ".gif"
            print(f"  video: mp4 writer unavailable ({exc}); falling back to {gif}")
            self.path, self._fmt = gif, "gif"
            return imageio.get_writer(gif, mode="I", fps=self.fps, loop=0)

    # ------------------------------------------------------------------ frames
    def capture(self, env, state: Optional[dict] = None, force: bool = False) -> None:
        """Render a frame every ``every_n`` control steps (always when ``force``)."""
        if self._writer is None:
            return
        self._counter += 1
        if not force and (self._counter % self.every_n) != 0:
            return
        panes = []
        for camera in self.cameras:
            self._renderer.update_scene(env.data, camera=camera)
            panes.append(self._label_pane(self._renderer.render(), camera))
        frame = np.concatenate(panes, axis=1)
        if self.hud:
            frame = np.concatenate([frame, self._hud(frame.shape[1], state or {})], axis=0)
        self._writer.append_data(self._pad(frame))
        self.n_frames += 1

    @staticmethod
    def _pad(frame: np.ndarray, multiple: int = 16) -> np.ndarray:
        h, w = frame.shape[:2]
        ph, pw = (-h) % multiple, (-w) % multiple
        if ph or pw:
            frame = np.pad(frame, ((0, ph), (0, pw), (0, 0)), mode="edge")
        return frame

    def _label_pane(self, image: np.ndarray, camera: str) -> np.ndarray:
        from PIL import Image, ImageDraw

        img = Image.fromarray(np.ascontiguousarray(image))
        draw = ImageDraw.Draw(img, "RGBA")
        draw.rectangle([0, 0, img.width, PANE_LABEL_HEIGHT], fill=(0, 0, 0, 150))
        draw.text((8, 4), camera, fill=_INK, font=self._label_font)
        return np.asarray(img)

    def _hud(self, width: int, state: dict) -> np.ndarray:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (width, HUD_HEIGHT), _BG)
        draw = ImageDraw.Draw(img)

        phase = str(state.get("phase", "-"))
        in_contact = bool(state.get("in_contact", False))
        f_des = float(state.get("force_desired", 0.0))
        f_meas = float(state.get("force_measured", 0.0))
        f_max = self.force_gauge_max
        over = f_meas > f_max

        left = (f"t = {float(state.get('t', 0.0)):6.2f} s     "
                f"stroke {int(state.get('stroke', -1)):>2d}     {phase}")
        draw.text((14, 10), left, fill=_INK, font=self._hud_font)
        draw.text((14, 34), f"planner: {state.get('planner', '-')}",
                  fill=_INK_DIM, font=self._hud_font_small)
        draw.text((14, 50), f"collected {int(state.get('collected', 0))}"
                            f" / {int(state.get('total', 0))}",
                  fill=_INK_DIM, font=self._hud_font_small)

        # normal-force gauge: filled bar = measured, tick = desired setpoint
        bar_x0, bar_x1 = int(width * 0.42), int(width * 0.90)
        bar_y, bar_h = 26, 16
        draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + bar_h], fill=(52, 52, 56))
        span = max(bar_x1 - bar_x0, 1)
        filled = int(span * min(max(f_meas / max(f_max, 1e-6), 0.0), 1.0))
        colour = _WARN if (over or (f_meas > f_des * 1.25 and f_des > 0.05)) else _ACCENT
        if filled > 0:
            draw.rectangle([bar_x0, bar_y, bar_x0 + filled, bar_y + bar_h], fill=colour)
        tick = bar_x0 + int(span * min(max(f_des / max(f_max, 1e-6), 0.0), 1.0))
        draw.line([tick, bar_y - 4, tick, bar_y + bar_h + 4], fill=_INK, width=2)
        draw.text((bar_x0, 8), "normal force", fill=_INK_DIM, font=self._hud_font_small)
        draw.text((bar_x0, bar_y + bar_h + 6),
                  f"measured {f_meas:5.2f} N     target {f_des:5.2f} N"
                  f"     scale 0 - {f_max:.1f} N"
                  + ("   OVER" if over else ""),
                  fill=_WARN if over else _INK_DIM, font=self._hud_font_small)

        dot_x = bar_x1 + 16
        if dot_x + 12 < width:
            draw.ellipse([dot_x, bar_y + 2, dot_x + 12, bar_y + 14],
                         fill=_GOOD if in_contact else (70, 70, 74))
        return np.asarray(img)

    # ------------------------------------------------------------------ finish
    def close(self) -> Optional[str]:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:                          # pragma: no cover
                pass
            self._renderer = None
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def recorder_state(env, controller, planner_name: str) -> dict:
    """Build the HUD state dict from the live controller and environment."""
    record = controller.trace[-1] if controller.trace else None
    collected = int(env.collected_mask().sum()) if hasattr(env, "collected_mask") else 0
    total = len(getattr(env, "layout", []) or [])
    return {
        "t": env.time,
        "phase": record.phase if record else "-",
        "stroke": record.stroke if record else -1,
        "force_desired": record.force_desired if record else 0.0,
        "force_measured": record.force_filtered if record else 0.0,
        "in_contact": bool(record.in_contact) if record else False,
        "collected": collected,
        "total": total,
        "planner": planner_name,
    }
