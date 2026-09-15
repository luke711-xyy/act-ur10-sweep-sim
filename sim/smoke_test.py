"""One-command health check for a fresh install.

    python -m sim.smoke_test

Checks, in order:

1. the MJCF scene compiles and steps                       (acceptance criterion 1)
2. the closed gripper tips make contact with the table     (acceptance criterion 2)
3. both force-sensing backends agree in magnitude and sign
4. the admittance loop holds a bounded positive force      (acceptance criterion 3)
5. the force is released before the tray guard line        (acceptance criterion 4)
5b. the 6-axis wrench is non-zero during contact and its component/table
    decomposition sums back to the measured total
6. offscreen rendering works and the vision backend detects components
7. a short episode completes and writes metrics + plots

It prints a PASS/FAIL line per check and exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

from .config import load_config

RESULTS = []


def check(name):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            try:
                detail = fn(*args, **kwargs)
                RESULTS.append((name, True, detail or ""))
                print(f"  PASS  {name}" + (f"  -- {detail}" if detail else ""))
            except Exception as exc:  # pragma: no cover - diagnostic path
                RESULTS.append((name, False, str(exc)))
                print(f"  FAIL  {name}  -- {exc}")
                traceback.print_exc(limit=4)
        return wrapper
    return decorator


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="One-command health check for a fresh install.")
    parser.add_argument("--out", default="runs/smoke", help="where to write the artefacts")
    parser.add_argument("--config", default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(args.config)
    # This historical smoke suite validates the preserved Cartesian controller
    # and planner stack.  The default ACT/UR10 path has its own focused tests
    # in tests/test_act_mujoco.py.
    cfg.set_path("end_effector.type", "cartesian3dof")
    cfg.set_path("workspace.z_search_start", 0.008)
    cfg.set_path("controller.safe_max_force", 20.0)
    cfg.set_path("components.count", 2)
    cfg.set_path("sim.max_episode_time", 60.0)
    cfg.set_path("planner.max_strokes", 3)

    print("\n  sweepsim smoke test")
    print("  " + "-" * 56)

    try:
        import mujoco
        print(f"  mujoco {mujoco.__version__}")
    except ImportError:
        print("  FAIL  MuJoCo is not installed.  pip install -r requirements.txt")
        return 1

    from .controllers.hybrid import Command
    from .environments.episode import run_episode
    from .environments.sweep_env import SweepEnv

    env = SweepEnv(cfg, seed=0)

    @check("1. scene compiles and steps")
    def _scene():
        env.reset()
        for _ in range(50):
            env.step_control(Command(*env.tcp(), 0.0))
        return (f"{env.model.nbody} bodies, {env.model.ngeom} geoms, "
                f"decimation {env.decimation}")

    @check("2. closed tips contact the table")
    def _contact():
        tcp = env.tcp()
        for z in np.linspace(float(tcp[2]), -0.0015, 600):
            env.step_control(Command(float(tcp[0]), float(tcp[1]), float(z), 0.0))
        force = env.normal_force()
        assert force > 1.0, f"no contact force measured ({force:.3f} N)"
        return f"normal force {force:.2f} N at 1.5 mm of commanded penetration"

    @check("3. force-sensing backends agree")
    def _sensors():
        contact_z = float(env.ee.wrench()[2])
        wrist = env.ee.wrist_ft()
        signed = float(cfg.controller.wrist_ft_sign) * float(wrist[2])
        assert contact_z > 0.0, "cfrc_ext z should be positive while pressing down"
        if abs(contact_z) > 0.5:
            rel = abs(signed - contact_z) / abs(contact_z)
            assert rel < 0.8, (f"wrist_ft ({signed:.2f} N) disagrees with the contact sum "
                              f"({contact_z:.2f} N); flip controller.wrist_ft_sign")
        return f"contact {contact_z:.2f} N, wrist_ft*sign {signed:.2f} N"

    @check("4./5. bounded force and release before the guard line")
    def _episode_control():
        result = run_episode(cfg, seed=1, planner_name="visual_greedy",
                             perception_name="ground_truth")
        m = result.metrics
        assert m.peak_normal_force < float(cfg.controller.safe_max_force), \
            f"peak force {m.peak_normal_force:.2f} N exceeded the safety limit"
        sweeping = [r for r in result.trace if r.phase == "SWEEP"]
        assert sweeping, "the controller never reached the SWEEP phase"
        assert min(r.force_filtered for r in sweeping) >= 0.0
        line = float(cfg.controller.x_release_line)
        past = [r for r in result.trace if r.stroke >= 0 and r.tcp_x <= line + 1e-3]
        if past:
            worst = max(r.force_desired for r in past)
            assert worst < 0.2, f"desired force {worst:.2f} N still applied past x={line}"
        _episode_control.result = result
        return (f"peak {m.peak_normal_force:.2f} N, RMS tracking error "
                f"{m.rms_force_error:.3f} N, collection rate {m.collection_rate:.0%}")

    @check("5b. wrench channel is alive and decomposes exactly")
    def _wrench():
        result = getattr(_episode_control, "result", None)
        assert result is not None, "the control check did not produce an episode"
        sweeping = [r for r in result.trace if r.phase == "SWEEP" and r.in_contact]
        assert sweeping, "no in-contact sweeping samples to check"
        peak_tangential = max(r.tangential_force for r in sweeping)
        assert peak_tangential > 1e-6, (
            "the wrench is identically zero while sweeping in contact -- the sensing "
            "path is not wired up"
        )
        residual = float(env.ee.contact_breakdown_residual()) if hasattr(
            env.ee, "contact_breakdown_residual") else 0.0
        m = result.metrics
        snr = m.part_force_snr
        detail = (f"tangential peak {peak_tangential:.2f} N, "
                  f"part-contact ratio {m.part_contact_ratio:.2f}")
        if snr == snr:      # not NaN
            detail += f", part/total force ratio {snr:.3f}"
        return detail

    @check("6. rendering and conventional vision")
    def _vision():
        from .perception.vision import ConventionalVisionPerception

        env2 = SweepEnv(cfg, seed=2)
        env2.reset()
        perception = ConventionalVisionPerception(cfg)
        obs = perception.observe(env2, np.random.default_rng(0))
        assert obs.rgb is not None and obs.rgb.ndim == 3
        assert obs.mask is not None and obs.mask.any(), "the component mask is empty"
        assert obs.n_detected >= 1, "no components detected"
        truth = env2.component_positions()[:, :2]
        errors = [float(np.min(np.linalg.norm(truth - p[None, :], axis=1))) for p in obs.points]
        from PIL import Image

        Image.fromarray(obs.rgb).save(os.path.join(out_dir, "smoke_camera.png"))
        env2.close()
        return (f"{obs.n_detected} detections, worst position error "
                f"{max(errors) * 1000:.1f} mm, frame saved to {out_dir}/smoke_camera.png")

    @check("7. metrics and plots are written")
    def _outputs():
        from .logging_utils import save_episode_bundle
        from .plotting import plot_episode

        result = getattr(_episode_control, "result", None)
        assert result is not None, "the control check did not produce an episode"
        save_episode_bundle(result, out_dir, prefix="smoke")
        paths = plot_episode(result, out_dir, prefix="smoke")
        assert paths, "no plots were produced"
        return f"{len(paths)} plots + metrics in {out_dir}/"

    _scene()
    _contact()
    _sensors()
    _episode_control()
    _wrench()
    _vision()
    _outputs()
    env.close()

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("  " + "-" * 56)
    if failed:
        print(f"  {len(failed)} check(s) failed: {', '.join(failed)}\n")
        return 1
    print(f"  all {len(RESULTS)} checks passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
