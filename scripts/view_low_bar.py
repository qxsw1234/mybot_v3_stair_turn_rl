#!/usr/bin/env python3
"""Headless snapshot viewer for the ROBOCON low-bar (限高杆) field.

The low-bar gate is a *collision* asset, not terrain heightfield, so it cannot
be seen in the terrain map -- it has to be rendered.  This machine has no X
display, so this script renders offscreen with an Isaac Gym camera sensor and
writes PNG files (no interactive window needed).

It builds the scene with the exact low_bar configuration from the training
entry point (by stubbing the PPO runner), then overrides the gate clearance for
the snapshot so any curriculum row can be previewed.

Run from the project root::

    PYTHONPATH=/home/ldl/mybot_v3_stair_turn_rl \
    python -u scripts/view_low_bar.py --clearance 0.30

Outputs: ``runs/low_bar_view/low_bar_c<cm>.png`` and ``..._side.png``.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for _p in (str(SCRIPTS_ROOT), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import isaacgym  # noqa: F401,E402  (must precede torch)
from isaacgym import gymapi  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


def build_low_bar_env(clearance: float):
    import robodog_gym.envs.wrappers.history_wrapper as hw
    import robodog_gym_learn.ppo_cse as ppo

    captured = {}

    class _Capture:
        def __init__(self, env, cfg=None, device=None):
            captured["env"] = env
            captured["cfg"] = cfg

        def learn(self, *a, **k):
            return

    real_runner = ppo.Runner
    real_wrapper = hw.HistoryWrapper
    ppo.Runner = _Capture
    hw.HistoryWrapper = lambda e: e
    try:
        spec = importlib.util.spec_from_file_location(
            "train_mybot_v3_stair_turn",
            str(PROJECT_ROOT / "scripts" / "train_mybot_v3_stair_turn.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.train_mybot_v3_stair_turn(
            headless=True, num_envs=4, iterations=0, robocon_obstacle="low_bar"
        )
    finally:
        ppo.Runner = real_runner
        hw.HistoryWrapper = real_wrapper

    env = captured["env"]
    # Pin the gate to the requested clearance, independent of terrain row, and
    # push the new pose through the real refresh path.
    env._low_bar_clearance_for_env = lambda i, c=clearance: c
    env._refresh_low_bar_poses(torch.arange(env.num_envs, device=env.device))
    return env, captured["cfg"]


def render_png(env, path: Path, eye, target, w=1000, h=750):
    g = env.gym
    props = gymapi.CameraProperties()
    props.width = w
    props.height = h
    cam = g.create_camera_sensor(env.envs[0], props)
    g.set_camera_location(cam, env.envs[0], gymapi.Vec3(*eye), gymapi.Vec3(*target))
    g.step_graphics(env.sim)
    g.render_all_camera_sensors(env.sim)
    img = g.get_camera_image(env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
    img = img.reshape(h, w, 4)[:, :, :3]           # BGRA -> BGR
    img = np.flipud(img)[:, :, ::-1]               # flip + BGR -> RGB
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    plt.imsave(str(path), img)
    g.destroy_camera_sensor(env.sim, cam)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clearance", type=float, default=0.30,
                    help="gate ground clearance in metres (curriculum range 0.25-0.35)")
    ap.add_argument("--out-dir", type=Path,
                    default=PROJECT_ROOT / "runs" / "low_bar_view")
    args = ap.parse_args()

    env, cfg = build_low_bar_env(args.clearance)
    ox, oy, oz = [float(v) for v in env.env_origins[0]]
    clr = float(env._low_bar_clearance_for_env(0))
    bar_x, bar_y, bar_z = env._low_bar_world_pose(0)
    thick = float(cfg.terrain.low_bar_thickness)
    width = float(cfg.terrain.low_bar_width)

    print("=" * 60)
    print("ROBOCON low-bar field")
    print(f"  requested clearance : {args.clearance:.3f} m")
    print(f"  robot spawn origin  : ({ox:.2f}, {oy:.2f}, {oz:.2f})")
    print(f"  gate centre (world) : ({bar_x:.2f}, {bar_y:.2f}, {bar_z:.2f})")
    print(f"  gate: width={width:.2f} m, cross-section={thick:.2f} m")
    print(f"  bar bottom = {bar_z - thick / 2:.3f} m")
    print(f"  forward gap (spawn->gate) = {bar_x - ox:.2f} m")
    print("=" * 60)

    cm = int(round(args.clearance * 100))
    eye = (ox - 1.2, oy - 2.4, oz + 1.3)
    tgt = (ox + 1.4, oy, oz + 0.30)
    out = args.out_dir / f"low_bar_c{cm:02d}.png"
    render_png(env, out, eye, tgt)
    print("wrote", out)

    eye2 = (bar_x + 0.9, oy - 2.2, oz + 0.45)
    tgt2 = (bar_x, oy, oz + clr)
    out2 = args.out_dir / f"low_bar_c{cm:02d}_side.png"
    render_png(env, out2, eye2, tgt2)
    print("wrote", out2)


if __name__ == "__main__":
    main()
