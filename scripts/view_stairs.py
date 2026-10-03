#!/usr/bin/env python3
"""Interactive Isaac Gym viewer for a fixed Mybot V3 stair difficulty.

This is intentionally separate from ``play_teleop.py``.  That older viewer
overrides the terrain to discrete obstacles, while this script creates the
ascending-stair terrain used by the stair/turn training task.

Examples (run from the project root):

    python -u scripts/view_stairs.py --iteration 37500 --level 4
    python -u scripts/view_stairs.py --iteration 39250 --level 8
    python -u scripts/view_stairs.py --iteration 37500 --level 11
"""

import argparse
import os
import pickle as pkl
import random
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (str(PROJECT_ROOT), str(SCRIPTS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import isaacgym  # noqa: F401,E402  # MUST come before torch
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaacgym import gymtorch  # noqa: E402

from robodog_gym.envs import *  # noqa: F401,F403,E402
from robodog_gym.envs.base.legged_robot_config import Cfg  # noqa: E402
from robodog_gym.envs.robodog.velocity_tracking import (  # noqa: E402
    VelocityTrackingEasyEnv,
)
from robodog_gym.envs.wrappers.history_wrapper import HistoryWrapper  # noqa: E402
from play_teleop import (  # noqa: E402
    RobotController,
    convert_weights_to_jit,
    load_config,
    load_policy,
)


DEFAULT_RUN = (
    PROJECT_ROOT
    / "runs/mybot_v3_stair_turn_velocity_curriculum_resume_006000"
    / "2026-09-26_08-56-53.624919"
)


def stair_height(level: int) -> float:
    """Match the stair-height formula used by eval_stairs_cpu.py."""
    return 0.05 + (level / 12.0 * 0.90) * (0.23 - 0.05)


def disable_domain_randomization():
    for name in (
        "push_robots",
        "randomize_gravity",
        "randomize_restitution",
        "randomize_motor_offset",
        "randomize_motor_strength",
        "randomize_friction",
        "randomize_friction_indep",
        "randomize_ground_friction",
        "randomize_base_mass",
        "randomize_com_displacement",
        "randomize_Kd_factor",
        "randomize_Kp_factor",
        "randomize_joint_friction",
    ):
        if hasattr(Cfg.domain_rand, name):
            setattr(Cfg.domain_rand, name, False)


def make_environment(run: Path, level: int):
    with open(run / "parameters.pkl", "rb") as file:
        cfg_dict = pkl.load(file)["Cfg"]
    load_config(cfg_dict, Cfg)

    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The interactive Isaac Gym viewer needs the "
            "GPU environment; use eval_stairs_cpu.py for headless statistics."
        )

    # Robot and evaluation settings.
    Cfg.asset.file = "{MINI_GYM_ROOT_DIR}/resources/robots/mybot_v3/urdf/mybot_v3.urdf"
    Cfg.asset.self_collisions = 1
    Cfg.env.num_envs = 1
    Cfg.env.num_recording_envs = 0
    Cfg.env.record_video = False
    Cfg.env.episode_length_s = 1000000
    Cfg.commands.resampling_time = 1000000
    # Training has heading_command=True, where the environment overwrites
    # commands[:, 2] from a random heading target every step.  In this
    # interactive viewer Q/E should directly control yaw velocity instead.
    # The policy still receives the same three command channels and was
    # trained on the resulting yaw-velocity channel.
    Cfg.commands.heading_command = False
    Cfg.commands.command_curriculum = False
    disable_domain_randomization()

    # One terrain type, one fixed difficulty row, and a compact 12-level map.
    # Terrain type index 2 is ascending stairs.  Keep curriculum=True here so
    # LeggedRobot honors min/max_init_terrain_level instead of replacing them
    # with the full 0..11 range during origin assignment.
    Cfg.terrain.curriculum = True
    Cfg.terrain.selected = False
    Cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Cfg.terrain.min_init_terrain_level = level
    Cfg.terrain.max_init_terrain_level = level
    Cfg.terrain.border_size = 5.0
    Cfg.terrain.terrain_length = 8.0
    Cfg.terrain.terrain_width = 8.0
    Cfg.terrain.num_rows = 12
    Cfg.terrain.num_cols = 3
    Cfg.terrain.difficulty_scale = 0.90
    Cfg.terrain.max_step_height = 0.23
    Cfg.terrain.max_platform_height = 0.23
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.center_robots = False
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.terrain.yaw_init_range = 0.0

    # Keep the same GPU Isaac Gym path as the training viewer.
    Cfg.sim.use_gpu_pipeline = True
    base_env = VelocityTrackingEasyEnv(
        sim_device="cuda:0", headless=False, cfg=Cfg, debug_viz=True
    )

    # Keep the viewer on this fixed level even if the robot falls and resets.
    base_env._update_terrain_curriculum = lambda env_ids: None

    # The terrain origin is far from world (0, 0) because each difficulty row
    # occupies an 8 m tile.  The default camera therefore looks at empty space.
    origin = base_env.env_origins[0].detach().cpu().numpy()
    base_env.set_camera(
        [float(origin[0] - 4.0), float(origin[1] - 4.0), float(origin[2] + 3.0)],
        [float(origin[0]), float(origin[1]), float(origin[2] + 0.3)],
    )
    print(
        "Viewer camera target: "
        f"({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})"
    )
    return HistoryWrapper(base_env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--iteration", type=int, default=37500)
    parser.add_argument("--level", type=int, default=8, choices=range(12))
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    if not (run / "parameters.pkl").is_file():
        raise FileNotFoundError(f"parameters.pkl not found under {run}")
    if args.iteration < 0:
        raise ValueError("--iteration must be a saved checkpoint number")
    checkpoint = run / "checkpoints" / f"ac_weights_{args.iteration:06d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    env = make_environment(run, args.level)
    # Generate the indexed JIT files expected by the common policy loader.
    convert_weights_to_jit(env, str(run), args.iteration, Cfg)
    policy = load_policy(str(run), args.iteration)
    controller = RobotController()

    listener = None
    try:
        from pynput import keyboard

        listener = keyboard.Listener(
            on_press=controller.on_press,
            on_release=controller.on_release,
        )
        listener.start()
    except Exception as exc:
        print(f"Keyboard input unavailable: {exc}")

    height = stair_height(args.level)
    print(
        f"Viewing checkpoint {args.iteration} on ascending stairs: "
        f"level={args.level}, step_height={height:.3f} m"
    )
    print("W/S forward, A/D lateral, Q/E yaw, arrow keys apply pushes, Ctrl+C exits")

    obs = env.reset()
    dt = float(env.dt)
    try:
        while True:
            start = time.time()
            controller.update_control_commands()
            env.commands[:, 0] = controller.x_vel_cmd
            env.commands[:, 1] = controller.y_vel_cmd
            env.commands[:, 2] = controller.yaw_vel_cmd
            with torch.no_grad():
                actions, _ = policy(obs)
            obs, _, _, _ = env.step(actions)

            if controller.push_robot:
                env.root_states[:, 7] = controller.x_vel_push
                env.root_states[:, 8] = controller.y_vel_push
                env.gym.set_actor_root_state_tensor(
                    env.sim, gymtorch.unwrap_tensor(env.root_states)
                )

            elapsed = time.time() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        if listener is not None:
            listener.stop()
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
