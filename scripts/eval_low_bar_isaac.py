#!/usr/bin/env python3
"""Deterministic Isaac Gym evaluation for ROBOCON low-bar checkpoints.

Success requires the whole robot to cross the gate, no contact force on the
low-bar actor, no early termination, and one second of stable motion after the
crossing.  Multiple checkpoints share one simulator instance so a training run
can be ranked quickly and consistently.
"""

import argparse
import json
import os
import pickle as pkl
import random
import sys

import isaacgym  # noqa: F401  # must precede torch
from isaacgym import gymapi, gymtorch
import numpy as np
import torch


PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

from play_teleop import load_config  # noqa: E402
from robodog_gym.envs.base.legged_robot_config import Cfg  # noqa: E402
from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv  # noqa: E402
from robodog_gym.envs.wrappers.history_wrapper import HistoryWrapper  # noqa: E402
from robodog_gym_learn.ppo_cse.actor_critic import ActorCritic  # noqa: E402
from robodog_gym_learn.ppo_cse.checkpoint_utils import (  # noqa: E402
    migrate_observation_expansion,
)


def _disable_domain_randomization(cfg):
    for name in (
        "push_robots",
        "randomize_gravity",
        "randomize_restitution",
        "randomize_motor_offset",
        "randomize_motor_strength",
        "randomize_friction_indep",
        "randomize_ground_friction",
        "randomize_base_mass",
        "randomize_Kd_factor",
        "randomize_Kp_factor",
        "randomize_joint_friction",
        "randomize_joint_damping",
        "randomize_joint_armature",
        "randomize_com_displacement",
        "randomize_friction",
        "randomize_lag_timesteps",
    ):
        if hasattr(cfg.domain_rand, name):
            setattr(cfg.domain_rand, name, False)


def _roll_pitch(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    return roll, pitch


def _checkpoint_path(run, iteration):
    name = "ac_weights_last.pt" if iteration < 0 else f"ac_weights_{iteration:06d}.pt"
    path = os.path.join(run, "checkpoints", name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _load_policy(env, checkpoint_path):
    model = ActorCritic(
        env.num_policy_obs,
        env.num_estimator_obs,
        env.num_privileged_obs,
        env.num_actions,
        Cfg.cfg_ppo,
    ).to(env.device)
    source = torch.load(checkpoint_path, map_location=env.device)
    target = model.state_dict()
    if all(key in source and source[key].shape == value.shape
           for key, value in target.items()):
        model.load_state_dict(source)
        migration = []
    else:
        migrated, migration = migrate_observation_expansion(
            model,
            source,
            policy_history_length=env.policy_obs_history_length,
            estimator_history_length=env.estimator_obs_history_length,
        )
        model.load_state_dict(migrated)
    model.eval()
    return model, migration


def _evaluate(env, model, args):
    base_env = env.env
    device = base_env.device
    num_envs = base_env.num_envs
    env_ids = torch.arange(num_envs, device=device)

    # Reinstall deterministic terrain rows before every candidate.
    rows = env_ids % int(Cfg.terrain.num_rows)
    base_env.terrain_levels[:] = rows
    base_env.env_origins[:] = base_env.terrain_origins[rows, base_env.terrain_types]
    obs = env.reset()

    # Query simulator-domain indices instead of relying on actor creation
    # order. Isaac Gym warns when actors are appended to earlier envs, and a
    # wrong contiguous slice can silently report robot contacts as bar hits.
    all_contact_forces = gymtorch.wrap_tensor(
        base_env.gym.acquire_net_contact_force_tensor(base_env.sim))
    bar_body_indices = torch.as_tensor([
        base_env.gym.get_actor_rigid_body_index(
            base_env.envs[i], base_env.low_bar_actor_handles[i], 0,
            gymapi.DOMAIN_SIM)
        for i in range(num_envs)
    ], device=device, dtype=torch.long)
    robot_base_indices = [
        base_env.gym.get_actor_rigid_body_index(
            base_env.envs[i], base_env.actor_handles[i], 0,
            gymapi.DOMAIN_SIM)
        for i in range(num_envs)
    ]
    expected_robot_bases = [i * base_env.num_bodies for i in range(num_envs)]
    if robot_base_indices != expected_robot_bases:
        raise RuntimeError(
            "Robot rigid bodies are not contiguous in the order assumed by "
            "LeggedRobot.rigid_body_state")
    expected_bar_start = num_envs * base_env.num_bodies
    expected_bar_indices = torch.arange(
        expected_bar_start, expected_bar_start + num_envs,
        device=device, dtype=torch.long)
    print(
        "bar rigid-body indices:",
        f"actual=[{int(bar_body_indices[0])}..{int(bar_body_indices[-1])}]",
        f"expected=[{expected_bar_start}..{expected_bar_start + num_envs - 1}]",
        f"contiguous={bool(torch.equal(bar_body_indices, expected_bar_indices))}",
        flush=True,
    )

    bar_x = base_env.env_origins[:, 0] + float(Cfg.terrain.low_bar_x)
    clearance_table = torch.as_tensor(
        Cfg.terrain.robocon_low_bar_clearance_by_level,
        device=device,
        dtype=torch.float32,
    )
    clearances = clearance_table[rows]

    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros_like(active)
    collision = torch.zeros_like(active)
    passed = torch.zeros_like(active)
    crossed_gate_plane = torch.zeros_like(active)
    crossed_outside_gate = torch.zeros_like(active)
    recovered = torch.zeros_like(active)
    stable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    max_bar_force = torch.zeros(num_envs, device=device)
    min_base_height = torch.full((num_envs,), float("inf"), device=device)
    max_abs_roll_pitch = torch.zeros(num_envs, device=device)
    start_x = base_env.base_pos[:, 0].clone()

    steps = int(round(args.duration * 50.0))
    recovery_steps = int(round(args.recovery_s * 50.0))
    for _ in range(steps):
        base_env.commands[:, 0] = args.vx
        base_env.commands[:, 1] = 0.0
        base_env.commands[:, 2] = 0.0
        with torch.inference_mode():
            latent = model.adaptation_module(obs["estimator_obs"])
            action = model.actor_body(torch.cat((obs["policy_obs"], latent), dim=-1))
        obs, _, done, _ = env.step(action)

        current_force = torch.linalg.vector_norm(
            all_contact_forces.index_select(0, bar_body_indices), dim=1)
        max_bar_force = torch.maximum(max_bar_force, current_force)
        collision |= active & (current_force > args.contact_force_threshold)

        # rigid_body_state is deliberately the contiguous robot-only slice.
        robot_bodies = base_env.rigid_body_state.view(
            num_envs, base_env.num_bodies, 13)
        whole_robot_x = robot_bodies[:, :, 0].amin(dim=1)
        body_lateral_extent = (
            robot_bodies[:, :, 1] - base_env.env_origins[:, 1].unsqueeze(1)
        ).abs().amax(dim=1)
        opening_half_width = (
            0.5 * float(Cfg.terrain.low_bar_width)
            - float(Cfg.terrain.low_bar_thickness)
            - args.gate_margin
        )
        crossed_plane = active & (whole_robot_x > bar_x + args.pass_margin)
        first_crossing = crossed_plane & ~crossed_gate_plane
        inside_gate = body_lateral_extent < opening_half_width
        crossed_outside_gate |= first_crossing & ~inside_gate
        just_passed = first_crossing & inside_gate
        crossed_gate_plane |= first_crossing
        passed |= just_passed

        roll, pitch = _roll_pitch(base_env.base_quat)
        abs_rp = torch.maximum(roll.abs(), pitch.abs())
        max_abs_roll_pitch = torch.maximum(max_abs_roll_pitch, abs_rp)
        min_base_height = torch.minimum(min_base_height, base_env.base_pos[:, 2])
        stable = (
            active
            & passed
            & (abs_rp < args.stable_roll_pitch)
            & (base_env.base_pos[:, 2] > args.stable_base_height)
        )
        stable_steps = torch.where(stable, stable_steps + 1, torch.zeros_like(stable_steps))
        recovered |= stable_steps >= recovery_steps

        early = done.bool() & base_env.early_termi_buf.bool()
        fell |= active & early
        active &= ~done.bool()

    success = passed & recovered & ~collision & ~fell
    distance = base_env.base_pos[:, 0] - start_x

    records = []
    for i in range(num_envs):
        records.append({
            "env": i,
            "row": int(rows[i].item()),
            "clearance_m": float(clearances[i].item()),
            "success": bool(success[i].item()),
            "passed": bool(passed[i].item()),
            "recovered": bool(recovered[i].item()),
            "crossed_outside_gate": bool(crossed_outside_gate[i].item()),
            "collision": bool(collision[i].item()),
            "fell": bool(fell[i].item()),
            "max_bar_force_n": float(max_bar_force[i].item()),
            "min_base_height_m": float(min_base_height[i].item()),
            "max_abs_roll_pitch_rad": float(max_abs_roll_pitch[i].item()),
            "final_displacement_x_m": float(distance[i].item()),
        })

    def rate(field, subset):
        return 100.0 * sum(1 for item in subset if item[field]) / max(1, len(subset))

    by_row = {}
    for row in sorted(set(int(x) for x in rows.tolist())):
        subset = [item for item in records if item["row"] == row]
        by_row[str(row)] = {
            "clearance_m": round(float(subset[0]["clearance_m"]), 4),
            "episodes": len(subset),
            "success_rate_percent": round(rate("success", subset), 1),
            "pass_rate_percent": round(rate("passed", subset), 1),
            "outside_gate_rate_percent": round(
                rate("crossed_outside_gate", subset), 1),
            "collision_rate_percent": round(rate("collision", subset), 1),
            "fall_rate_percent": round(rate("fell", subset), 1),
        }

    summary = {
        "episodes": len(records),
        "success_rate_percent": round(rate("success", records), 1),
        "pass_rate_percent": round(rate("passed", records), 1),
        "recovery_rate_percent": round(rate("recovered", records), 1),
        "outside_gate_rate_percent": round(rate("crossed_outside_gate", records), 1),
        "collision_rate_percent": round(rate("collision", records), 1),
        "fall_rate_percent": round(rate("fell", records), 1),
        "mean_max_bar_force_n": round(float(np.mean(
            [item["max_bar_force_n"] for item in records])), 3),
        "mean_min_base_height_m": round(float(np.mean(
            [item["min_base_height_m"] for item in records])), 4),
        "by_row": by_row,
    }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Low-bar run containing parameters.pkl")
    parser.add_argument("--iterations", type=int, nargs="+", default=[-1])
    parser.add_argument("--baseline-run")
    parser.add_argument("--baseline-iteration", type=int, default=59950)
    parser.add_argument("--num-envs", type=int, default=48)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--recovery-s", type=float, default=1.0)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--contact-force-threshold", type=float, default=1.0)
    parser.add_argument("--pass-margin", type=float, default=0.05)
    parser.add_argument("--gate-margin", type=float, default=0.02)
    parser.add_argument("--stable-roll-pitch", type=float, default=0.6)
    parser.add_argument("--stable-base-height", type=float, default=0.16)
    parser.add_argument("--sim-device", choices=["cuda:0", "cpu"], default="cuda:0")
    parser.add_argument("--out")
    args = parser.parse_args()

    with open(os.path.join(args.run, "parameters.pkl"), "rb") as handle:
        cfg_dict = pkl.load(handle)["Cfg"]
    load_config(cfg_dict, Cfg)
    np.random.seed(7)
    random.seed(7)
    torch.manual_seed(7)

    Cfg.env.num_envs = args.num_envs
    Cfg.env.episode_length_s = args.duration + 2.0
    Cfg.commands.resampling_time = args.duration + 2.0
    Cfg.commands.heading_command = False
    Cfg.commands.command_curriculum = False
    Cfg.terrain.curriculum = True
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.max_init_terrain_level = int(Cfg.terrain.num_rows) - 1
    Cfg.terrain.num_cols = max(12, args.num_envs // int(Cfg.terrain.num_rows))
    Cfg.terrain.border_size = 2.0
    Cfg.terrain.x_init_range = 0.0
    Cfg.terrain.y_init_range = 0.0
    Cfg.terrain.yaw_init_range = 0.0
    Cfg.noise.add_noise = False
    Cfg.asset.self_collisions = 1
    _disable_domain_randomization(Cfg)
    if args.sim_device == "cpu":
        Cfg.sim.use_gpu_pipeline = False

    base_env = VelocityTrackingEasyEnv(
        sim_device=args.sim_device, headless=True, cfg=Cfg, debug_viz=False)
    base_env.cfg.terrain.curriculum = False
    env = HistoryWrapper(base_env)

    candidates = []
    if args.baseline_run:
        candidates.append((
            f"baseline_{args.baseline_iteration:06d}",
            _checkpoint_path(args.baseline_run, args.baseline_iteration),
        ))
    for iteration in args.iterations:
        label = "last" if iteration < 0 else f"{iteration:06d}"
        candidates.append((label, _checkpoint_path(args.run, iteration)))

    result = {
        "protocol": {
            "num_envs": args.num_envs,
            "duration_s": args.duration,
            "vx_mps": args.vx,
            "recovery_s": args.recovery_s,
            "contact_force_threshold_n": args.contact_force_threshold,
            "pass_margin_m": args.pass_margin,
            "gate_margin_m": args.gate_margin,
            "seed": 7,
        },
        "candidates": {},
    }
    for label, checkpoint in candidates:
        print(f"\n=== evaluating {label}: {checkpoint} ===", flush=True)
        model, migration = _load_policy(env, checkpoint)
        summary, records = _evaluate(env, model, args)
        summary["checkpoint"] = checkpoint
        summary["checkpoint_migration"] = migration
        summary["records"] = records
        result["candidates"][label] = summary
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("records", "by_row")},
                         ensure_ascii=False, indent=2), flush=True)

    out = args.out or os.path.join(args.run, "eval_low_bar_isaac.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
