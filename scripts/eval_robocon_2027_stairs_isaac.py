#!/usr/bin/env python3
"""Isaac smoke evaluator for the historical 8/10/12 cm straight stairs.

Geometry matches the MuJoCo three-lane reference (10 risers, 0.30 m tread,
2.0 m width, first riser 0.55 m ahead). This is a base-gait comparison and
is not the full 2027 gully with a 360 mm gap and 20-degree ramp.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys

import isaacgym  # noqa: F401, import before torch
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
from play_teleop import load_config  # noqa: E402
from robodog_gym.envs.base.legged_robot_config import Cfg  # noqa: E402
from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv  # noqa: E402
from robodog_gym.envs.wrappers.history_wrapper import HistoryWrapper  # noqa: E402
from robodog_gym_learn.ppo_cse.actor_critic import ActorCritic  # noqa: E402

HEIGHTS = (0.08, 0.10, 0.12)
NUM_STEPS = 10
TREAD = 0.30
START = 0.55
TOP_LENGTH = 0.90


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return [round(max(0, mid-half), 4), round(min(1, mid+half), 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="Run containing parameters.pkl")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-envs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2026101615)
    ap.add_argument("--vx", type=float, default=0.40)
    ap.add_argument("--sim-device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.num_envs < 3 or args.num_envs % 3 or args.vx <= 0:
        ap.error("num-envs must be a positive multiple of three, vx positive")
    with open(os.path.join(args.run, "parameters.pkl"), "rb") as f:
        load_config(pickle.load(f)["Cfg"], Cfg)
    Cfg.env.num_envs = args.num_envs
    Cfg.env.episode_length_s = 22.0
    Cfg.commands.heading_command = False
    Cfg.commands.command_curriculum = False
    Cfg.commands.resampling_time = 24.0
    Cfg.noise.add_noise = False
    for key in dir(Cfg.domain_rand):
        if key.startswith("randomize_") or key == "push_robots":
            setattr(Cfg.domain_rand, key, False)
    Cfg.terrain.sim2sim_straight_stairs = True
    Cfg.terrain.sim2sim_bridge = False
    Cfg.terrain.bridge_replay = False
    Cfg.commands.bridge_replay_commands = False
    Cfg.terrain.robocon_beam = False
    Cfg.terrain.robocon_low_bar = False
    Cfg.terrain.sim2sim_stair_start_offset = START
    Cfg.terrain.sim2sim_stair_tread_depth = TREAD
    Cfg.terrain.sim2sim_stair_width = 2.0
    Cfg.terrain.sim2sim_stair_num_steps = NUM_STEPS
    Cfg.terrain.sim2sim_stair_top_length = TOP_LENGTH
    Cfg.terrain.sim2sim_stair_step_heights = list(HEIGHTS)
    Cfg.terrain.num_rows = 3
    Cfg.terrain.num_cols = 6
    Cfg.terrain.terrain_length = 10.0  # Keep the full 0.90 m top platform inside the tile.
    Cfg.terrain.terrain_width = 10.0
    Cfg.terrain.border_size = 2.0
    Cfg.terrain.horizontal_scale = 0.05
    Cfg.terrain.curriculum = True
    Cfg.terrain.max_init_terrain_level = 2
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Cfg.terrain.teleport_robots = False
    Cfg.terrain.x_init_range = 0.03
    Cfg.terrain.y_init_range = 0.03
    Cfg.terrain.yaw_init_range = 0.035
    Cfg.asset.self_collisions = 1
    if args.sim_device == "cpu":
        Cfg.sim.use_gpu_pipeline = False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    base = VelocityTrackingEasyEnv(sim_device=args.sim_device, headless=True,
                                   cfg=Cfg, debug_viz=False)
    base.cfg.terrain.curriculum = False
    ids = torch.arange(args.num_envs, device=base.device)
    rows = ids % 3
    base.terrain_levels[:] = rows
    base.env_origins[:] = base.terrain_origins[rows, base.terrain_types]
    base.reset_idx(ids)
    env = HistoryWrapper(base)
    state = torch.load(args.checkpoint, map_location=base.device)
    model = ActorCritic(env.num_policy_obs, env.num_estimator_obs,
                        env.num_privileged_obs, env.num_actions, Cfg.cfg_ppo).to(base.device)
    model.load_state_dict(state)
    model.eval()
    obs = env.reset()
    first_top = [-1] * args.num_envs
    stable = [0] * args.num_envs
    touched = [0] * args.num_envs
    records = [None] * args.num_envs
    top_x = START + NUM_STEPS*TREAD + 0.10
    for step in range(int(base.max_episode_length) + 2):
        commands = torch.full((args.num_envs,), args.vx, device=base.device)
        commands[torch.tensor([x >= 0 for x in first_top], device=base.device)] = 0.0
        base.commands[:, 0] = commands
        base.commands[:, 1:3] = 0.0
        with torch.inference_mode():
            latent = model.adaptation_module(obs["estimator_obs"])
            actions = model.actor_body(torch.cat((obs["policy_obs"], latent), dim=-1))
        obs, _, done, _ = env.step(actions)
        x = (base.base_pos[:, 0] - base.env_origins[:, 0]).detach().cpu().tolist()
        y = (base.base_pos[:, 1] - base.env_origins[:, 1]).detach().cpu().tolist()
        z = (base.base_pos[:, 2] - base.env_origins[:, 2]).detach().cpu().tolist()
        done_ids = set(done.nonzero(as_tuple=False).flatten().detach().cpu().tolist())
        foot_x = (base.foot_positions[:, :, 0] - base.env_origins[:, 0:1]).detach().cpu().numpy()
        foot_z = (base.foot_positions[:, :, 2] - base.env_origins[:, 2:3]).detach().cpu().numpy()
        support = (base.contact_forces[:, base.feet_indices, 2] > 1.0).detach().cpu().numpy()
        for i in range(args.num_envs):
            if records[i] is not None:
                continue
            height = HEIGHTS[i % 3]
            if i in done_ids:
                records[i] = {"env_id": i, "row": i % 3,
                              "height_m": height, "success": False,
                              "reason": "terminated_before_stable_top",
                              "first_top_s": None if first_top[i] < 0 else round(first_top[i]*base.dt, 3),
                              "touched_step_bits": touched[i]}
                continue
            for j in range(NUM_STEPS):
                left = START + j*TREAD
                right = left + TREAD
                foot_on_step = support[i] & (foot_x[i] >= left) & (foot_x[i] < right) & (foot_z[i] >= (j+1)*height - 0.03)
                if foot_on_step.any():
                    touched[i] |= 1 << j
            top = x[i] >= top_x and abs(y[i]) <= 0.8 and z[i] >= NUM_STEPS*height + 0.20
            if top and first_top[i] < 0:
                first_top[i] = step
            if first_top[i] >= 0:
                stable[i] = stable[i] + 1 if top else 0
                if stable[i] >= 100:
                    records[i] = {"env_id": i, "row": i % 3,
                                  "height_m": height, "success": True,
                                  "reason": "stable_top_2s",
                                  "first_top_s": round(first_top[i]*base.dt, 3),
                                  "touched_step_bits": touched[i]}
        if all(row is not None for row in records):
            break
    for i, row in enumerate(records):
        if row is None:
            records[i] = {"env_id": i, "row": i % 3, "height_m": HEIGHTS[i % 3],
                          "success": False, "reason": "timeout",
                          "first_top_s": None if first_top[i] < 0 else round(first_top[i]*base.dt, 3),
                          "touched_step_bits": touched[i]}
    by_height = {}
    for row, height in enumerate(HEIGHTS):
        sample = [r for r in records if r["row"] == row]
        successes = sum(r["success"] for r in sample)
        by_height[f"{int(height*100)}cm"] = {"successes": successes,
                                             "trials": len(sample),
                                             "rate": round(successes/len(sample), 4),
                                             "wilson95": wilson(successes, len(sample)),
                                             "full_step_contact_count": sum(r["touched_step_bits"] == (1 << NUM_STEPS)-1 for r in sample)}
    payload = {"kind": "historical_stairs_isaac_smoke_not_2027_gully",
               "source_run": args.run, "checkpoint": args.checkpoint,
               "checkpoint_sha256": sha256(args.checkpoint),
               "evaluator_sha256": sha256(__file__),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
               "seed": args.seed, "command_vx_mps": args.vx,
               "geometry": {"heights_m": HEIGHTS, "num_steps": NUM_STEPS,
                            "tread_depth_m": TREAD, "start_offset_m": START,
                            "top_length_m": TOP_LENGTH, "width_m": 2.0},
               "criterion": "reach top and remain there for 2 s without early termination",
               "by_height": by_height, "records": records}
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2); f.write("\n")
    print(json.dumps(by_height, indent=2), flush=True)
    print("saved:", output, flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
