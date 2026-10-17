#!/usr/bin/env python3
"""Held-out Isaac rollout for the finite bridge course; one episode per env.

This is stage evidence for the heightfield training geometry, not a rule-size
box-bridge or physical-robot result. The terminal flags are captured before the
environment resets. Run each checkpoint in a fresh process and verify initial
poses before calling the outcomes paired.
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

import isaacgym  # noqa: F401, must precede torch
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from play_teleop import load_config  # noqa: E402
from robodog_gym.envs.base.legged_robot_config import Cfg  # noqa: E402
from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv  # noqa: E402
from robodog_gym.envs.wrappers.history_wrapper import HistoryWrapper  # noqa: E402
from robodog_gym_learn.ppo_cse.actor_critic import ActorCritic  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(count, total):
    z = 1.959963984540054
    p = count / total
    d = 1 + z*z/total
    c = (p + z*z/(2*total))/d
    r = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/d
    return [round(max(0, c-r), 4), round(min(1, c+r), 4)]


class BridgeEvalEnv(VelocityTrackingEasyEnv):
    def reset_idx(self, env_ids):
        self.terminal_snapshot = {}
        if len(env_ids) and hasattr(self, "beam_success_ever"):
            x = self.base_pos[env_ids, 0] - self.env_origins[env_ids, 0]
            y = self.base_pos[env_ids, 1] - self.env_origins[env_ids, 1]
            self.terminal_snapshot = {
                "ids": env_ids.detach().cpu().tolist(),
                "success": self.beam_success_ever[env_ids].detach().cpu().tolist(),
                "outside": self.beam_outside_ever[env_ids].detach().cpu().tolist(),
                "entered": self.beam_entered_ever[env_ids].detach().cpu().tolist(),
                "timeout": self.time_out_buf[env_ids].detach().cpu().tolist(),
                "early": self.early_termi_buf[env_ids].detach().cpu().tolist(),
                "base_x_m": x.detach().cpu().tolist(),
                "base_y_m": y.detach().cpu().tolist(),
                "steps": self.episode_length_buf[env_ids].detach().cpu().tolist(),
            }
        return super().reset_idx(env_ids)


def load_model(env, path):
    model = ActorCritic(env.num_policy_obs, env.num_estimator_obs,
                        env.num_privileged_obs, env.num_actions, Cfg.cfg_ppo).to(env.device)
    state = torch.load(path, map_location=env.device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate(env, model, seed, speed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    obs = env.reset()
    base = env.env
    n = base.num_envs
    initial_root = base.root_states[:n, :7].detach().cpu().clone()
    initial_dof = base.dof_pos[:n].detach().cpu().clone()
    initial_root[:, :3] -= base.env_origins[:n].detach().cpu()
    records = [None] * n
    for _ in range(int(base.max_episode_length) + 5):
        base.commands[:, 0] = speed
        base.commands[:, 1:3] = 0.0
        with torch.inference_mode():
            latent = model.adaptation_module(obs["estimator_obs"])
            action = model.actor_body(torch.cat((obs["policy_obs"], latent), dim=-1))
        obs, _, done, _ = env.step(action)
        snapshot = base.terminal_snapshot
        ended = done.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        if ended:
            if snapshot.get("ids") != ended:
                raise RuntimeError("terminal snapshot mismatch")
            for j, env_id in enumerate(ended):
                if records[env_id] is not None:
                    continue
                records[env_id] = {
                    "env_id": env_id, "seed": seed,
                    "initial_base_xyz_quat": initial_root[env_id].tolist(),
                    "initial_dof_pos": initial_dof[env_id].tolist(),
                    **{key: snapshot[key][j] for key in
                       ("success", "outside", "entered", "timeout", "early",
                        "base_x_m", "base_y_m", "steps")},
                }
        if all(item is not None for item in records):
            return records
    raise RuntimeError("not all bridge episodes terminated")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="Bridge run containing parameters.pkl")
    ap.add_argument("--candidate", default=None, help="Candidate ac_weights checkpoint")
    ap.add_argument("--baseline", default=None, help="Baseline ac_weights checkpoint")
    ap.add_argument("--num-envs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026101606)
    ap.add_argument("--sim-device", default="cuda:0")
    ap.add_argument("--profile", choices=("nominal", "pose"), default="pose")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.num_envs < 1:
        ap.error("num-envs must be positive")
    if bool(args.baseline) == bool(args.candidate):
        ap.error("provide exactly one of --baseline or --candidate; run each in a fresh process")
    with open(os.path.join(args.run, "parameters.pkl"), "rb") as f:
        load_config(pickle.load(f)["Cfg"], Cfg)
    if not Cfg.terrain.robocon_beam or not Cfg.terrain.sim2sim_bridge:
        raise RuntimeError("run is not the finite bridge course")
    source_has_replay = bool(getattr(Cfg.terrain, "bridge_replay", False))
    speed = (float(Cfg.commands.bridge_replay_vx) if source_has_replay
             else float(Cfg.commands.lin_vel_x[0]))
    # Evaluate all robots on the same bridge geometry, even when training
    # mixed bridge and replay tiles. Keep replay commands out of the rollout.
    Cfg.terrain.bridge_replay = False
    Cfg.terrain.num_cols = 8
    Cfg.commands.bridge_replay_commands = False
    Cfg.commands.lin_vel_x = [speed, speed]
    Cfg.commands.limit_vel_x = [speed, speed]
    Cfg.env.num_envs = args.num_envs
    Cfg.terrain.curriculum = False
    Cfg.commands.heading_command = False
    Cfg.commands.command_curriculum = False
    Cfg.commands.resampling_time = Cfg.env.episode_length_s + 2
    Cfg.noise.add_noise = False
    Cfg.terrain.height_measurements_per_step_xy_noise_std = 0.0
    Cfg.terrain.height_measurements_per_step_z_noise_std = 0.0
    Cfg.terrain.height_measurements_per_env_xy_noise_std = 0.0
    Cfg.terrain.height_measurements_per_env_z_noise_std = 0.0
    Cfg.terrain.height_measurements_per_env_noise_prob = 0.0
    if args.profile == "nominal":
        Cfg.terrain.x_init_range = 0.0
        Cfg.terrain.y_init_range = 0.0
        Cfg.terrain.yaw_init_range = 0.0
        Cfg.init_state.dof_pos_scale_range = [1.0, 1.0]
        Cfg.init_state.base_velocity_half_range = 0.0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.sim_device == "cpu":
        Cfg.sim.use_gpu_pipeline = False
    base = BridgeEvalEnv(sim_device=args.sim_device, headless=True, cfg=Cfg, debug_viz=False)
    env = HistoryWrapper(base)
    output = {
        "kind": "beam_stage_isaac_heightfield_heldout",
        "geometry_m": {key: float(getattr(Cfg.terrain, key)) for key in
                       ("bridge_width", "bridge_length", "bridge_height", "bridge_start_offset")},
        "profile": args.profile, "seed": args.seed, "num_envs": args.num_envs,
        "command_vx_mps": speed, "sim_device": args.sim_device,
        "source_run_has_replay": source_has_replay,
        "evaluation_bridge_only": True,
        "run": args.run,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "evaluator_sha256": sha256(__file__),
        "candidates": {},
    }
    selected = ("baseline", args.baseline) if args.baseline else ("candidate", args.candidate)
    for name, path in (selected,):
        model = load_model(env, path)
        records = evaluate(env, model, args.seed, speed)
        count = sum(bool(item["success"]) for item in records)
        output["candidates"][name] = {
            "checkpoint": path, "checkpoint_sha256": sha256(path),
            "successes": count, "success_rate": round(count/len(records), 4),
            "wilson95": wilson(count, len(records)),
            "outside_count": sum(bool(item["outside"]) for item in records),
            "timeout_count": sum(bool(item["timeout"]) for item in records),
            "records": records,
        }
        print(f"{name}: {count}/{len(records)} success", flush=True)
    output["pairing_note"] = (
        "Compare this file with a fresh-process run of the other checkpoint; "
        "verify recorded initial_base_xyz_quat and initial_dof_pos before pairing."
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("saved:", args.out, flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)  # Isaac Gym/PhysX may segfault during interpreter teardown.


if __name__ == "__main__":
    main()
