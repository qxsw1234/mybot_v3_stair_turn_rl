"""CPU-physics evaluation for a Mybot V3 stair+turn checkpoint.

Zero GPU memory (CPU PhysX + CPU tensor pipeline), so it can run beside the
GPU training process. Protocol deliberately matches the `model_5321` protocol
used in legged_gym/HIMLoco so the two stacks can be compared in spirit:

    episode 8 s, command vx = 0.5 m/s (vy = yaw = 0),
    success = did not fall AND net forward distance >= 2.0 m

Terrain: ascending stairs only (proportions [0,0,1,0,...]), curriculum off so
envs are spread over all 12 difficulty rows. Stair height per row:
    height(row) = 0.05 + difficulty * (max_step_height - 0.05)
    difficulty  = row / num_rows * difficulty_scale   (= row * 0.075 here)
=> height(row) = 0.05 + 0.0135 * row  -> row 4 = 10.4 cm, row 8 = 15.8 cm

Usage:
    <robodog_gym python> eval_stairs_cpu.py --run <run dir> --iteration 5500
"""
import argparse
import isaacgym  # noqa: F401  # MUST come before torch
import torch
torch.set_grad_enabled(False)
import numpy as np
import pickle as pkl
import random
import sys
import os
import json

PROJECT = "/home/ldl/mybot_v3_stair_turn_rl"
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

from robodog_gym.envs import *  # noqa: F401,F403
from robodog_gym.envs.base.legged_robot_config import Cfg
from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv
from robodog_gym.envs.wrappers.history_wrapper import HistoryWrapper
from play_teleop import load_config  # reuse the run-config loader


def row_height(row, num_rows=12, difficulty_scale=0.90, max_step_height=0.23):
    difficulty = row / num_rows * difficulty_scale
    return 0.05 + difficulty * (max_step_height - 0.05)


def _roll_pitch(q):
    """roll, pitch from a (N,4) wxyz quaternion tensor."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    return torch.stack([roll, pitch], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--iteration", type=int, default=-1, help="-1 = ac_weights_last.pt")
    ap.add_argument("--num-envs", type=int, default=48)
    ap.add_argument("--episodes-per-env", type=int, default=3)
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--episode-s", type=float, default=8.0)
    ap.add_argument("--stress", action="store_true",
                    help="keep domain randomization on (training settings)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    run = args.run
    it_label = "last" if args.iteration == -1 else "%06d" % args.iteration

    # ---- restore the exact training config of that run ----
    with open(os.path.join(run, "parameters.pkl"), "rb") as f:
        cfg_dict = pkl.load(f)["Cfg"]
    load_config(cfg_dict, Cfg)

    np.random.seed(7); random.seed(7); torch.manual_seed(7)

    Cfg.env.num_envs = args.num_envs
    Cfg.env.episode_length_s = args.episode_s
    Cfg.commands.resampling_time = args.episode_s  # command fixed per episode
    Cfg.terrain.curriculum = False               # -> envs spread over all rows
    Cfg.terrain.max_init_terrain_level = 11
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # stairs up only
    # CPU PhysX segfaults inside add_triangle_mesh on a full-size map: the
    # training map is border 25 m + 50x8 m tiles ~= 6.5M vertices. Shrink the
    # map (fewer columns / smaller border) but keep all 12 difficulty rows so
    # the stair-height coverage (row 4 = 10.4 cm, row 8 = 15.8 cm) is intact.
    Cfg.terrain.border_size = 5.0
    Cfg.terrain.num_cols = 8
    Cfg.asset.file = '{MINI_GYM_ROOT_DIR}/resources/robots/mybot_v3/urdf/mybot_v3.urdf'
    Cfg.asset.self_collisions = 1

    # CPU physics: no GPU memory, runs beside the training process.
    # Mirror exactly what train_mybot_v3_stair_turn.py does for --sim-device cpu:
    # only use_gpu_pipeline=False; the CPU device is selected via sim_device='cpu'
    # (do NOT also flip Cfg.sim.physx.use_gpu here — that mismatch segfaults).
    Cfg.sim.use_gpu_pipeline = False

    if not args.stress:
        Cfg.domain_rand.push_robots = False
        Cfg.domain_rand.randomize_gravity = False
        Cfg.domain_rand.randomize_restitution = False
        Cfg.domain_rand.randomize_motor_offset = False
        Cfg.domain_rand.randomize_motor_strength = False
        Cfg.domain_rand.randomize_friction_indep = False
        Cfg.domain_rand.randomize_ground_friction = False
        Cfg.domain_rand.randomize_base_mass = False
        Cfg.domain_rand.randomize_Kd_factor = False
        Cfg.domain_rand.randomize_Kp_factor = False
        Cfg.domain_rand.randomize_joint_friction = False
        Cfg.domain_rand.randomize_com_displacement = False
        Cfg.domain_rand.randomize_friction = False
    Cfg.env.num_recording_envs = 1

    env = VelocityTrackingEasyEnv(sim_device='cpu', headless=True, cfg=Cfg, debug_viz=False)
    env = HistoryWrapper(env)

    # ---- policy from ac_weights (kept on CPU, no CUDA context) ----
    from robodog_gym_learn.ppo_cse.actor_critic import ActorCritic
    weights = torch.load(os.path.join(run, "checkpoints", "ac_weights_%s.pt" % it_label),
                         map_location='cpu')
    ac = ActorCritic(env.num_policy_obs, env.num_estimator_obs,
                     env.num_privileged_obs, env.num_actions, Cfg.cfg_ppo)
    ac.load_state_dict(weights)
    ac.eval()

    def policy(obs):
        latent = ac.adaptation_module(obs["estimator_obs"].to('cpu'))
        act = ac.actor_body(torch.cat((obs["policy_obs"].to('cpu'), latent), dim=-1))
        return act, latent

    n = args.num_envs
    rows = env.terrain_levels.clone().long()
    heights = torch.tensor([row_height(int(r)) for r in rows])
    _origins = getattr(env, "env_origins", None)
    env_origins_z0 = float(_origins[0, 2]) if _origins is not None else float("nan")
    # The terrain grid's world z is offset (env_origins z can be ~-1.26 m), so
    # "body height" must be measured relative to each env's own origin.
    origin_z = (_origins[:, 2].clone() if _origins is not None else torch.zeros(n))
    print("envs=%d  rows=%s" % (n, sorted(set(int(r) for r in rows))))
    print("stair heights (m) by row:", {int(r): round(row_height(int(r)), 3)
                                        for r in sorted(set(rows.tolist()))})

    total_steps = (int(args.episode_s * 50) + 5) * args.episodes_per_env
    # +5: the env auto-resets one step AFTER max_episode_length, so a bare
    # episode_s*50 steps would end just before the first episode is recorded.
    rec = []  # (row, height, fell, distance, end_z, max_rp, steps)

    obs = env.reset()
    start_x = env.base_pos[:, 0].clone()
    prev_x = start_x.clone()
    prev_z = env.base_pos[:, 2].clone()
    q = env.root_states[:, 3:7]
    prev_rp = _roll_pitch(q).abs().max(dim=1).values
    max_rp = torch.zeros(n)
    trace = []  # (t, x0, z0) for env 0, to debug the vertical reference
    for ep in range(args.episodes_per_env):
        for t in range(total_steps):
            with torch.no_grad():
                act, _ = policy(obs)
            env.commands[:, 0] = args.vx
            env.commands[:, 1] = args.vy
            env.commands[:, 2] = args.yaw
            obs, rew, done, info = env.step(act)

            x = env.base_pos[:, 0]
            z = env.base_pos[:, 2]
            rp = _roll_pitch(env.root_states[:, 3:7]).abs().max(dim=1).values
            max_rp = torch.maximum(max_rp, rp)
            ended = env.reset_buf.bool().clone()
            fell = env.early_termi_buf.bool().clone()
            if ended.any():
                for i in ended.nonzero(as_tuple=False).flatten().tolist():
                    rec.append((int(rows[i]), float(heights[i]), bool(fell[i]),
                                float(prev_x[i] - start_x[i]),
                                float(prev_z[i] - origin_z[i]),
                                float(max_rp[i]), t + 1))
                    max_rp[i] = 0.0
                start_x[ended] = x[ended]
            prev_x = x.clone()
            prev_z = z.clone()
            if t % 25 == 0:
                trace.append((t, float(x[0]), float(z[0])))

    env.close() if hasattr(env, "close") else None

    print("trace env0 (t, x, z):", [(t, round(xx, 2), round(zz, 2)) for t, xx, zz in trace[:8]],
          "...", [(t, round(xx, 2), round(zz, 2)) for t, xx, zz in trace[-4:]])
    print("env origin z (env0):", float(env_origins_z0), " heights[0]:", float(heights[0]))

    # ---- aggregate ----
    print("\ncommand: vx=%.2f vy=%.2f yaw=%.2f | episode=%.1fs | stress=%s"
          % (args.vx, args.vy, args.yaw, args.episode_s, args.stress))
    print("%-6s %-8s %-6s %-9s %-9s %-8s %-9s" %
          ("row", "height", "n", "fall%", "net_x(m)", "end_z", "max_rp"))
    by_row = {}
    for row, h, fell, dist, ez, mrp, _ in rec:
        by_row.setdefault(row, []).append((h, fell, dist, ez, mrp))
    for row in sorted(by_row):
        v = by_row[row]
        h = v[0][0]
        falls = sum(1 for _, f, _, _, _ in v if f)
        dists = [d for _, _, d, _, _ in v]
        ez = [e for _, _, _, e, _ in v]
        mrp = [m for _, _, _, _, m in v]
        succ = sum(1 for _, f, d, e, m in v if (not f) and d >= 2.0)
        upr = sum(1 for _, f, d, e, m in v if (e > 0.25 and m < 0.7))
        print("%-6d %-8.3f %-6d %-9.1f %-9.2f %-8.2f %-9.2f  upright%%=%.0f succ%%=%.0f"
              % (row, h, len(v), 100.0 * falls / len(v), float(np.mean(dists)),
                 float(np.mean(ez)), float(np.mean(mrp)), 100.0 * upr / len(v),
                 100.0 * succ / len(v)))
    allf = sum(1 for _, _, f, _, _, _, _ in rec if f)
    alld = [d for _, _, _, d, _, _, _ in rec]
    allez = [e for _, _, _, _, e, _, _ in rec]
    allmrp = [m for _, _, _, _, _, m, _ in rec]
    succ = sum(1 for _, _, f, d, _, _, _ in rec if (not f) and d >= 2.0)
    upr = sum(1 for _, _, f, d, e, m, _ in rec if (e > 0.25 and m < 0.7))
    print("%-6s %-8s %-6d %-9.1f %-9.2f %-8.2f %-9.2f  upright%%=%.0f succ%%=%.0f"
          % ("ALL", "-", len(rec), 100.0 * allf / len(rec), float(np.mean(alld)),
             float(np.mean(allez)), float(np.mean(allmrp)),
             100.0 * upr / len(rec), 100.0 * succ / len(rec)))
    print("* succ% = 未摔倒 且 8 s 内净前进 >= 2.0 m（与 model_5321 协议一致）")
    print("* upright% = 结束时相对机身高度>0.25 m 且全程最大|roll/pitch|<0.7 rad")

    out = args.out or "/tmp/eval_%s_%s.json" % (os.path.basename(run), it_label)
    json.dump({
        "run": run, "iteration": it_label, "command": [args.vx, args.vy, args.yaw],
        "episode_s": args.episode_s, "stress": args.stress, "num_envs": n,
        "records": rec,
    }, open(out, "w"))
    print("saved:", out)


if __name__ == "__main__":
    main()
