"""CPU-physics evaluation for a Mybot V3 stair+turn checkpoint.

Zero GPU memory (CPU PhysX + CPU tensor pipeline), so it can run beside the
GPU training process. Protocol deliberately matches the `model_5321` protocol
used in legged_gym/HIMLoco so the two stacks can be compared in spirit:

    episode 8 s, command vx = 0.5 m/s (vy = yaw = 0),
    success = did not fall AND net forward distance >= 2.0 m AND the
              expected stair elevation change was achieved

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
    """Roll and pitch from Isaac Gym's (N,4) xyzw quaternion tensor."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin((2 * (w * y - z * x)).clamp(-1.0, 1.0))
    return torch.stack([roll, pitch], dim=1)


def _yaw(q):
    """Yaw from Isaac Gym's (N,4) xyzw quaternion tensor."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


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
    ap.add_argument("--terrain-cols", type=int, default=8)
    ap.add_argument("--border-size", type=float, default=2.0)
    ap.add_argument("--terrain-type", type=int, default=2,
                    help="0=smooth slope 1=rough slope 2=stairs up 3=stairs down "
                         "4=discrete 5=stepping stones 7=smooth flat 8=rough flat")
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
    # Otherwise the environment overwrites commands[:, 2] from a sampled
    # heading after every physics step, invalidating a requested yaw command.
    Cfg.commands.heading_command = False
    # Build deterministic curriculum terrain so the terrain row really maps
    # to the documented stair height. Dynamic promotion is disabled after the
    # terrain map has been constructed.
    Cfg.terrain.curriculum = True
    Cfg.terrain.max_init_terrain_level = 11
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.terrain_proportions = [0.0] * 9
    Cfg.terrain.terrain_proportions[args.terrain_type] = 1.0
    # CPU PhysX segfaults inside add_triangle_mesh on a full-size map: the
    # training map is border 25 m + 50x8 m tiles ~= 6.5M vertices. Shrink the
    # map (fewer columns / smaller border) but keep all 12 difficulty rows so
    # the stair-height coverage (row 4 = 10.4 cm, row 8 = 15.8 cm) is intact.
    Cfg.terrain.border_size = args.border_size
    Cfg.terrain.num_cols = args.terrain_cols
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

    base_env = VelocityTrackingEasyEnv(sim_device='cpu', headless=True, cfg=Cfg, debug_viz=False)
    base_env.cfg.terrain.curriculum = False
    all_env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    deterministic_rows = all_env_ids % int(Cfg.terrain.num_rows)
    base_env.terrain_levels[:] = deterministic_rows
    base_env.env_origins[:] = base_env.terrain_origins[
        deterministic_rows, base_env.terrain_types]
    base_env.reset_idx(all_env_ids)
    env = HistoryWrapper(base_env)

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
    print("envs=%d  rows=%s" % (n, sorted(set(int(r) for r in rows))))
    print("stair heights (m) by row:", {int(r): round(row_height(int(r)), 3)
                                        for r in sorted(set(rows.tolist()))})

    obs = env.reset()
    env.commands[:, 0] = args.vx
    env.commands[:, 1] = args.vy
    env.commands[:, 2] = args.yaw
    start_x = env.base_pos[:, 0].clone()
    start_y = env.base_pos[:, 1].clone()
    start_z = env.base_pos[:, 2].clone()
    prev_x = start_x.clone()
    prev_y = start_y.clone()
    prev_z = env.base_pos[:, 2].clone()
    prev_yaw = _yaw(env.root_states[:, 3:7])
    accumulated_yaw = torch.zeros(n)
    max_rp = torch.zeros(n)
    completed = torch.zeros(n, dtype=torch.long)
    episode_start_step = torch.zeros(n, dtype=torch.long)
    # (row, height, fell, forward_distance, elevation_gain, max_rp, steps,
    #  accumulated_yaw, planar_drift)
    rec = []
    trace = []  # (t, x0, z0) for env 0, to debug the vertical reference
    episode_steps = int(args.episode_s * 50)
    max_total_steps = (episode_steps + 5) * (args.episodes_per_env + 1)
    global_step = 0
    while bool((completed < args.episodes_per_env).any().item()) and global_step < max_total_steps:
        # Commands must be installed before policy inference. With heading mode
        # disabled, the environment will preserve the requested yaw command.
        env.commands[:, 0] = args.vx
        env.commands[:, 1] = args.vy
        env.commands[:, 2] = args.yaw
        with torch.no_grad():
            act, _ = policy(obs)
        obs, rew, done, info = env.step(act)

        x = env.base_pos[:, 0]
        y = env.base_pos[:, 1]
        z = env.base_pos[:, 2]
        rp = _roll_pitch(env.root_states[:, 3:7]).abs().max(dim=1).values
        max_rp = torch.maximum(max_rp, rp)
        ended = done.bool().clone()
        current_yaw = _yaw(env.root_states[:, 3:7])
        yaw_delta = torch.atan2(
            torch.sin(current_yaw - prev_yaw), torch.cos(current_yaw - prev_yaw))
        # Ended environments have already been reset by env.step; do not count
        # the reset pose as part of the completed episode's rotation.
        yaw_delta[ended] = 0.0
        accumulated_yaw += yaw_delta
        fell = env.early_termi_buf.bool().clone()
        if ended.any():
            for i in ended.nonzero(as_tuple=False).flatten().tolist():
                if completed[i] < args.episodes_per_env:
                    rec.append((int(rows[i]), float(heights[i]), bool(fell[i]),
                                float(prev_x[i] - start_x[i]),
                                float(prev_z[i] - start_z[i]),
                                float(max_rp[i]),
                                int(global_step + 1 - episode_start_step[i]),
                                float(accumulated_yaw[i]),
                                float(torch.sqrt(
                                    (prev_x[i] - start_x[i]).pow(2) +
                                    (prev_y[i] - start_y[i]).pow(2)))))
                    completed[i] += 1
                max_rp[i] = 0.0
                accumulated_yaw[i] = 0.0
                episode_start_step[i] = global_step + 1
            # env.step has already reset ended environments, so x/z are the
            # correct origins for their next episodes.
            start_x[ended] = x[ended]
            start_y[ended] = y[ended]
            start_z[ended] = z[ended]
        prev_x = x.clone()
        prev_y = y.clone()
        prev_z = z.clone()
        prev_yaw = current_yaw.clone()
        if global_step % 25 == 0:
            trace.append((global_step, float(x[0]), float(z[0])))
        global_step += 1

    expected_records = n * args.episodes_per_env
    if len(rec) != expected_records:
        raise RuntimeError(
            "evaluation ended with %d/%d episodes; completed range=%d..%d" %
            (len(rec), expected_records, int(completed.min()), int(completed.max())))

    env.close() if hasattr(env, "close") else None

    print("trace env0 (t, x, z):", [(t, round(xx, 2), round(zz, 2)) for t, xx, zz in trace[:8]],
          "...", [(t, round(xx, 2), round(zz, 2)) for t, xx, zz in trace[-4:]])

    # ---- aggregate ----
    min_elevation = 0.06
    turn_test = abs(args.yaw) >= 0.05 and abs(args.vx) < 0.05 and abs(args.vy) < 0.05
    expected_turn = abs(args.yaw) * args.episode_s

    def succeeded(fell, distance, elevation, yaw_travel, planar_drift):
        if turn_test:
            signed_turn = yaw_travel if args.yaw > 0 else -yaw_travel
            return (not fell) and signed_turn >= 0.70 * expected_turn and planar_drift <= 0.75
        elevation_ok = ((args.terrain_type != 2 or elevation >= min_elevation) and
                        (args.terrain_type != 3 or elevation <= -min_elevation))
        return (not fell) and distance >= 2.0 and elevation_ok

    print("\ncommand: vx=%.2f vy=%.2f yaw=%.2f | episode=%.1fs | stress=%s"
          % (args.vx, args.vy, args.yaw, args.episode_s, args.stress))
    print("%-6s %-8s %-6s %-9s %-9s %-8s %-9s %-9s %-9s" %
          ("row", "height", "n", "fall%", "net_x(m)", "delta_z", "max_rp",
           "yaw(rad)", "drift(m)"))
    by_row = {}
    for row, h, fell, dist, dz, mrp, _, yaw_travel, drift in rec:
        by_row.setdefault(row, []).append((h, fell, dist, dz, mrp, yaw_travel, drift))
    for row in sorted(by_row):
        v = by_row[row]
        h = v[0][0]
        falls = sum(1 for _, f, _, _, _, _, _ in v if f)
        dists = [d for _, _, d, _, _, _, _ in v]
        dz = [e for _, _, _, e, _, _, _ in v]
        mrp = [m for _, _, _, _, m, _, _ in v]
        yaw_travel = [yy for _, _, _, _, _, yy, _ in v]
        drift = [dd for _, _, _, _, _, _, dd in v]
        succ = sum(1 for _, f, d, e, _, yy, dd in v if succeeded(f, d, e, yy, dd))
        upr = sum(1 for _, f, _, _, m, _, _ in v if (not f) and m < 0.7)
        print("%-6d %-8.3f %-6d %-9.1f %-9.2f %-8.2f %-9.2f %-9.2f %-9.2f  upright%%=%.0f succ%%=%.0f"
              % (row, h, len(v), 100.0 * falls / len(v), float(np.mean(dists)),
                 float(np.mean(dz)), float(np.mean(mrp)), float(np.mean(yaw_travel)),
                 float(np.mean(drift)), 100.0 * upr / len(v), 100.0 * succ / len(v)))
    allf = sum(1 for _, _, f, _, _, _, _, _, _ in rec if f)
    alld = [d for _, _, _, d, _, _, _, _, _ in rec]
    alldz = [e for _, _, _, _, e, _, _, _, _ in rec]
    allmrp = [m for _, _, _, _, _, m, _, _, _ in rec]
    allyaw = [yy for _, _, _, _, _, _, _, yy, _ in rec]
    alldrift = [dd for _, _, _, _, _, _, _, _, dd in rec]
    succ = sum(1 for _, _, f, d, e, _, _, yy, dd in rec
               if succeeded(f, d, e, yy, dd))
    upr = sum(1 for _, _, f, _, _, m, _, _, _ in rec if (not f) and m < 0.7)
    print("%-6s %-8s %-6d %-9.1f %-9.2f %-8.2f %-9.2f %-9.2f %-9.2f  upright%%=%.0f succ%%=%.0f"
          % ("ALL", "-", len(rec), 100.0 * allf / len(rec), float(np.mean(alld)),
             float(np.mean(alldz)), float(np.mean(allmrp)), float(np.mean(allyaw)),
             float(np.mean(alldrift)),
             100.0 * upr / len(rec), 100.0 * succ / len(rec)))
    if turn_test:
        print("* succ% = 未摔倒、累计转角达到指令的70%，且8秒内平移漂移不超过0.75 m")
    else:
        print("* succ% = 未摔倒、净前进 >= 2.0 m，并达到楼梯方向所需的 0.06 m 高度变化")
    print("* upright% = 未摔倒且全程记录到的最大|roll/pitch|<0.7 rad")

    out = args.out or "/tmp/eval_%s_%s.json" % (os.path.basename(run), it_label)
    with open(out, "w") as out_file:
        json.dump({
            "run": run, "iteration": it_label, "command": [args.vx, args.vy, args.yaw],
            "episode_s": args.episode_s, "stress": args.stress, "num_envs": n,
            "terrain_type": args.terrain_type, "success_min_elevation": min_elevation,
            "records": rec,
        }, out_file)
    print("saved:", out)
    # Isaac Gym's CPU backend can double-free graphics state during interpreter
    # teardown even in headless mode. All results are flushed at this point.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
