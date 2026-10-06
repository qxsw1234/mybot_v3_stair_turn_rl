#!/usr/bin/env python3
"""M1 acceptance harness: batch-evaluate a checkpoint on RoboCon obstacles.

Obstacles and success criteria follow rulebook V2.0 (see M0_REPORT_20261005.md):

  t_stairs  T字形台阶 (up +x over the platform, down -x):
      success = no fall, platform reached, all 7 route surfaces touched by a
      foot (left steps L1-L3, platform, right steps R1-R3), |y| stays on the
      obstacle (<=0.85 m), finishes past the far flight on the ground.
  slope     大斜坡 (up 3 m @ 11.3 deg, plateau, down 3 m):
      success = no fall, walked >=1.0 m along the 3 m direction on EACH face,
      |y| <= 1.0 m (stays on the 2 m wide structure), finishes past the ramp.

Modes: nominal (fixed params) or robust (randomized spawn/friction/mass/
damping/motor strength/action lag/command, ranges identical to the 3-stairs
robust instrument).

Usage:
    python scripts/eval_mujoco_obstacles.py --obstacle both --trials 30 \
        --output-dir evaluations/m1_obstacles
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import mujoco  # noqa: E402

from sim2sim_mujoco import apply_policy_profile, load_config  # noqa: E402
from sim2sim_mujoco_obstacles import (  # noqa: E402
    DEFAULT_ADAPTATION,
    DEFAULT_BODY,
    DEFAULT_CONFIG,
    ObstacleSim,
    SLOPE,
    T_STAIRS,
)

STAND_OFFSET = np.array(
    [-0.05, 0.02, 0.10, +0.05, 0.02, 0.10, -0.08, 0.02, 0.11, +0.08, 0.02, 0.11],
    dtype=np.float32,
)

T_STAIRS_REQUIRED = {
    "tstair_left_1", "tstair_left_2", "tstair_left_3",
    "tstair_platform",
    "tstair_right_1", "tstair_right_2", "tstair_right_3",
}


def reset_policy_memory(sim):
    sim.policy_history.zero_()
    sim.estimator_history.zero_()
    sim.last_actions.zero_()
    sim.action_lag_buffer = [
        np.zeros(12, dtype=np.float32) for _ in range(sim.action_lag_steps + 1)
    ]


def sample_parameters(rng, mode):
    if mode == "nominal":
        return {
            "spawn_x_m": 0.0,
            "spawn_y_offset_m": 0.0,
            "spawn_yaw_deg": 0.0,
            "friction": 1.0,
            "base_mass_scale": 1.0,
            "joint_damping_scale": 1.0,
            "motor_strength_scale": 1.0,
            "action_lag_steps": 0,
            "command_vx_mps": 0.4,
        }
    return {
        "spawn_x_m": float(rng.uniform(-0.06, 0.06)),
        "spawn_y_offset_m": float(rng.uniform(-0.12, 0.12)),
        "spawn_yaw_deg": float(rng.uniform(-4.0, 4.0)),
        "friction": float(rng.uniform(0.70, 1.20)),
        "base_mass_scale": float(rng.uniform(0.90, 1.10)),
        "joint_damping_scale": float(rng.uniform(0.80, 1.20)),
        "motor_strength_scale": float(rng.uniform(0.90, 1.10)),
        "action_lag_steps": int(rng.integers(0, 3)),
        "command_vx_mps": float(rng.uniform(0.35, 0.45)),
    }


def apply_perturbations(sim, params):
    friction_ids = list(sim.stair_geom_ids)
    floor_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        friction_ids.append(int(floor_id))
    for geom_id in friction_ids:
        sim.model.geom_friction[geom_id, 0] = params["friction"]

    base_id = sim.base_body_id
    sim.model.body_mass[base_id] *= params["base_mass_scale"]
    sim.model.body_inertia[base_id] *= params["base_mass_scale"]
    for dof_index in sim.joint_qvel_idx:
        sim.model.dof_damping[dof_index] *= params["joint_damping_scale"]

    strength = params["motor_strength_scale"]
    sim.kp *= strength
    sim.kd *= strength
    sim.torque_limits *= strength
    sim.action_lag_steps = params["action_lag_steps"]
    reset_policy_memory(sim)


def stand(sim, last_target, motion_kp, motion_kd, seconds):
    stand_target = np.clip(sim.policy_dof_pos + STAND_OFFSET, sim.lower, sim.upper)
    sim.kp[:] = 2.0 * motion_kp
    sim.kd[:] = 2.0 * motion_kd
    steps = max(1, int(seconds / sim.control_dt))
    for _ in range(steps):
        last_target += 0.12 * (stand_target - last_target)
        sim.step_control(last_target, None)
    return last_target


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def run_trial(obstacle, params, body_path, adaptation_path, max_move_seconds,
              heading_gain=0.0, lateral_gain=0.0):
    config = load_config(DEFAULT_CONFIG)
    apply_policy_profile(config, "config")
    sim = ObstacleSim(
        config, body_path, adaptation_path, DEFAULT_CONFIG.parents[2],
        "aligned_damped", obstacle,
    )
    apply_perturbations(sim, params)
    spawn_x, spawn_y = ObstacleSim.OBSTACLE_SPAWNS[obstacle]
    sim.teleport(
        spawn_x + params["spawn_x_m"],
        spawn_y + params["spawn_y_offset_m"],
        math.radians(params["spawn_yaw_deg"]),
    )
    reset_policy_memory(sim)

    motion_kp = sim.kp.copy()
    motion_kd = sim.kd.copy()
    last_target = sim.policy_dof_pos.copy()
    stand(sim, last_target, motion_kp, motion_kd, 1.0)
    reset_policy_memory(sim)

    sim.kp[:] = motion_kp
    sim.kd[:] = motion_kd
    command = np.array([params["command_vx_mps"], 0.0, 0.0], dtype=np.float32)
    action_scale = float(config["action_scale"])
    corridor_y = spawn_y  # keep the robot on the obstacle's centre line
    max_abs_yaw_err = 0.0

    touched = set()
    max_abs_y = 0.0
    region_max_abs_y = 0.0
    floor_interior_steps = 0
    platform_reached = False
    finished_at = None
    up_net = 0.0
    down_net = 0.0
    up_entry_x = None
    up_exit_x = None
    down_entry_x = None
    down_exit_x = None
    trajectory = []

    def in_region(px):
        if obstacle == "t_stairs":
            return -1.55 <= px <= 1.55
        return 0.15 <= px <= 6.85

    n_max = max(1, int(max_move_seconds / sim.control_dt))
    steps_done = 0
    x, y, z = (float(v) for v in sim.data.qpos[:3])
    for step in range(n_max):
        steps_done = step + 1
        if heading_gain > 0.0 or lateral_gain > 0.0:
            px, py, pz = (float(v) for v in sim.data.qpos[:3])
            w, qx, qy, qz = sim.data.qpos[3:7]
            yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            yaw_err = wrap_to_pi(yaw)
            max_abs_yaw_err = max(max_abs_yaw_err, abs(yaw_err))
            yaw_cmd = float(np.clip(-heading_gain * yaw_err, -0.8, 0.8))
            vy_cmd = float(np.clip(-lateral_gain * (py - corridor_y), -0.30, 0.30))
            command = np.array([params["command_vx_mps"], vy_cmd, yaw_cmd], dtype=np.float32)
        action = sim.infer(command)
        target = sim.target_from_action(action, action_scale)
        last_target = np.clip(target, sim.lower, sim.upper)
        sim.step_control(last_target, None)
        x, y, z = (float(v) for v in sim.data.qpos[:3])
        hit, floor_hit = sim.contact_labels()
        touched |= hit
        dy = abs(y - spawn_y)
        max_abs_y = max(max_abs_y, dy)
        if in_region(x):
            region_max_abs_y = max(region_max_abs_y, dy)
        if step % 25 == 0:
            trajectory.append([round(steps_done * sim.control_dt, 2), round(x, 3),
                               round(y, 3), round(z, 3)])

        if obstacle == "t_stairs":
            if -0.35 <= x <= 0.35 and z > 0.60:
                platform_reached = True
            if -1.35 <= x <= 1.45 and floor_hit:
                floor_interior_steps += 1
            if sim.fell:
                break
            if x >= 1.55 and z <= 0.55 and dy <= 0.9:
                finished_at = steps_done * sim.control_dt
                break
            if x >= 2.60:
                break
        else:  # slope
            if up_entry_x is None and x > 0.15:
                up_entry_x = x
            if up_exit_x is None and x >= 3.0:
                up_exit_x = x
            if down_entry_x is None and x >= 3.8:
                down_entry_x = x
            if down_exit_x is None and x >= 6.8:
                down_exit_x = x
            if 0.30 < x < 6.50 and floor_hit:
                floor_interior_steps += 1
            if sim.fell:
                break
            if x >= 6.85 and dy <= 1.1:
                finished_at = steps_done * sim.control_dt
                break
            if x >= 7.5:
                break

    crossed = finished_at is not None
    if obstacle == "slope":
        if up_entry_x is not None:
            up_net = (up_exit_x if up_exit_x is not None else min(x, 3.0)) - up_entry_x
        if down_entry_x is not None:
            down_net = (down_exit_x if down_exit_x is not None else max(x, 3.8)) - down_entry_x
        up_net = max(0.0, up_net)
        down_net = max(0.0, down_net)

    # post-motion stand check (rules: robot must be stable after crossing)
    stood_ok = None
    if not sim.fell:
        reset_policy_memory(sim)
        stand(sim, last_target, motion_kp, motion_kd, 1.0)
        stood_ok = not sim.fell

    x, y, z = (float(v) for v in sim.data.qpos[:3])
    if sim.fell:
        reason = "fell"
    elif region_max_abs_y > (0.85 if obstacle == "t_stairs" else 1.0):
        reason = "side_drift"
    elif obstacle == "t_stairs":
        if not platform_reached:
            reason = "no_platform"
        elif not crossed:
            reason = "stuck_on_stairs"
        elif not T_STAIRS_REQUIRED.issubset(touched):
            reason = "missing_touches:" + ",".join(sorted(T_STAIRS_REQUIRED - touched))
        elif not stood_ok:
            reason = "unstable_after_cross"
        else:
            reason = "success"
    else:
        if up_net < 1.0:
            reason = "short_up"
        elif down_net < 1.0:
            reason = "short_down"
        elif not crossed:
            reason = "timeout_before_finish"
        elif not stood_ok:
            reason = "unstable_after_cross"
        else:
            reason = "success"

    summary = sim.summary()
    return {
        "success": reason == "success",
        "reason": reason,
        "finished_at_s": finished_at,
        "final_base_xyz": [x, y, z],
        "max_abs_y_m": max_abs_y,
        "region_max_abs_y_m": region_max_abs_y,
        "platform_reached": platform_reached if obstacle == "t_stairs" else None,
        "up_net_m": round(up_net, 3) if obstacle == "slope" else None,
        "down_net_m": round(down_net, 3) if obstacle == "slope" else None,
        "touched": sorted(touched),
        "floor_interior_steps": floor_interior_steps,
        "max_tilt_deg": summary["max_tilt_deg"],
        "mean_abs_action": summary["mean_abs_action"],
        "action_saturation_fraction": summary["action_saturation_fraction"],
        "fell": summary["fell"],
        "max_abs_yaw_err_deg": round(math.degrees(max_abs_yaw_err), 1),
        "trajectory": trajectory,
        "parameters": params,
    }


def aggregate(obstacle, records):
    successes = [r for r in records if r["success"]]
    reasons = {}
    for record in records:
        key = record["reason"].split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    entry = {
        "trials": len(records),
        "successes": len(successes),
        "success_rate_percent": round(100.0 * len(successes) / max(1, len(records)), 1),
        "failure_reasons": reasons,
        "mean_max_abs_y_m": round(float(np.mean([r["max_abs_y_m"] for r in records])), 3),
        "mean_region_max_abs_y_m": round(
            float(np.mean([r["region_max_abs_y_m"] for r in records])), 3
        ),
        "mean_finish_time_s": (
            round(float(np.mean([r["finished_at_s"] for r in records if r["finished_at_s"]])), 2)
            if any(r["finished_at_s"] for r in records) else None
        ),
        "mean_action_saturation_percent": round(
            100.0 * float(np.mean([r["action_saturation_fraction"] for r in records])), 1
        ),
        "fell_count": sum(1 for r in records if r["fell"]),
    }
    if obstacle == "t_stairs":
        entry["mean_floor_interior_steps"] = round(
            float(np.mean([r["floor_interior_steps"] for r in records])), 1
        )
        entry["platform_reach_count"] = sum(1 for r in records if r["platform_reached"])
    else:
        entry["mean_up_net_m"] = round(float(np.mean([r["up_net_m"] for r in records])), 3)
        entry["mean_down_net_m"] = round(float(np.mean([r["down_net_m"] for r in records])), 3)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obstacle", choices=("t_stairs", "slope", "both"), default="both")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--mode", choices=("nominal", "robust"), default="robust")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--body", type=Path, default=DEFAULT_BODY)
    parser.add_argument("--adaptation", type=Path, default=DEFAULT_ADAPTATION)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--heading-gain", type=float, default=0.0,
                        help="external heading stabilizer gain (yaw command = -k*yaw_err; 0=off)")
    parser.add_argument("--lateral-gain", type=float, default=0.0,
                        help="external lateral stabilizer gain (vy command = -k*(y-y_ref); 0=off)")
    parser.add_argument("--max-move-seconds-stairs", type=float, default=20.0)
    parser.add_argument("--max-move-seconds-slope", type=float, default=50.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluations/m1_obstacles")
    args = parser.parse_args()

    obstacles = ("t_stairs", "slope") if args.obstacle == "both" else (args.obstacle,)
    body_path = args.body.expanduser().resolve()
    adaptation_path = args.adaptation.expanduser().resolve()

    rng = np.random.default_rng(args.seed)
    tasks = []
    for obstacle in obstacles:
        max_move = (
            args.max_move_seconds_stairs if obstacle == "t_stairs"
            else args.max_move_seconds_slope
        )
        for trial_index in range(args.trials):
            params = sample_parameters(rng, args.mode)
            tasks.append((obstacle, trial_index, params, str(body_path), str(adaptation_path),
                          max_move, args.heading_gain, args.lateral_gain))

    started = time.time()
    payload = {
        "checkpoint": body_path.name,
        "profile": "aligned_damped",
        "mode": args.mode,
        "seed": args.seed,
        "obstacles": {},
    }

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_trial, *task)
                for task in tasks
            ]
            results = [f.result() for f in futures]
    else:
        results = [run_trial(*task) for task in tasks]

    index = 0
    for obstacle in obstacles:
        records = []
        for _ in range(args.trials):
            task_obstacle, trial_index, params, *_ = tasks[index]
            record = results[index]
            record["trial"] = trial_index
            records.append(record)
            print(
                f"[{obstacle}] trial={trial_index + 1}/{args.trials} "
                f"success={record['success']} reason={record['reason']} "
                f"xyz={[round(v, 2) for v in record['final_base_xyz']]} "
                f"max|y|={record['max_abs_y_m']:.2f}"
            )
            sys.stdout.flush()
            index += 1
        payload["obstacles"][obstacle] = {
            "aggregate": aggregate(obstacle, records),
            "records": records,
        }

    payload["elapsed_seconds"] = round(time.time() - started, 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.mode}_seed{args.seed}_{body_path.stem}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    for obstacle in obstacles:
        print(json.dumps({obstacle: payload["obstacles"][obstacle]["aggregate"]},
                         ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
