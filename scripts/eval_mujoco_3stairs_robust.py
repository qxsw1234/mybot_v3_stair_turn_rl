#!/usr/bin/env python3
"""Batch robustness evaluation for checkpoint 059950 in MuJoCo.

The robust mode randomizes initial pose, stair friction, base mass, joint
damping, motor strength, command speed, and zero-to-two policy-step action
delay.  A trial only succeeds if the robot reaches the top, remains within
the lane, does not fall, and is still valid after two seconds of stand hold.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from sim2sim_mujoco import apply_policy_profile, load_config  # noqa: E402
from sim2sim_mujoco_3stairs import (  # noqa: E402
    DEFAULT_ADAPTATION,
    DEFAULT_BODY,
    DEFAULT_CONFIG,
    LANES,
    NUM_STEPS,
    TREAD_DEPTH,
    X_START,
    ThreeStairsSim,
)


STAND_OFFSET = np.array(
    [
        -0.05, 0.02, 0.10,
        +0.05, 0.02, 0.10,
        -0.08, 0.02, 0.11,
        +0.08, 0.02, 0.11,
    ],
    dtype=np.float32,
)


def reset_policy_memory(sim):
    sim.policy_history.zero_()
    sim.estimator_history.zero_()
    sim.last_actions.zero_()
    sim.action_lag_buffer = [
        np.zeros(12, dtype=np.float32)
        for _ in range(sim.action_lag_steps + 1)
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
    floor_id = mujoco.mj_name2id(
        sim.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    if floor_id >= 0:
        friction_ids.append(floor_id)
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
    stand_target = np.clip(
        sim.policy_dof_pos + STAND_OFFSET, sim.lower, sim.upper
    )
    sim.kp[:] = 2.0 * motion_kp
    sim.kd[:] = 2.0 * motion_kd
    joint_velocity_samples = []
    steps = max(1, int(seconds / sim.control_dt))
    for step in range(steps):
        last_target += 0.12 * (stand_target - last_target)
        sim.step_control(last_target, None)
        if step >= steps // 2:
            joint_velocity_samples.append(sim.joint_vel().copy())
    rms = float(np.sqrt(np.mean(np.square(joint_velocity_samples))))
    return last_target, rms


def failure_reason(sim, lane_y, top_height, stair_end):
    x, y, z = map(float, sim.data.qpos[:3])
    if sim.fell:
        return "fell"
    if x < stair_end - 0.15:
        return "short_x"
    if abs(y - lane_y) > 0.8:
        return "side_drift"
    if z < top_height + 0.20:
        return "low_z"
    return "success"


def run_trial(config, lane_index, params, max_move_seconds):
    sim = ThreeStairsSim(
        config,
        DEFAULT_BODY,
        DEFAULT_ADAPTATION,
        DEFAULT_CONFIG.parents[2],
        "aligned_damped",
    )
    apply_perturbations(sim, params)
    lane_y, height, _ = LANES[lane_index]
    sim.teleport(
        params["spawn_x_m"],
        lane_y + params["spawn_y_offset_m"],
        math.radians(params["spawn_yaw_deg"]),
    )
    reset_policy_memory(sim)

    motion_kp = sim.kp.copy()
    motion_kd = sim.kd.copy()
    last_target = sim.policy_dof_pos.copy()
    last_target, initial_stand_rms = stand(
        sim, last_target, motion_kp, motion_kd, 1.0
    )
    reset_policy_memory(sim)

    sim.kp[:] = motion_kp
    sim.kd[:] = motion_kd
    command = np.array(
        [params["command_vx_mps"], 0.0, 0.0], dtype=np.float32
    )
    action_scale = float(config["action_scale"])
    stair_end = X_START + NUM_STEPS * TREAD_DEPTH
    top_height = NUM_STEPS * height
    reached_top_at = None
    for step in range(max(1, int(max_move_seconds / sim.control_dt))):
        action = sim.infer(command)
        target = sim.target_from_action(action, action_scale)
        last_target = np.clip(target, sim.lower, sim.upper)
        sim.step_control(last_target, None)
        x, y, z = map(float, sim.data.qpos[:3])
        if sim.fell:
            break
        if (
            x >= stair_end + 0.10
            and z >= top_height + 0.20
            and abs(y - lane_y) <= 0.8
        ):
            reached_top_at = (step + 1) * sim.control_dt
            break

    reset_policy_memory(sim)
    if not sim.fell:
        last_target, final_stand_rms = stand(
            sim, last_target, motion_kp, motion_kd, 2.0
        )
    else:
        final_stand_rms = None

    reason = failure_reason(sim, lane_y, top_height, stair_end)
    summary = sim.summary()
    return {
        "height_m": height,
        "success": reason == "success" and reached_top_at is not None,
        "reason": reason if reached_top_at is not None else (
            reason if reason != "success" else "top_not_reached_in_motion"
        ),
        "reached_top_at_s": reached_top_at,
        "final_base_xyz": [float(v) for v in sim.data.qpos[:3]],
        "lane_y_m": lane_y,
        "lateral_error_m": abs(float(sim.data.qpos[1]) - lane_y),
        "initial_stand_joint_velocity_rms": initial_stand_rms,
        "final_stand_joint_velocity_rms": final_stand_rms,
        "max_tilt_deg": summary["max_tilt_deg"],
        "mean_abs_action": summary["mean_abs_action"],
        "action_saturation_fraction": summary["action_saturation_fraction"],
        "fell": summary["fell"],
        "parameters": params,
    }


def aggregate(records):
    result = {}
    for height in (0.08, 0.10, 0.12):
        selected = [r for r in records if abs(r["height_m"] - height) < 1e-6]
        successes = [r for r in selected if r["success"]]
        reasons = {}
        for record in selected:
            reasons[record["reason"]] = reasons.get(record["reason"], 0) + 1
        result[f"{int(height * 100)}cm"] = {
            "trials": len(selected),
            "successes": len(successes),
            "success_rate_percent": round(100.0 * len(successes) / len(selected), 1),
            "failure_reasons": reasons,
            "mean_lateral_error_m": round(
                float(np.mean([r["lateral_error_m"] for r in selected])), 3
            ),
            "mean_action_saturation_percent": round(
                100.0 * float(np.mean([
                    r["action_saturation_fraction"] for r in selected
                ])),
                1,
            ),
        }
    rates = [entry["success_rate_percent"] for entry in result.values()]
    result["overall_success_rate_percent"] = round(float(np.mean(rates)), 1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("nominal", "robust"), default="robust")
    parser.add_argument("--trials-per-height", type=int, default=30)
    parser.add_argument("--seed", type=int, default=59950)
    parser.add_argument("--max-move-seconds", type=float, default=18.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials_per_height <= 0:
        parser.error("--trials-per-height must be positive")

    config = load_config(DEFAULT_CONFIG)
    apply_policy_profile(config, "config")
    rng = np.random.default_rng(args.seed)
    records = []
    started = time.time()
    for lane_index, (_, height, _) in enumerate(LANES):
        for trial_index in range(args.trials_per_height):
            params = sample_parameters(rng, args.mode)
            record = run_trial(
                config, lane_index, params, args.max_move_seconds
            )
            record["trial"] = trial_index
            records.append(record)
            print(
                f"[{args.mode}] {height*100:.0f}cm "
                f"trial={trial_index + 1}/{args.trials_per_height} "
                f"success={record['success']} reason={record['reason']} "
                f"xyz={[round(v, 2) for v in record['final_base_xyz']]} "
                f"lag={params['action_lag_steps']}"
            )
            sys.stdout.flush()

    payload = {
        "checkpoint": "059950",
        "profile": "aligned_damped",
        "mode": args.mode,
        "seed": args.seed,
        "max_move_seconds": args.max_move_seconds,
        "stand_hold_seconds": 2.0,
        "elapsed_seconds": round(time.time() - started, 3),
        "aggregate": aggregate(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
