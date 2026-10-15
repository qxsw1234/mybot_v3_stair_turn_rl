#!/usr/bin/env python3
"""Reproducible Isaac Gym evaluation for ROBOCON low-bar checkpoints.

Success requires the whole robot to cross the gate, no contact force on the
low-bar actor, no early termination, and one second of stable motion after the
crossing.  Multiple checkpoints share one simulator instance so a training run
can be ranked quickly and consistently. ``nominal`` removes pose/sensor/dynamics
randomization, while ``train-matched`` preserves the run's saved randomization
contract and uses a fixed held-out seed for fair checkpoint comparisons.

Every candidate also runs the same held-out seeds, so per-episode outcomes are
paired and two candidates can be compared with a paired delta, its 95% interval
and an exact McNemar test instead of two independent rates that cannot resolve a
two-percentage-point difference.  Reported rates carry Wilson 95% intervals and
are split into the 0.30 m competition clearance and the full 0.25-0.35 m range.
The output records the git commit, the evaluator hash and the run's dirty state.

Examples::

    # Quick screening during training (one seed, few hundred episodes).
    python -u scripts/eval_low_bar_isaac.py --run RUN --iterations 60900 60910 \
        --num-envs 384 --profile train-matched --seed 20261017 --out /tmp/screen.json

    # Formal candidate gate: 2 held-out seeds, 1000+ episodes per candidate.
    python -u scripts/eval_low_bar_isaac.py --run RUN --iterations 60900 60910 \
        --baseline-run BASE_RUN --baseline-iteration 60898 \
        --num-envs 500 --profile train-matched \
        --seeds 20261017 20261018 --out /tmp/gate.json
"""

import argparse
import hashlib
import json
import math
import os
import pickle as pkl
import random
import subprocess
import sys
from fractions import Fraction

import isaacgym  # noqa: F401  # must precede torch
from isaacgym import gymapi, gymtorch
import numpy as np
import torch


PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

from play_teleop import load_config  # noqa: E402
from robodog_gym.envs.base.legged_robot_config import Cfg  # noqa: E402
from robodog_gym.envs.robodog.velocity_tracking import VelocityTrackingEasyEnv  # noqa: E402
from robodog_gym.envs.wrappers.history_wrapper import HistoryWrapper  # noqa: E402
from robodog_gym_learn.ppo_cse.actor_critic import ActorCritic  # noqa: E402
from robodog_gym_learn.ppo_cse.checkpoint_utils import (  # noqa: E402
    migrate_observation_expansion,
)


DOMAIN_RANDOMIZATION_FLAGS = (
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
)
HEIGHT_NOISE_FLAGS = (
    "height_measurements_per_step_xy_noise_std",
    "height_measurements_per_step_z_noise_std",
    "height_measurements_per_env_xy_noise_std",
    "height_measurements_per_env_z_noise_std",
    "height_measurements_per_env_noise_prob",
)


def _disable_domain_randomization(cfg):
    for name in DOMAIN_RANDOMIZATION_FLAGS:
        if hasattr(cfg.domain_rand, name):
            setattr(cfg.domain_rand, name, False)


def _set_joint_passive_damping(props, dof_names, values):
    """Set Isaac DOF passive damping; this does not change the PD controller's Kd."""
    for index, name in enumerate(dof_names):
        lower_name = str(name).lower()
        kind = next((kind for kind in ("hip", "thigh", "calf")
                     if kind in lower_name), None)
        if kind is None:
            raise ValueError(f"Unknown low-frame DOF kind: {name}")
        props["damping"][index] = values[kind]
    return props


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


WILSON_Z = 1.959963984540054  # two-sided 95%

RATE_FIELDS = (
    ("success", "success_rate_percent"),
    ("passed", "pass_rate_percent"),
    ("recovered", "recovery_rate_percent"),
    ("crossed_outside_gate", "outside_gate_rate_percent"),
    ("collision", "collision_rate_percent"),
    ("violent_collision", "violent_collision_rate_percent"),
    ("envelope_violation", "envelope_violation_rate_percent"),
    ("fell", "fall_rate_percent"),
)


def _wilson_interval(successes, total, z=WILSON_Z):
    """Wilson score interval for a binomial proportion, returned in percent."""
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    spread = z * math.sqrt(
        phat * (1.0 - phat) / total + z * z / (4.0 * total * total))
    low = (centre - spread) / denominator
    high = (centre + spread) / denominator
    return [round(100.0 * min(1.0, max(0.0, low)), 1),
            round(100.0 * min(1.0, max(0.0, high)), 1)]


def _summarize(records):
    """Metric block with explicit counts and Wilson 95% intervals.

    A rate without an interval is not usable for Go/No-Go decisions: at a few
    hundred episodes the binomial half-width is several percentage points, so
    every reported rate carries its own ``*_ci95_percent`` companion.
    """
    total = len(records)
    block = {"episodes": total}
    for field, key in RATE_FIELDS:
        count = sum(1 for item in records if item[field])
        block[key] = round(100.0 * count / max(1, total), 1)
        block[key.replace("_percent", "_ci95_percent")] = _wilson_interval(
            count, total)
        if field == "success":
            block["success_count"] = count
    block["mean_max_bar_force_n"] = round(float(np.mean(
        [item["max_bar_force_n"] for item in records])) if records else 0.0, 3)
    block["mean_min_base_height_m"] = round(float(np.mean(
        [item["min_base_height_m"] for item in records])) if records else 0.0, 4)
    block["mean_max_gate_lateral_extent_m"] = round(float(np.mean(
        [item["max_gate_lateral_extent_m"] for item in records]))
        if records else 0.0, 4)
    return block


def _by_row(records):
    by_row = {}
    for row in sorted(set(int(item["row"]) for item in records)):
        subset = [item for item in records if item["row"] == row]
        block = _summarize(subset)
        block["clearance_m"] = round(float(subset[0]["clearance_m"]), 4)
        by_row[str(row)] = block
    return by_row


def _grouped(records, target_clearance):
    """Competition-spec clearance versus the full 0.25-0.35 m range.

    The plan requires separate reporting for the 0.30 m competition clearance
    and for the randomized range; the two halves are included because a mean
    over the range can hide an asymmetry between easy and hard rows.
    """
    buckets = {}
    for item in records:
        clearance = float(item["clearance_m"])
        if abs(clearance - target_clearance) < 1e-4:
            buckets.setdefault("competition_spec", []).append(item)
        elif clearance > target_clearance:
            buckets.setdefault("easier_than_spec", []).append(item)
        else:
            buckets.setdefault("harder_than_spec", []).append(item)
    groups = {"full_range": _summarize(records)}
    for name in ("competition_spec", "easier_than_spec", "harder_than_spec"):
        groups[name] = _summarize(buckets.get(name, []))
    return groups


def _mcnemar_exact_p_value(reference_only, candidate_only):
    """Two-sided exact McNemar test over the discordant pairs."""
    discordant = reference_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k)
               for k in range(0, min(reference_only, candidate_only) + 1))
    p_value = 2.0 * float(Fraction(tail, 2 ** discordant))
    return round(min(1.0, p_value), 6)


def _paired_compare(reference, candidate):
    """Paired success comparison for candidates sharing seeds and env indices.

    Every candidate is re-seeded identically before each episode, so env ``i``
    under seed ``s`` faces the same terrain row, spawn offset, heading and
    sensor noise.  Pairing on ``(seed, env)`` removes the between-episode
    variance that dominates an unpaired comparison at a few hundred episodes.
    """
    reference_by_key = {(item["seed"], item["env"]): item for item in reference}
    candidate_by_key = {(item["seed"], item["env"]): item for item in candidate}
    keys = sorted(set(reference_by_key) & set(candidate_by_key))
    both_success = reference_only_success = candidate_only_success = 0
    both_fail = 0
    for key in keys:
        reference_success = bool(reference_by_key[key]["success"])
        candidate_success = bool(candidate_by_key[key]["success"])
        if reference_success and candidate_success:
            both_success += 1
        elif reference_success:
            reference_only_success += 1
        elif candidate_success:
            candidate_only_success += 1
        else:
            both_fail += 1
    n_pairs = len(keys)
    delta = 100.0 * (candidate_only_success - reference_only_success) / max(1, n_pairs)
    variance = (reference_only_success + candidate_only_success
                - (candidate_only_success - reference_only_success) ** 2
                / max(1, n_pairs)) / max(1, n_pairs) ** 2
    half_width = WILSON_Z * math.sqrt(max(0.0, variance)) * 100.0
    return {
        "n_pairs": n_pairs,
        "both_success": both_success,
        "reference_only_success": reference_only_success,
        "candidate_only_success": candidate_only_success,
        "both_fail": both_fail,
        "success_delta_percent": round(delta, 2),
        "success_delta_ci95_percent": [round(delta - half_width, 2),
                                       round(delta + half_width, 2)],
        "mcnemar_exact_p": _mcnemar_exact_p_value(
            reference_only_success, candidate_only_success),
    }


def _git_state():
    """Commit hash and dirty flag so every result names the code that made it."""
    def run(*command):
        try:
            completed = subprocess.run(
                command, cwd=PROJECT, capture_output=True, text=True,
                timeout=20, check=False)
            return completed.stdout.strip()
        except Exception:
            return ""

    commit = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return {
        "git_commit": commit or None,
        "git_dirty": bool(dirty) if commit else None,
        "git_dirty_files": dirty.splitlines()[:20],
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _evaluate(env, model, args, seed, traces=None):
    base_env = env.env
    device = base_env.device
    num_envs = base_env.num_envs
    env_ids = torch.arange(num_envs, device=device)

    # Give every candidate the same held-out random poses/noise sequence.
    # Re-seeding per held-out seed keeps the pairing valid: env i under seed s
    # faces the same spawn, terrain row and sensor noise for every candidate.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Reinstall balanced terrain rows before every candidate.
    rows = env_ids % int(Cfg.terrain.num_rows)
    base_env.terrain_levels[:] = rows
    base_env.env_origins[:] = base_env.terrain_origins[rows, base_env.terrain_types]
    obs = env.reset()
    initial_trace_count = min(int(getattr(args, "trace_envs", 0) or 0), num_envs)
    initial_foot_normal_force = (
        base_env.contact_forces[:initial_trace_count]
        .index_select(1, base_env.feet_indices)[:, :, 2]
        .cpu().numpy().astype(np.float32).copy()
        if traces is not None and initial_trace_count > 0 else None)
    initial_base_quat = (
        base_env.base_quat[:initial_trace_count].cpu().numpy()
        .astype(np.float32).copy()
        if traces is not None and initial_trace_count > 0 else None)
    initial_base_z = (
        base_env.base_pos[:initial_trace_count, 2].cpu().numpy()
        .astype(np.float32).copy()
        if traces is not None and initial_trace_count > 0 else None)
    initial_foot_center_z = (
        base_env.rigid_body_state[:initial_trace_count]
        .index_select(1, base_env.feet_indices)[:, :, 2]
        .cpu().numpy().astype(np.float32).copy()
        if traces is not None and initial_trace_count > 0 else None)
    initial_base_lin_vel = (
        base_env.root_states[:initial_trace_count, 7:10]
        .cpu().numpy().astype(np.float32).copy()
        if traces is not None and initial_trace_count > 0 else None)
    initial_base_ang_vel = (
        base_env.root_states[:initial_trace_count, 10:13]
        .cpu().numpy().astype(np.float32).copy()
        if traces is not None and initial_trace_count > 0 else None)
    initial_joint_pos = (
        base_env.dof_pos[:initial_trace_count]
        .cpu().numpy().astype(np.float32).copy()
        if traces is not None and initial_trace_count > 0 else None)
    initial_joint_vel = (
        base_env.dof_vel[:initial_trace_count]
        .cpu().numpy().astype(np.float32).copy()
        if traces is not None and initial_trace_count > 0 else None)
    initial_policy_obs = (
        obs["policy_obs"][:1].detach().cpu().numpy().copy()
        if traces is not None else None)
    initial_estimator_obs = (
        obs["estimator_obs"][:1].detach().cpu().numpy().copy()
        if traces is not None else None)

    # Query simulator-domain indices and assert the supported in-order layout:
    # [robot0 bodies, bar0, robot1 bodies, bar1, ...].  Isaac Gym requires all
    # actors in an environment before creating the next environment; the old
    # two-pass actor creation violated that contract and yielded false passes.
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
    bodies_per_env = base_env.num_bodies + 1
    expected_robot_bases = [i * bodies_per_env for i in range(num_envs)]
    if robot_base_indices != expected_robot_bases:
        raise RuntimeError(
            "Unexpected interleaved robot rigid-body layout")
    expected_bar_indices = (
        torch.arange(num_envs, device=device, dtype=torch.long)
        * bodies_per_env + base_env.num_bodies)
    print(
        "bar rigid-body indices:",
        f"actual={bar_body_indices[:min(4, num_envs)].tolist()}...",
        f"expected={expected_bar_indices[:min(4, num_envs)].tolist()}...",
        f"layout_ok={bool(torch.equal(bar_body_indices, expected_bar_indices))}",
        flush=True,
    )
    if not torch.equal(bar_body_indices, expected_bar_indices):
        raise RuntimeError("Unexpected interleaved low-bar rigid-body layout")

    bar_x = base_env.env_origins[:, 0] + float(Cfg.terrain.low_bar_x)
    clearance_table = torch.as_tensor(
        Cfg.terrain.robocon_low_bar_clearance_by_level,
        device=device,
        dtype=torch.float32,
    )
    clearances = clearance_table[rows]
    bar_body_state = base_env.all_rigid_body_state.index_select(
        0, bar_body_indices)
    expected_bar_z = clearances + 0.5 * float(Cfg.terrain.low_bar_thickness)
    bar_pose_error = torch.stack([
        bar_body_state[:, 0] - bar_x,
        bar_body_state[:, 1] - base_env.env_origins[:, 1],
        bar_body_state[:, 2] - expected_bar_z,
    ], dim=1)
    print(
        "bar pose audit:",
        f"actual_xyz={bar_body_state[:min(4, num_envs), :3].tolist()}",
        f"expected_xz={torch.stack((bar_x, expected_bar_z), dim=1)[:min(4, num_envs)].tolist()}",
        f"max_abs_error_m={float(bar_pose_error.abs().max()):.6f}",
        flush=True,
    )

    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    fell = torch.zeros_like(active)
    collision = torch.zeros_like(active)
    violent_collision = torch.zeros_like(active)
    envelope_violation = torch.zeros_like(active)
    passed = torch.zeros_like(active)
    crossed_gate_plane = torch.zeros_like(active)
    crossed_outside_gate = torch.zeros_like(active)
    recovered = torch.zeros_like(active)
    success_latched = torch.zeros_like(active)
    stable_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
    max_bar_force = torch.zeros(num_envs, device=device)
    min_base_height = torch.full((num_envs,), float("inf"), device=device)
    max_abs_roll_pitch = torch.zeros(num_envs, device=device)
    max_gate_lateral_extent = torch.zeros(num_envs, device=device)
    start_x = base_env.base_pos[:, 0].clone()

    # Diagnostic only: superimpose a mesh-verified symmetric crouch residual
    # on the existing locomotion policy.  This separates action/control
    # feasibility from PPO exploration.  A scale of zero is the normal
    # deployment evaluator and remains the default.
    crouch_residual = torch.zeros(
        (1, base_env.num_actions), dtype=torch.float32, device=device)
    if args.scripted_crouch_scale > 0.0:
        action_scale = float(Cfg.control.action_scale)
        defaults = Cfg.init_state.default_joint_angles
        for index, name in enumerate(base_env.dof_names):
            lname = str(name).lower()
            if 'hip' in lname:
                target = 0.10 if lname.startswith(('fl', 'rl')) else -0.10
            elif 'thigh' in lname:
                target = 1.60
            elif 'calf' in lname:
                target = -2.15
            else:
                target = float(defaults[name])
            crouch_residual[0, index] = (
                target - float(defaults[name])) / action_scale

    # Optional per-step trace for the first few environments.  Used by
    # scripts/analyze_low_bar_trace.py to locate the bar contact in time and to
    # check whether the robot lowers its body at all before the bar.
    trace_count = min(int(getattr(args, "trace_envs", 0) or 0), num_envs)
    trace_rows = [] if (traces is not None and trace_count > 0) else None
    dynamics_rows = [] if trace_rows is not None else None
    base_quat_rows = [] if trace_rows is not None else None
    foot_normal_force_rows = [] if trace_rows is not None else None
    foot_center_z_rows = [] if trace_rows is not None else None
    termination_events = [] if trace_rows is not None else None
    contact_snapshots = [] if trace_rows is not None else None
    seen_bar_hits = torch.zeros(num_envs, dtype=torch.bool, device=device)
    prev_body_x = prev_body_z = None

    steps = int(round(args.duration * 50.0))
    recovery_steps = int(round(args.recovery_s * 50.0))
    for step_index in range(steps):
        base_env.commands[:, 0] = args.vx
        base_env.commands[:, 1] = 0.0
        base_env.commands[:, 2] = 0.0
        with torch.inference_mode():
            latent = model.adaptation_module(obs["estimator_obs"])
            action = model.actor_body(torch.cat((obs["policy_obs"], latent), dim=-1))
            if args.scripted_crouch_scale > 0.0:
                # Fully crouched before the robot's front reaches the frame;
                # keep the residual until its rear has cleared, then recover.
                x = base_env.base_pos[:, 0]
                ramp_in = torch.clamp(
                    (x - (bar_x - args.scripted_crouch_window))
                    / max(args.scripted_crouch_window - 0.40, 1e-6),
                    0.0, 1.0)
                recover_start = (
                    bar_x + float(Cfg.terrain.low_bar_length) + 0.35)
                ramp_out = torch.clamp(
                    (recover_start + 0.35 - x) / 0.35, 0.0, 1.0)
                phase = torch.minimum(ramp_in, ramp_out).unsqueeze(1)
                strength = torch.full_like(
                    x, float(args.scripted_crouch_scale))
                if args.scripted_crouch_height_target is not None:
                    strength = strength + args.scripted_crouch_height_gain * (
                        base_env.base_pos[:, 2]
                        - args.scripted_crouch_height_target)
                    strength = torch.clamp(strength, min=0.0, max=1.25)
                action = action + (
                    strength.unsqueeze(1) * phase * crouch_residual)
            if args.action_noise_std > 0.0:
                action = action + torch.randn_like(action) * args.action_noise_std
        obs, _, done, _ = env.step(action)

        current_force = torch.linalg.vector_norm(
            all_contact_forces.index_select(0, bar_body_indices), dim=1)
        max_bar_force = torch.maximum(max_bar_force, current_force)
        collision_this_step = active & (
            current_force > args.contact_force_threshold)
        collision |= collision_this_step
        violent_collision_this_step = active & (
            current_force > args.violent_contact_force_threshold)
        violent_collision |= violent_collision_this_step

        # Snapshot the whole robot the first time the bar is touched.  This
        # frame is valid because it is read before the environment can reset.
        if contact_snapshots is not None:
            newly_hit = collision_this_step & ~seen_bar_hits
            if newly_hit[:trace_count].any():
                hit_index = newly_hit[:trace_count].nonzero().flatten()
                contact_snapshots.append({
                    "step": step_index,
                    "env": hit_index.cpu().numpy(),
                    "base_x_rel": (
                        base_env.base_pos[hit_index, 0] - bar_x[hit_index]
                    ).cpu().numpy(),
                    "base_z": base_env.base_pos[hit_index, 2].cpu().numpy(),
                    "x_rel": (
                        base_env.rigid_body_state[hit_index, :, 0]
                        - bar_x[hit_index].unsqueeze(1)
                    ).cpu().numpy(),
                    "body_z": base_env.rigid_body_state[
                        hit_index, :, 2].cpu().numpy(),
                    "clearance": clearances[hit_index].cpu().numpy(),
                    "bar_force_n": current_force[hit_index].cpu().numpy(),
                })
            seen_bar_hits |= collision_this_step

        robot_bodies = base_env.rigid_body_state
        collision_lower, collision_upper, collision_valid = (
            base_env.low_bar_collision_bounds())
        collision_top_by_body = torch.where(
            collision_valid.unsqueeze(0),
            collision_upper[:, :, 2],
            torch.full_like(collision_upper[:, :, 2], float('-inf')),
        )
        collision_top, collision_top_body = collision_top_by_body.max(dim=1)
        if trace_rows is not None:
            trace_rows.append(np.stack([
                base_env.base_pos[:trace_count, 0].cpu().numpy(),
                base_env.base_pos[:trace_count, 1].cpu().numpy(),
                base_env.base_pos[:trace_count, 2].cpu().numpy(),
                collision_top[:trace_count].cpu().numpy(),
                collision_top_body[:trace_count].float().cpu().numpy(),
                current_force[:trace_count].cpu().numpy(),
                action[:trace_count].abs().amax(dim=1).cpu().numpy(),
                active[:trace_count].float().cpu().numpy(),
            ], axis=-1).astype(np.float32))
            # Base quaternion uses Isaac Gym xyzw order.  Capture foot net
            # contact force along world +z in the asset body order.  These
            # channels let paired traces locate the first contact divergence.
            base_quat_rows.append(
                base_env.base_quat[:trace_count].cpu().numpy().astype(np.float32))
            foot_normal_force_rows.append(
                base_env.contact_forces[:trace_count]
                .index_select(1, base_env.feet_indices)[:, :, 2]
                .cpu().numpy().astype(np.float32))
            foot_center_z_rows.append(
                base_env.rigid_body_state[:trace_count]
                .index_select(1, base_env.feet_indices)[:, :, 2]
                .cpu().numpy().astype(np.float32))
            # Per-step joint dynamics for paired pre-contact sim-to-sim audits.
            # The Isaac torque sample is the final physics substep of env.step.
            dynamics_rows.append(np.stack([
                base_env.dof_pos[:trace_count].cpu().numpy(),
                base_env.dof_vel[:trace_count].cpu().numpy(),
                base_env.joint_pos_target[:trace_count].cpu().numpy(),
                base_env.torques[:trace_count].cpu().numpy(),
            ], axis=-1).astype(np.float32))
            # Which robot body actually touched the bar?  env.step() has already
            # reset a terminated environment, so the contact forces still belong
            # to the physics step that caused the termination while the poses
            # are the respawned ones.  Keep the previous frame's per-body poses.
            newly_done = (done.bool() & active)[:trace_count]
            if newly_done.any():
                env_index = newly_done.nonzero().flatten()
                robot_contact = torch.linalg.vector_norm(
                    base_env.contact_forces, dim=-1)
                events = {
                    "step": step_index,
                    "env": env_index.cpu().numpy(),
                    "contact_n": robot_contact[env_index].cpu().numpy(),
                }
                if prev_body_x is not None:
                    events["prev_body_x_rel"] = (
                        prev_body_x[env_index] - bar_x[env_index].unsqueeze(1)
                    ).cpu().numpy()
                    events["prev_body_z"] = prev_body_z[env_index].cpu().numpy()
                termination_events.append(events)
            prev_body_x = robot_bodies[:trace_count, :, 0].clone()
            prev_body_z = robot_bodies[:trace_count, :, 2].clone()
        valid = collision_valid.unsqueeze(0)
        whole_robot_x = torch.where(
            valid,
            collision_lower[:, :, 0],
            torch.full_like(collision_lower[:, :, 0], float('inf')),
        ).amin(dim=1)
        origin_y = base_env.env_origins[:, 1].unsqueeze(1)
        lateral_extent = torch.maximum(
            (collision_lower[:, :, 1] - origin_y).abs(),
            (collision_upper[:, :, 1] - origin_y).abs(),
        )
        lateral_extent = torch.where(
            valid, lateral_extent, torch.zeros_like(lateral_extent))
        body_lateral_extent = lateral_extent.amax(dim=1)
        opening_half_width = (
            0.5 * float(Cfg.terrain.low_bar_width)
            - args.gate_margin
        )
        frame_length = float(getattr(Cfg.terrain, "low_bar_length", 0.0))
        envelope_tolerance = float(getattr(
            Cfg.terrain, "low_bar_envelope_tolerance", 0.005))
        overlaps_tunnel = (
            valid
            & (collision_upper[:, :, 0] >= bar_x.unsqueeze(1))
            & (collision_lower[:, :, 0]
               <= bar_x.unsqueeze(1) + frame_length)
        )
        envelope_violation_this_step = (
            active
            & (overlaps_tunnel
               & (collision_upper[:, :, 2]
                  > clearances.unsqueeze(1) + envelope_tolerance)).any(dim=1)
        )
        envelope_violation |= envelope_violation_this_step
        crossed_plane = active & (
            whole_robot_x > bar_x + frame_length + args.pass_margin)
        first_crossing = crossed_plane & ~crossed_gate_plane
        inside_gate = body_lateral_extent < opening_half_width
        # Alignment is evaluated during approach, but bypass becomes a rule
        # failure only once some collision geometry overlaps the tunnel.
        near_frame = (
            valid
            & (collision_upper[:, :, 0] >= bar_x.unsqueeze(1))
            & (collision_lower[:, :, 0]
               <= bar_x.unsqueeze(1) + frame_length)
        )
        gate_lateral_extent = torch.where(
            near_frame, lateral_extent,
            torch.zeros_like(lateral_extent)).amax(dim=1)
        max_gate_lateral_extent = torch.maximum(
            max_gate_lateral_extent, gate_lateral_extent)
        bypassed_this_step = (
            active & near_frame.any(dim=1)
            & (gate_lateral_extent >= opening_half_width)
        )
        missed_gate_this_step = (
            bypassed_this_step | (first_crossing & ~inside_gate))
        crossed_outside_gate |= missed_gate_this_step
        just_passed = first_crossing & inside_gate & ~crossed_outside_gate
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
        # Low-bar collisions and missed-gate events intentionally terminate
        # episodes and are included in early_termi_buf. They are task failures,
        # not physical falls, so keep the categories mutually meaningful.
        task_failure_this_step = (
            violent_collision_this_step | envelope_violation_this_step
            | missed_gate_this_step)
        fell |= active & early & ~task_failure_this_step
        active &= ~done.bool() & ~task_failure_this_step

        # The attempt is complete once the robot has cleared the frame and
        # remained stable for the requested recovery window.  Evaluation keeps
        # the simulator running so all candidates get the same wall-clock
        # horizon, but events after that completion point belong to subsequent
        # free locomotion and must not retroactively invalidate the obstacle.
        # Freeze the attempt here, matching the training environment's
        # terminate-on-success contract and the rule-level event ledger.
        success_this_step = (
            active & passed & recovered & ~violent_collision
            & ~envelope_violation & ~fell)
        success_latched |= success_this_step
        active &= ~success_this_step

    success = success_latched
    distance = base_env.base_pos[:, 0] - start_x

    records = []
    for i in range(num_envs):
        records.append({
            "seed": seed,
            "env": i,
            "row": int(rows[i].item()),
            "clearance_m": float(clearances[i].item()),
            "success": bool(success[i].item()),
            "passed": bool(passed[i].item()),
            "recovered": bool(recovered[i].item()),
            "crossed_outside_gate": bool(crossed_outside_gate[i].item()),
            "collision": bool(collision[i].item()),
            "violent_collision": bool(violent_collision[i].item()),
            "envelope_violation": bool(envelope_violation[i].item()),
            "fell": bool(fell[i].item()),
            "max_bar_force_n": float(max_bar_force[i].item()),
            "min_base_height_m": float(min_base_height[i].item()),
            "max_abs_roll_pitch_rad": float(max_abs_roll_pitch[i].item()),
            "max_gate_lateral_extent_m": float(
                max_gate_lateral_extent[i].item()),
            "final_displacement_x_m": float(distance[i].item()),
        })

    summary = _summarize(records)
    summary["seed"] = seed
    summary["by_row"] = _by_row(records)
    summary["groups"] = _grouped(
        records, float(Cfg.terrain.low_bar_target_clearance))
    if trace_rows is not None:
        traces[(seed, trace_count)] = {
            "trace": np.stack(trace_rows),          # (steps, envs, fields)
            "joint_dynamics": np.stack(dynamics_rows),  # (steps, envs, joints, fields)
            "base_quat_xyzw": np.stack(base_quat_rows),  # (steps, envs, 4)
            "initial_base_quat_xyzw": initial_base_quat,
            "initial_base_z_m": initial_base_z,
            "initial_foot_normal_force_n": initial_foot_normal_force,
            "initial_foot_center_z_m": initial_foot_center_z,
            "initial_base_lin_vel_m_s": initial_base_lin_vel,
            "initial_base_ang_vel_rad_s": initial_base_ang_vel,
            "initial_joint_pos_rad": initial_joint_pos,
            "initial_joint_vel_rad_s": initial_joint_vel,
            "foot_center_z_m": np.stack(foot_center_z_rows),
            "foot_normal_force_n": np.stack(foot_normal_force_rows),  # (steps, envs, feet)
            "foot_names": np.array([
                base_env.body_names[int(index)]
                for index in base_env.feet_indices.cpu().tolist()]),
            "joint_dynamics_fields": np.array([
                "joint_pos_rad", "joint_vel_rad_s",
                "pd_target_rad", "last_substep_torque_nm"]),
            "joint_names": np.array(base_env.dof_names),
            "initial_policy_obs": initial_policy_obs,
            "initial_estimator_obs": initial_estimator_obs,
            "bar_x": bar_x[:trace_count].cpu().numpy(),
            "clearance_m": clearances[:trace_count].cpu().numpy(),
            "rows": rows[:trace_count].cpu().numpy(),
            # np.savez cannot store a list of dicts portably, so keep the
            # per-env outcome records as JSON text.
            "records_json": json.dumps(records[:trace_count]),
            "fields": np.array(["base_x", "base_y", "base_z",
                                "max_collision_top_z",
                                "max_collision_top_body_index",
                                "bar_force_n", "action_absmax", "active"]),
            "body_names": np.array(base_env.gym.get_actor_rigid_body_names(
                base_env.envs[0], base_env.actor_handles[0])),
            # Raw actor output is clipped at this value inside the env, so
            # saturation can only be judged against it, not against unit scale.
            "clip_actions": np.array(float(Cfg.normalization.clip_actions)),
            "action_scale": np.array(float(Cfg.control.action_scale)),
            "termination_events_json": json.dumps([
                {key: (value.tolist() if hasattr(value, "tolist") else value)
                 for key, value in event.items()}
                for event in (termination_events or [])
            ]),
            "contact_snapshots_json": json.dumps([
                {key: (value.tolist() if hasattr(value, "tolist") else value)
                 for key, value in snapshot.items()}
                for snapshot in (contact_snapshots or [])
            ]),
            "dt_s": 0.02,
        }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Low-bar run containing parameters.pkl")
    parser.add_argument("--iterations", type=int, nargs="+", default=[-1])
    parser.add_argument("--baseline-run")
    parser.add_argument("--baseline-iteration", type=int, default=59950)
    parser.add_argument("--num-envs", type=int, default=48)
    parser.add_argument(
        "--profile", choices=["nominal", "train-matched"], default="nominal",
        help=("nominal disables pose/sensor/dynamics randomization; "
              "train-matched preserves the randomization saved by the run"),
    )
    parser.add_argument(
        "--only-randomization", nargs="+", default=None,
        choices=("joint-init", "base-velocity", "observation-noise",
                 "height-noise", "dynamics"),
        help=("Ablation: start from nominal and restore only these saved "
              "train-matched randomizations. Omit to preserve --profile "
              "behavior; lateral/yaw overrides still apply afterward."),
    )
    parser.add_argument(
        "--joint-passive-damping", nargs=3, type=float,
        metavar=("HIP", "THIGH", "CALF"), default=None,
        help=("Override Isaac DOF passive damping in N m s/rad at actor "
              "creation; leaves controller PD damping unchanged."),
    )
    parser.add_argument("--seed", type=int, default=7,
                        help="Single held-out seed, reused for every candidate.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=("Held-out seeds reused for every candidate. Formal candidates "
              "should use two seeds; each seed is paired across candidates."),
    )
    parser.add_argument(
        "--reference", default=None,
        help=("Candidate label used as the paired-comparison reference. "
              "Defaults to the first candidate (the baseline when "
              "--baseline-run is given)."),
    )
    parser.add_argument(
        "--lateral-init-range", type=float, default=None,
        help="Optional override for spawn lateral half-range [m].",
    )
    parser.add_argument(
        "--yaw-init-range", type=float, default=None,
        help="Optional override for spawn yaw half-range [rad].",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--recovery-s", type=float, default=1.0)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument(
        "--bar-x", type=float, default=None,
        help="Optional low-bar forward offset override for physics diagnosis.",
    )
    parser.add_argument(
        "--competition-spec", action="store_true",
        help=("Override the legacy run geometry with the 2027 full frame: "
              "0.29 m high, 0.80 m inner width and 0.80 m length."),
    )
    parser.add_argument(
        "--low-bar-stage", type=int, choices=[1, 2, 3, 4], default=None,
        help=("Evaluate an exact planned geometry stage: "
              "1=0.36x1.10x0.40, 2=0.32x0.90x0.80, "
              "3=0.29x0.80x0.80, 4=0.2755x0.76x0.84 m."),
    )
    parser.add_argument(
        "--clearance", type=float, default=None,
        help=("Evaluate one fixed inner height in metres.  Applied after "
              "--competition-spec so curriculum stages such as 0.36 m can "
              "use the same 2027 full-frame geometry."),
    )
    parser.add_argument(
        "--action-noise-std", type=float, default=0.0,
        help=("Optional Gaussian action noise for diagnosing the stochastic "
              "training rollout; deployment evaluation should keep this at 0."),
    )
    parser.add_argument(
        "--scripted-crouch-scale", type=float, default=0.0,
        help=("Diagnostic residual strength added to the policy during the "
              "approach/tunnel phase; 0 is the unmodified deployed policy."),
    )
    parser.add_argument(
        "--scripted-crouch-window", type=float, default=0.80,
        help="Distance before the entrance at which the diagnostic crouch ramps in.",
    )
    parser.add_argument(
        "--scripted-crouch-height-target", type=float, default=None,
        help=("Optional diagnostic base-height feedback target in metres; "
              "used only when --scripted-crouch-scale is positive."),
    )
    parser.add_argument(
        "--scripted-crouch-height-gain", type=float, default=0.0,
        help="Gain applied to base height error in the diagnostic residual.",
    )
    parser.add_argument(
        "--action-clip-override", type=float, default=None,
        help=("Diagnostic override for the environment action clip.  Omit for "
              "the checkpoint/run configuration used in deployment."),
    )
    parser.add_argument(
        "--crouch-control-center", action="store_true",
        help=("Diagnostic: recenter zero action and reset pose on the "
              "mesh-verified low-crawl joint posture."),
    )
    parser.add_argument("--crouch-center-thigh", type=float, default=1.60)
    parser.add_argument("--crouch-center-calf", type=float, default=-2.15)
    parser.add_argument("--crouch-center-base-z", type=float, default=0.19)
    parser.add_argument("--contact-force-threshold", type=float, default=1.0)
    parser.add_argument(
        "--violent-contact-force-threshold", type=float, default=250.0,
        help=("Provisional impact threshold used only for the rule proxy; "
              "all lighter contacts are still recorded."),
    )
    parser.add_argument("--pass-margin", type=float, default=0.05)
    parser.add_argument("--gate-margin", type=float, default=0.02)
    parser.add_argument("--stable-roll-pitch", type=float, default=0.6)
    parser.add_argument("--stable-base-height", type=float, default=0.16)
    parser.add_argument("--sim-device", choices=["cuda:0", "cpu"], default="cuda:0")
    parser.add_argument(
        "--trace-envs", type=int, default=0,
        help=("Record a per-step trace for the first N environments and write "
              "it to --trace-out as an npz for bar-contact diagnosis."),
    )
    parser.add_argument(
        "--trace-out", default=None,
        help="Destination npz for --trace-envs (default: <out>.trace.npz).",
    )
    parser.add_argument(
        "--ignore-contact-termination", action="store_true",
        help=("Raise the environment's own bar-contact threshold so a bar hit "
              "no longer terminates the episode.  The evaluator still records "
              "the contact, which keeps the pose trace continuous through the "
              "bar instead of resetting the robot to spawn."),
    )
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.joint_passive_damping is not None and not all(
            math.isfinite(value) and value >= 0.0
            for value in args.joint_passive_damping):
        parser.error("--joint-passive-damping values must be finite and non-negative")
    if args.action_noise_std < 0.0:
        parser.error("--action-noise-std must be non-negative")
    if args.scripted_crouch_scale < 0.0:
        parser.error("--scripted-crouch-scale must be non-negative")
    if args.scripted_crouch_window <= 0.40:
        parser.error("--scripted-crouch-window must be greater than 0.40 m")
    if args.scripted_crouch_height_gain < 0.0:
        parser.error("--scripted-crouch-height-gain must be non-negative")
    if args.action_clip_override is not None and args.action_clip_override <= 0.0:
        parser.error("--action-clip-override must be positive")
    if args.violent_contact_force_threshold < args.contact_force_threshold:
        parser.error(
            "--violent-contact-force-threshold must be >= "
            "--contact-force-threshold")
    if args.clearance is not None and args.clearance <= 0.0:
        parser.error("--clearance must be positive")
    if args.competition_spec and args.low_bar_stage is not None:
        parser.error("--competition-spec and --low-bar-stage are mutually exclusive")

    with open(os.path.join(args.run, "parameters.pkl"), "rb") as handle:
        cfg_dict = pkl.load(handle)["Cfg"]
    load_config(cfg_dict, Cfg)
    saved_randomization = {
        "joint_init": list(Cfg.init_state.dof_pos_scale_range),
        "base_velocity": float(Cfg.init_state.base_velocity_half_range),
        "observation_noise": bool(Cfg.noise.add_noise),
        "height_noise": {name: getattr(Cfg.terrain, name)
                         for name in HEIGHT_NOISE_FLAGS if hasattr(Cfg.terrain, name)},
        "dynamics": {name: getattr(Cfg.domain_rand, name)
                     for name in DOMAIN_RANDOMIZATION_FLAGS
                     if hasattr(Cfg.domain_rand, name)},
    }
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    Cfg.env.num_envs = args.num_envs
    Cfg.env.episode_length_s = args.duration + 2.0
    Cfg.commands.resampling_time = args.duration + 2.0
    Cfg.commands.heading_command = False
    Cfg.commands.command_curriculum = False
    Cfg.terrain.curriculum = True
    Cfg.terrain.min_init_terrain_level = 0
    Cfg.terrain.max_init_terrain_level = int(Cfg.terrain.num_rows) - 1
    # Environments may reuse terrain origins; scaling num_cols with num_envs
    # creates a needlessly huge trimesh and can crash large held-out batches.
    Cfg.terrain.num_cols = max(12, min(int(Cfg.terrain.num_cols), 50))
    Cfg.terrain.border_size = 2.0
    Cfg.terrain.x_init_range = 0.0
    if args.competition_spec:
        Cfg.terrain.low_bar_clearance_min = 0.29
        Cfg.terrain.low_bar_clearance_max = 0.29
        Cfg.terrain.low_bar_target_clearance = 0.29
        Cfg.terrain.robocon_low_bar_clearance_by_level = [
            0.29 for _ in range(int(Cfg.terrain.num_rows))]
        Cfg.terrain.low_bar_width = 0.80
        Cfg.terrain.low_bar_length = 0.80
        Cfg.terrain.low_bar_asset_path = os.path.join(
            PROJECT, "resources/objects/low_bar/low_frame_2027.urdf")
    if args.low_bar_stage is not None:
        stage_specs = {
            1: (0.36, 1.10, 0.40, 'low_frame_stage1_h360_w1100_l400.urdf'),
            2: (0.32, 0.90, 0.80, 'low_frame_stage2_h320_w900_l800.urdf'),
            3: (0.29, 0.80, 0.80, 'low_frame_2027.urdf'),
            4: (0.2755, 0.76, 0.84, 'low_frame_stage4_h2755_w760_l840.urdf'),
        }
        height, width, length, asset_name = stage_specs[args.low_bar_stage]
        Cfg.terrain.low_bar_clearance_min = height
        Cfg.terrain.low_bar_clearance_max = height
        Cfg.terrain.low_bar_target_clearance = height
        Cfg.terrain.robocon_low_bar_clearance_by_level = [
            height for _ in range(int(Cfg.terrain.num_rows))]
        Cfg.terrain.low_bar_width = width
        Cfg.terrain.low_bar_length = length
        Cfg.terrain.low_bar_asset_path = os.path.join(
            PROJECT, "resources/objects/low_bar", asset_name)
    if args.clearance is not None:
        Cfg.terrain.low_bar_clearance_min = args.clearance
        Cfg.terrain.low_bar_clearance_max = args.clearance
        Cfg.terrain.low_bar_target_clearance = args.clearance
        Cfg.terrain.robocon_low_bar_clearance_by_level = [
            args.clearance for _ in range(int(Cfg.terrain.num_rows))]
    if args.bar_x is not None:
        Cfg.terrain.low_bar_x = args.bar_x
    if args.action_clip_override is not None:
        Cfg.normalization.clip_actions = args.action_clip_override
    if args.crouch_control_center:
        for name in list(Cfg.init_state.default_joint_angles):
            lname = str(name).lower()
            if 'hip' in lname:
                Cfg.init_state.default_joint_angles[name] = (
                    0.10 if lname.startswith(('fl', 'rl')) else -0.10)
            elif 'thigh' in lname:
                Cfg.init_state.default_joint_angles[name] = args.crouch_center_thigh
            elif 'calf' in lname:
                Cfg.init_state.default_joint_angles[name] = args.crouch_center_calf
        Cfg.init_state.pos[2] = args.crouch_center_base_z
    if args.profile == "nominal" or args.only_randomization is not None:
        Cfg.terrain.y_init_range = 0.0
        Cfg.terrain.yaw_init_range = 0.0
        Cfg.init_state.dof_pos_scale_range = [1.0, 1.0]
        Cfg.init_state.base_velocity_half_range = 0.0
        Cfg.noise.add_noise = False
        for name in HEIGHT_NOISE_FLAGS:
            if hasattr(Cfg.terrain, name):
                setattr(Cfg.terrain, name, 0.0)
        _disable_domain_randomization(Cfg)
    if args.only_randomization is not None:
        selected = set(args.only_randomization)
        if "joint-init" in selected:
            Cfg.init_state.dof_pos_scale_range = saved_randomization["joint_init"]
        if "base-velocity" in selected:
            Cfg.init_state.base_velocity_half_range = saved_randomization["base_velocity"]
        if "observation-noise" in selected:
            Cfg.noise.add_noise = saved_randomization["observation_noise"]
        if "height-noise" in selected:
            for name, value in saved_randomization["height_noise"].items():
                setattr(Cfg.terrain, name, value)
        if "dynamics" in selected:
            for name, value in saved_randomization["dynamics"].items():
                setattr(Cfg.domain_rand, name, value)
    if args.lateral_init_range is not None:
        Cfg.terrain.y_init_range = args.lateral_init_range
    if args.yaw_init_range is not None:
        Cfg.terrain.yaw_init_range = args.yaw_init_range
    Cfg.asset.self_collisions = 1
    Cfg.terrain.low_bar_external_evaluator = True
    Cfg.terrain.low_bar_terminate_on_success = False
    if args.ignore_contact_termination:
        Cfg.terrain.low_bar_contact_force_threshold = 1e9
    else:
        Cfg.terrain.low_bar_contact_force_threshold = (
            args.violent_contact_force_threshold)
    if args.sim_device == "cpu":
        Cfg.sim.use_gpu_pipeline = False

    env_class = VelocityTrackingEasyEnv
    if args.joint_passive_damping is not None:
        damping_values = dict(zip(
            ("hip", "thigh", "calf"), args.joint_passive_damping))

        class FixedPassiveDampingEnv(VelocityTrackingEasyEnv):
            def _process_dof_props(self, props, env_id):
                props = super()._process_dof_props(props, env_id)
                return _set_joint_passive_damping(
                    props, self.dof_names, damping_values)

        env_class = FixedPassiveDampingEnv

    base_env = env_class(
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

    seeds = args.seeds if args.seeds else [args.seed]
    reference_label = args.reference or candidates[0][0]
    if reference_label not in [label for label, _ in candidates]:
        parser.error(
            f"--reference {reference_label} is not one of "
            f"{[label for label, _ in candidates]}")

    result = {
        "protocol": {
            "profile": args.profile,
            "only_randomization": args.only_randomization,
            "joint_passive_damping_nms_per_rad": (
                None if args.joint_passive_damping is None else dict(zip(
                    ("hip", "thigh", "calf"), args.joint_passive_damping))),
            "num_envs": args.num_envs,
            "duration_s": args.duration,
            "vx_mps": args.vx,
            "action_noise_std": args.action_noise_std,
            "scripted_crouch_scale": args.scripted_crouch_scale,
            "scripted_crouch_window_m": args.scripted_crouch_window,
            "scripted_crouch_height_target_m": (
                args.scripted_crouch_height_target),
            "scripted_crouch_height_gain": args.scripted_crouch_height_gain,
            "action_clip_override": args.action_clip_override,
            "crouch_control_center": bool(args.crouch_control_center),
            "crouch_center_thigh_rad": args.crouch_center_thigh,
            "crouch_center_calf_rad": args.crouch_center_calf,
            "crouch_center_base_z_m": args.crouch_center_base_z,
            "recovery_s": args.recovery_s,
            "contact_force_threshold_n": args.contact_force_threshold,
            "violent_contact_force_threshold_n": (
                args.violent_contact_force_threshold),
            "pass_margin_m": args.pass_margin,
            "gate_margin_m": args.gate_margin,
            "competition_spec_geometry": bool(args.competition_spec),
            "low_bar_stage": args.low_bar_stage,
            "fixed_clearance_m": args.clearance,
            "frame_inner_width_m": float(Cfg.terrain.low_bar_width),
            "frame_length_m": float(getattr(
                Cfg.terrain, "low_bar_length", 0.0)),
            "seed": args.seed,
            "seeds": seeds,
            "episodes_per_seed": args.num_envs,
            "episodes_total_per_candidate": args.num_envs * len(seeds),
            "paired_comparison": True,
            "reference": reference_label,
            "competition_clearance_m": float(Cfg.terrain.low_bar_target_clearance),
            "clearance_by_row": [
                round(float(x), 4)
                for x in Cfg.terrain.robocon_low_bar_clearance_by_level
            ],
            "lateral_init_half_range_m": Cfg.terrain.y_init_range,
            "yaw_init_half_range_rad": Cfg.terrain.yaw_init_range,
            "dof_pos_init_scale_range": list(getattr(
                Cfg.init_state, "dof_pos_scale_range", [0.5, 1.5])),
            "base_velocity_init_half_range": float(getattr(
                Cfg.init_state, "base_velocity_half_range", 0.5)),
            "observation_noise": bool(Cfg.noise.add_noise),
            "height_measurement_noise": {
                "per_step_xy_std_m": Cfg.terrain.height_measurements_per_step_xy_noise_std,
                "per_step_z_std_m": Cfg.terrain.height_measurements_per_step_z_noise_std,
                "per_env_xy_std_m": Cfg.terrain.height_measurements_per_env_xy_noise_std,
                "per_env_z_std_m": Cfg.terrain.height_measurements_per_env_z_noise_std,
                "per_env_probability": Cfg.terrain.height_measurements_per_env_noise_prob,
            },
            "evaluator_sha256": _sha256(os.path.abspath(__file__)),
            **_git_state(),
        },
        "candidates": {},
        "paired_comparisons": {},
    }

    models = []
    for label, checkpoint in candidates:
        print(f"\n=== loading {label}: {checkpoint} ===", flush=True)
        model, migration = _load_policy(env, checkpoint)
        models.append((label, checkpoint, model, migration))

    for label, checkpoint, model, migration in models:
        pooled_records = []
        per_seed = {}
        traces = {} if args.trace_envs > 0 else None
        for seed in seeds:
            print(f"\n=== evaluating {label} (seed {seed}): {checkpoint} ===",
                  flush=True)
            seed_summary, seed_records = _evaluate(
                env, model, args, seed, traces)
            per_seed[str(seed)] = seed_summary
            pooled_records.extend(seed_records)
        if traces:
            trace_path = args.trace_out or os.path.splitext(
                args.out or os.path.join(args.run, "eval_low_bar_isaac.json"))[0] \
                + f".{label}.trace.npz"
            np.savez_compressed(
                trace_path,
                **{f"{seed}_{count}::{key}": value
                   for (seed, count), payload in traces.items()
                   for key, value in payload.items()},
            )
            print(f"wrote trace {trace_path}", flush=True)
        summary = _summarize(pooled_records)
        summary["seed"] = seeds[0] if len(seeds) == 1 else None
        summary["seeds"] = seeds
        summary["by_row"] = _by_row(pooled_records)
        summary["groups"] = _grouped(
            pooled_records, float(Cfg.terrain.low_bar_target_clearance))
        summary["per_seed"] = per_seed
        summary["checkpoint"] = checkpoint
        summary["checkpoint_migration"] = migration
        summary["records"] = pooled_records
        result["candidates"][label] = summary
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("records", "by_row", "per_seed")},
                         ensure_ascii=False, indent=2), flush=True)
        print("grouped:", json.dumps(summary["groups"], ensure_ascii=False),
              flush=True)

    reference_records = result["candidates"][reference_label]["records"]
    for label, _, _, _ in models:
        if label == reference_label:
            continue
        comparison = _paired_compare(
            reference_records, result["candidates"][label]["records"])
        comparison["reference"] = reference_label
        comparison["candidate"] = label
        result["paired_comparisons"][label] = comparison
        print(f"\npaired {label} vs {reference_label}: "
              f"delta={comparison['success_delta_percent']:+.2f}pp "
              f"CI95={comparison['success_delta_ci95_percent']} "
              f"p={comparison['mcnemar_exact_p']} "
              f"(discordant {comparison['reference_only_success']}"
              f"/{comparison['candidate_only_success']})", flush=True)

    out = args.out or os.path.join(args.run, "eval_low_bar_isaac.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")
    sys.stdout.flush()
    # Isaac Gym's CUDA teardown segfaults on exit for some driver/GPU
    # combinations, which would look like a failed evaluation to any script that
    # checks the exit code even though every artifact was written. Exit on
    # purpose once the results are on disk.
    os._exit(0)


if __name__ == "__main__":
    main()
