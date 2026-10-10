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


WILSON_Z = 1.959963984540054  # two-sided 95%

RATE_FIELDS = (
    ("success", "success_rate_percent"),
    ("passed", "pass_rate_percent"),
    ("recovered", "recovery_rate_percent"),
    ("crossed_outside_gate", "outside_gate_rate_percent"),
    ("collision", "collision_rate_percent"),
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


def _evaluate(env, model, args, seed):
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
            if args.action_noise_std > 0.0:
                action = action + torch.randn_like(action) * args.action_noise_std
        obs, _, done, _ = env.step(action)

        current_force = torch.linalg.vector_norm(
            all_contact_forces.index_select(0, bar_body_indices), dim=1)
        max_bar_force = torch.maximum(max_bar_force, current_force)
        collision_this_step = active & (
            current_force > args.contact_force_threshold)
        collision |= collision_this_step

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
        missed_gate_this_step = first_crossing & ~inside_gate
        crossed_outside_gate |= missed_gate_this_step
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
        # Low-bar collisions and missed-gate events intentionally terminate
        # episodes and are included in early_termi_buf. They are task failures,
        # not physical falls, so keep the categories mutually meaningful.
        task_failure_this_step = collision_this_step | missed_gate_this_step
        fell |= active & early & ~task_failure_this_step
        active &= ~done.bool()

    success = passed & recovered & ~collision & ~fell
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
            "fell": bool(fell[i].item()),
            "max_bar_force_n": float(max_bar_force[i].item()),
            "min_base_height_m": float(min_base_height[i].item()),
            "max_abs_roll_pitch_rad": float(max_abs_roll_pitch[i].item()),
            "final_displacement_x_m": float(distance[i].item()),
        })

    summary = _summarize(records)
    summary["seed"] = seed
    summary["by_row"] = _by_row(records)
    summary["groups"] = _grouped(
        records, float(Cfg.terrain.low_bar_target_clearance))
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
        "--action-noise-std", type=float, default=0.0,
        help=("Optional Gaussian action noise for diagnosing the stochastic "
              "training rollout; deployment evaluation should keep this at 0."),
    )
    parser.add_argument("--contact-force-threshold", type=float, default=1.0)
    parser.add_argument("--pass-margin", type=float, default=0.05)
    parser.add_argument("--gate-margin", type=float, default=0.02)
    parser.add_argument("--stable-roll-pitch", type=float, default=0.6)
    parser.add_argument("--stable-base-height", type=float, default=0.16)
    parser.add_argument("--sim-device", choices=["cuda:0", "cpu"], default="cuda:0")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.action_noise_std < 0.0:
        parser.error("--action-noise-std must be non-negative")

    with open(os.path.join(args.run, "parameters.pkl"), "rb") as handle:
        cfg_dict = pkl.load(handle)["Cfg"]
    load_config(cfg_dict, Cfg)
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
    if args.profile == "nominal":
        Cfg.terrain.y_init_range = 0.0
        Cfg.terrain.yaw_init_range = 0.0
        Cfg.noise.add_noise = False
        for name in (
            "height_measurements_per_step_xy_noise_std",
            "height_measurements_per_step_z_noise_std",
            "height_measurements_per_env_xy_noise_std",
            "height_measurements_per_env_z_noise_std",
            "height_measurements_per_env_noise_prob",
        ):
            if hasattr(Cfg.terrain, name):
                setattr(Cfg.terrain, name, 0.0)
        _disable_domain_randomization(Cfg)
    if args.lateral_init_range is not None:
        Cfg.terrain.y_init_range = args.lateral_init_range
    if args.yaw_init_range is not None:
        Cfg.terrain.yaw_init_range = args.yaw_init_range
    Cfg.asset.self_collisions = 1
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

    seeds = args.seeds if args.seeds else [args.seed]
    reference_label = args.reference or candidates[0][0]
    if reference_label not in [label for label, _ in candidates]:
        parser.error(
            f"--reference {reference_label} is not one of "
            f"{[label for label, _ in candidates]}")

    result = {
        "protocol": {
            "profile": args.profile,
            "num_envs": args.num_envs,
            "duration_s": args.duration,
            "vx_mps": args.vx,
            "action_noise_std": args.action_noise_std,
            "recovery_s": args.recovery_s,
            "contact_force_threshold_n": args.contact_force_threshold,
            "pass_margin_m": args.pass_margin,
            "gate_margin_m": args.gate_margin,
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
        for seed in seeds:
            print(f"\n=== evaluating {label} (seed {seed}): {checkpoint} ===",
                  flush=True)
            seed_summary, seed_records = _evaluate(env, model, args, seed)
            per_seed[str(seed)] = seed_summary
            pooled_records.extend(seed_records)
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
