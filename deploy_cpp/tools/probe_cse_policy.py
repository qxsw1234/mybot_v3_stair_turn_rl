#!/usr/bin/env python3
"""Probe CSE TorchScript policy inputs and outputs offline.

This is intentionally small and dependency-light: run it from an environment
with PyTorch and PyYAML installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml


NUM_JOINTS = 12
NUM_ACTIONS = 12
NUM_HEIGHT_POINTS = 77
HISTORY_LENGTH = 10


def tensor(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, -1)


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_observations(cfg, q, dq, command, gravity, height_distances, last_actions):
    q_t = tensor(q)
    dq_t = tensor(dq)
    command_t = tensor(command)
    gravity_t = tensor(gravity)
    height_t = tensor(height_distances)
    last_actions_t = tensor(last_actions)
    policy_dof_pos_t = tensor(cfg["policy_dof_pos"])
    command_scale_t = tensor(
        [
            cfg["command_lin_vel_scale"],
            cfg["command_lin_vel_scale"],
            cfg["command_yaw_scale"],
        ]
    )

    height_obs = (
        torch.clamp(cfg["height_bias"] - height_t, -1.0, 1.0)
        * cfg["height_scale"]
    )
    policy_obs = torch.cat(
        [
            command_t * command_scale_t,
            gravity_t * cfg["projected_gravity_scale"],
            (q_t - policy_dof_pos_t) * cfg["dof_pos_scale"],
            dq_t * cfg["dof_vel_scale"],
            last_actions_t,
            height_obs,
        ],
        dim=1,
    ).clamp(-cfg["clip_obs"], cfg["clip_obs"])
    estimator_obs = torch.cat(
        [
            gravity_t * cfg["projected_gravity_scale"],
            (q_t - policy_dof_pos_t) * cfg["dof_pos_scale"],
            dq_t * cfg["dof_vel_scale"],
            last_actions_t,
            height_obs,
        ],
        dim=1,
    ).clamp(-cfg["clip_obs"], cfg["clip_obs"])
    return policy_obs, estimator_obs


def make_history(obs, repeat_current):
    if repeat_current:
        return obs.repeat(1, HISTORY_LENGTH)
    hist = torch.zeros(1, obs.shape[1] * HISTORY_LENGTH, dtype=torch.float32)
    hist[0, -obs.shape[1] :] = obs
    return hist


def fmt(values):
    return "[" + ", ".join(f"{float(v):+.3f}" for v in values) + "]"


def run_case(name, cfg, adaptation, body, q, gravity, height_distances, repeat_current):
    policy_obs, estimator_obs = build_observations(
        cfg=cfg,
        q=q,
        dq=[0.0] * NUM_JOINTS,
        command=[0.0, 0.0, 0.0],
        gravity=gravity,
        height_distances=height_distances,
        last_actions=[0.0] * NUM_ACTIONS,
    )
    policy_hist = make_history(policy_obs, repeat_current)
    estimator_hist = make_history(estimator_obs, repeat_current)
    with torch.no_grad():
        latent = adaptation(estimator_hist)
        actions = body(torch.cat([policy_hist, latent], dim=1)).clamp(
            -cfg["clip_actions"], cfg["clip_actions"]
        )
    policy_dof_pos = tensor(cfg["policy_dof_pos"])
    target = policy_dof_pos + actions * cfg["action_scale"]
    print(f"\n{name} ({'repeat-current' if repeat_current else 'zero-padded'} history)")
    print(
        "  obs dims: "
        f"policy_step={policy_obs.shape[1]} estimator_step={estimator_obs.shape[1]} "
        f"policy_hist={policy_hist.shape[1]} estimator_hist={estimator_hist.shape[1]}"
    )
    print(
        "  obs stats: "
        f"policy[min,max]=[{float(policy_obs.min()):+.3f},{float(policy_obs.max()):+.3f}] "
        f"est[min,max]=[{float(estimator_obs.min()):+.3f},{float(estimator_obs.max()):+.3f}]"
    )
    print(
        "  action stats: "
        f"min={float(actions.min()):+.3f} max={float(actions.max()):+.3f} "
        f"mean_abs={float(actions.abs().mean()):.3f}"
    )
    print(f"  actions: {fmt(actions.flatten())}")
    print(f"  target : {fmt(target.flatten())}")
    print(f"  latent : {fmt(latent.flatten())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("deploy_cpp/config/robots/mybot_v2_1_cse.yaml"),
    )
    parser.add_argument("--body", type=Path, default=None)
    parser.add_argument("--adaptation", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = args.config.resolve().parents[2]
    body_path = args.body or root / cfg["body_path"]
    adaptation_path = args.adaptation or root / cfg["adaptation_module_path"]

    body = torch.jit.load(str(body_path), map_location="cpu").eval()
    adaptation = torch.jit.load(str(adaptation_path), map_location="cpu").eval()

    print(f"config: {args.config}")
    print(f"body: {body_path}")
    print(f"adaptation: {adaptation_path}")
    print("expected dims: policy_step=119 estimator_step=116 policy_hist=1190 body_in=1200")

    flat_distance = [cfg["nominal_base_height"]] * NUM_HEIGHT_POINTS
    zero_height = [0.0] * NUM_HEIGHT_POINTS
    negative_distance = [-cfg["nominal_base_height"]] * NUM_HEIGHT_POINTS

    cases = [
        (
            "nominal standing, gravity z=-1, flat distance",
            cfg["policy_dof_pos"],
            [0.0, 0.0, -1.0],
            flat_distance,
        ),
        (
            "nominal standing, gravity z=+1, flat distance",
            cfg["policy_dof_pos"],
            [0.0, 0.0, 1.0],
            flat_distance,
        ),
        (
            "nominal standing, gravity z=-1, zero height topic",
            cfg["policy_dof_pos"],
            [0.0, 0.0, -1.0],
            zero_height,
        ),
        (
            "nominal standing, gravity z=-1, negative distance topic",
            cfg["policy_dof_pos"],
            [0.0, 0.0, -1.0],
            negative_distance,
        ),
        (
            "default_dof_pos pose, gravity z=-1, flat distance",
            cfg["default_dof_pos"],
            [0.0, 0.0, -1.0],
            flat_distance,
        ),
    ]

    for case in cases:
        for repeat_current in (False, True):
            name, q, gravity, height_distances = case
            run_case(
                name,
                cfg,
                adaptation,
                body,
                q,
                gravity,
                height_distances,
                repeat_current,
            )


if __name__ == "__main__":
    main()
