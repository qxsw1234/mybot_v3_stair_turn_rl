#!/usr/bin/env python3
"""Paired MuJoCo trial of walking -> beam expert -> walking.

This is a switching prototype. The beam checkpoint is exploratory and is not
approved for deployment. Both policies see identical spawn perturbations and
are scored at x >= beam_length + 0.45 m so the return to walking is exercised.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from eval_robocon_2027_baselines import (
    LOCOMOTION_CKPT, aggregate, git_provenance, run_trial, sha256_file,
)
from robodog_gym.envs.robocon_2027 import load_rules

ROOT = Path(__file__).resolve().parents[1]
EXPERT_CKPT = (
    ROOT / "runs/mybot_v3_beam_expert_resume_059974"
    / "2026-10-15_11-45-23.345648/checkpoints"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--blend-duration", type=float, default=0.3)
    ap.add_argument("--expert-body", type=Path, default=EXPERT_CKPT / "body_latest.jit")
    ap.add_argument("--expert-adaptation", type=Path,
                    default=EXPERT_CKPT / "adaptation_module_latest.jit")
    ap.add_argument("--output", type=Path, default=ROOT /
                    "evaluations/robocon_2027/beam_mode_switch_pilot.json")
    args = ap.parse_args()
    if args.trials <= 0:
        ap.error("--trials must be positive")
    for path in (args.expert_body, args.expert_adaptation):
        if not path.is_file():
            ap.error(f"missing expert model: {path}")
    rules = load_rules()
    base_body = LOCOMOTION_CKPT / "body_059950.jit"
    base_adaptation = LOCOMOTION_CKPT / "adaptation_module_059950.jit"
    rng = np.random.default_rng(args.seed)
    baseline, switched = [], []
    for trial in range(args.trials):
        perturb = (
            float(rng.uniform(-0.03, 0.03)),
            float(rng.uniform(-0.03, 0.03)),
            float(rng.uniform(math.radians(-2), math.radians(2))),
        )
        shared = dict(obstacle="beam", rules=rules, max_seconds=60.0,
                      controller="pursuit", spawn_perturbation=perturb,
                      body_path=base_body, adaptation_path=base_adaptation,
                      beam_finish_extra_m=0.45)
        baseline.append(run_trial(**shared))
        switched.append(run_trial(
            **shared, beam_expert_body_path=args.expert_body,
            beam_expert_adaptation_path=args.expert_adaptation,
            beam_switch_blend_duration_s=args.blend_duration))
        print(f"{trial+1}/{args.trials}: baseline={baseline[-1]['success']} "
              f"switched={switched[-1]['success']} "
              f"modes={len(switched[-1]['beam_switch_events'])}", flush=True)
    result = {
        "kind": "exploratory_paired_mujoco_mode_switch",
        "scope": "sim_only_no_deployment_promotion",
        "seed": args.seed,
        "geometry": {k: rules["obstacles"]["beam"][k] for k in
                     ("width_m", "length_m", "height_m")},
        "finish_extra_m": 0.45,
        "blend_duration_s": args.blend_duration,
        "models": {
            "locomotion_body": str(base_body), "locomotion_body_sha256": sha256_file(base_body),
            "locomotion_adaptation": str(base_adaptation),
            "locomotion_adaptation_sha256": sha256_file(base_adaptation),
            "expert_body": str(args.expert_body), "expert_body_sha256": sha256_file(args.expert_body),
            "expert_adaptation": str(args.expert_adaptation),
            "expert_adaptation_sha256": sha256_file(args.expert_adaptation),
        },
        "provenance": git_provenance(),
        "baseline": aggregate("beam", baseline),
        "switched": aggregate("beam", switched),
        "paired": [
            {"trial": i, "spawn": baseline[i]["spawn"],
             "baseline_success": baseline[i]["success"],
             "switched_success": switched[i]["success"],
             "baseline_time_s": baseline[i]["completion_time_s"],
             "switched_time_s": switched[i]["completion_time_s"],
             "switch_events": switched[i]["beam_switch_events"],
             "baseline_reason": baseline[i]["reason"],
             "switched_reason": switched[i]["reason"]}
            for i in range(args.trials)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "baseline": result["baseline"],
        "switched": result["switched"],
        "both_switches": sum(len(r["beam_switch_events"]) == 2 for r in switched),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
