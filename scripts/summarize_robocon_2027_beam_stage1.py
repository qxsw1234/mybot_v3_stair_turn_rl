#!/usr/bin/env python3
"""Summarize stage-1 bridge rollouts and audit their pairing limits."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wilson(count, total):
    z = 1.959963984540054
    p = count / total
    den = 1 + z*z/total
    centre = (p + z*z/(2*total))/den
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [round(max(0, centre-radius), 4), round(min(1, centre+radius), 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--mujoco-baseline", type=Path, required=True)
    ap.add_argument("--mujoco-candidate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    isaac = {"baseline": [], "candidate": []}
    per_seed = []
    protocol = None
    max_initial_root_difference = 0.0
    max_initial_dof_difference = 0.0
    for seed in a.seeds:
        rows = {}
        hashes = {}
        for label in ("baseline", "candidate"):
            path = a.directory / f"beam_stage1_isaac_{label}50_seed{seed}.json"
            data = json.loads(path.read_text())
            if data["seed"] != seed or data["num_envs"] != 50 or data["profile"] != "pose":
                raise ValueError(f"unexpected Isaac protocol in {path}")
            identity = (data["geometry_m"], data["command_vx_mps"],
                        data["evaluator_sha256"], data["git_commit"])
            if protocol is None:
                protocol = identity
            elif identity != protocol:
                raise ValueError(f"Isaac geometry, command or evaluator changed in {path}")
            rows[label] = data["candidates"][label]["records"]
            hashes[label] = sha256(path)
        if len(rows["baseline"]) != len(rows["candidate"]) or len(rows["baseline"]) != 50:
            raise ValueError(f"wrong episode count for seed {seed}")
        for x, y in zip(rows["baseline"], rows["candidate"]):
            if x["env_id"] != y["env_id"] or x["seed"] != y["seed"]:
                raise ValueError(f"mismatched environment ids for seed {seed}")
            max_initial_root_difference = max(
                max_initial_root_difference,
                float(np.max(np.abs(np.asarray(x["initial_base_xyz_quat"]) -
                                    np.asarray(y["initial_base_xyz_quat"])))))
            max_initial_dof_difference = max(
                max_initial_dof_difference,
                float(np.max(np.abs(np.asarray(x["initial_dof_pos"]) -
                                    np.asarray(y["initial_dof_pos"])))))
        counts = {label: sum(bool(r["success"]) for r in records)
                  for label, records in rows.items()}
        per_seed.append({"seed": seed, "episodes": 50,
                         "successes": counts, "source_sha256": hashes})
        for label in isaac:
            isaac[label].extend(rows[label])
    total = len(isaac["baseline"])
    baseline_success = np.asarray([int(x["success"]) for x in isaac["baseline"]])
    candidate_success = np.asarray([int(x["success"]) for x in isaac["candidate"]])
    rng = np.random.default_rng(2026101501)
    bootstrap = (rng.choice(candidate_success, size=(20000, total), replace=True).mean(axis=1)
                 - rng.choice(baseline_success, size=(20000, total), replace=True).mean(axis=1))
    isaac_summary = {
        "episodes": total,
        "per_seed": per_seed,
        "protocol": {"geometry_m": protocol[0], "command_vx_mps": protocol[1],
                     "evaluator_sha256": protocol[2], "git_commit": protocol[3]},
        "initial_base_and_joint_states_verified_identical": (
            max_initial_root_difference == 0.0 and max_initial_dof_difference == 0.0),
        "max_initial_root_difference": round(max_initial_root_difference, 7),
        "max_initial_dof_difference_rad": round(max_initial_dof_difference, 7),
        "baseline": {}, "candidate": {},
        "unpaired_rate_difference": {
            "candidate_minus_baseline": round(float(candidate_success.mean() - baseline_success.mean()), 4),
            "independent_bootstrap95": [round(float(x), 4) for x in np.quantile(bootstrap, [0.025, 0.975])],
        },
    }
    for label, records in isaac.items():
        n = sum(bool(r["success"]) for r in records)
        isaac_summary[label] = {
            "successes": n, "rate": round(n/total, 4),
            "wilson95": wilson(n, total),
            "outside": sum(bool(r["outside"]) for r in records),
            "timeout": sum(bool(r["timeout"]) for r in records),
        }
    mj = {}
    mj_rows = {}
    for label, path in (("baseline", a.mujoco_baseline), ("candidate", a.mujoco_candidate)):
        data = json.loads(path.read_text())
        if data["obstacles"]["beam"]["aggregate"]["trials"] != 200:
            raise ValueError(f"expected 200 MuJoCo episodes in {path}")
        records = data["obstacles"]["beam"]["records"]
        mj_rows[label] = records
        n = sum(bool(r["success"]) for r in records)
        mj[label] = {
            "source": str(path), "source_sha256": sha256(path),
            "successes": n, "episodes": len(records),
            "rate": round(n/len(records), 4), "wilson95": wilson(n, len(records)),
            "p50_success_s": data["obstacles"]["beam"]["aggregate"]["p50_completion_time_s"],
            "p95_success_s": data["obstacles"]["beam"]["aggregate"]["p95_completion_time_s"],
        }
    if any(x["spawn"] != y["spawn"] for x, y in zip(mj_rows["baseline"], mj_rows["candidate"])):
        raise ValueError("MuJoCo initial poses are not paired")
    mj["initial_spawns_verified_identical"] = True
    mj["paired"] = {
        "baseline_only": sum(x["success"] and not y["success"]
                             for x, y in zip(mj_rows["baseline"], mj_rows["candidate"])),
        "candidate_only": sum(y["success"] and not x["success"]
                              for x, y in zip(mj_rows["baseline"], mj_rows["candidate"])),
    }
    candidate = isaac_summary["candidate"]["rate"]
    cross_gap = round(abs(candidate - mj["candidate"]["rate"]), 4)
    out = {
        "kind": "beam_stage1_exploratory_independent_isaac",
        "geometry_m": {"width": 0.80, "length": 1.50, "height": 0.03},
        "training_source": "059950, 25-iteration smoke then 200-iteration short run",
        "candidate_checkpoint": "060173",
        "isaac_heightfield": isaac_summary,
        "mujoco_box": mj,
        "cross_engine_rate_gap": cross_gap,
        "cross_engine_initial_states_identical": False,
        "base_gait_regression_200_complete": False,
        "promotion_to_next_width_allowed": False,
        "reason": "Isaac GPU reset states were not exactly paired, cross-engine initial states are not explicitly aligned, and base gait regression is incomplete. Do not promote on training reward alone.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"isaac": isaac_summary["unpaired_rate_difference"],
                      "isaac_rates": {k: isaac_summary[k]["rate"] for k in ("baseline","candidate")},
                      "mujoco_rates": {k: mj[k]["rate"] for k in ("baseline","candidate")},
                      "cross_engine_rate_gap": cross_gap}, indent=2))
    print("saved:", a.out)


if __name__ == "__main__":
    main()
