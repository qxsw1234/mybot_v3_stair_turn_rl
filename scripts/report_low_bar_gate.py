#!/usr/bin/env python3
"""Report and verdict for an ``eval_low_bar_isaac.py`` result file.

Prints the protocol, per-candidate rates with their Wilson 95% intervals, the
per-clearance groups and the paired comparisons, then applies the M1.1 gate:
a candidate only counts as an improvement when its paired delta against the
in-process baseline has a 95% interval that excludes zero.
"""

import argparse
import json
import sys


def _fmt_ci(block, key="success_rate_ci95_percent"):
    interval = block.get(key)
    return "n/a" if interval is None else f"[{interval[0]:.1f}, {interval[1]:.1f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", help="eval_low_bar_isaac.py JSON output")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Required lower CI bound, in percentage points.")
    args = parser.parse_args()

    with open(args.result, encoding="utf-8") as handle:
        data = json.load(handle)

    protocol = data.get("protocol", {})
    print("protocol")
    for key in ("profile", "num_envs", "seeds", "episodes_total_per_candidate",
                "reference", "lateral_init_half_range_m",
                "yaw_init_half_range_rad", "observation_noise",
                "competition_clearance_m", "git_commit", "git_dirty",
                "evaluator_sha256", "checkpoint_selection"):
        if key in protocol:
            print(f"  {key}: {protocol[key]}")
    if "clearance_by_row" in protocol:
        print(f"  clearance_by_row: {protocol['clearance_by_row']}")

    print("\ncandidates")
    for label, block in data.get("candidates", {}).items():
        print(f"  {label}: n={block['episodes']} "
              f"success={block['success_rate_percent']}% {_fmt_ci(block)} "
              f"outside_gate={block['outside_gate_rate_percent']}% "
              f"collision={block['collision_rate_percent']}% "
              f"fall={block['fall_rate_percent']}%")
        print(f"      mean_min_base_height_m={block['mean_min_base_height_m']} "
              f"mean_max_bar_force_n={block['mean_max_bar_force_n']}")
        for name, group in (block.get("groups") or {}).items():
            print(f"      {name:<18} n={group['episodes']:<5} "
                  f"success={group['success_rate_percent']:>5.1f}% "
                  f"{_fmt_ci(group)} "
                  f"outside={group['outside_gate_rate_percent']:>5.1f}% "
                  f"collision={group['collision_rate_percent']:>5.1f}%")
        for seed, seed_block in (block.get("per_seed") or {}).items():
            print(f"      seed {seed:<10} n={seed_block['episodes']:<5} "
                  f"success={seed_block['success_rate_percent']:>5.1f}% "
                  f"{_fmt_ci(seed_block)}")

    comparisons = data.get("paired_comparisons") or {}
    print("\npaired comparisons")
    if not comparisons:
        print("  none (need at least two candidates in one invocation)")
    verdicts = []
    for label, comparison in comparisons.items():
        low, high = comparison["success_delta_ci95_percent"]
        improved = low > max(0.0, args.min_delta)
        regressed = high < 0.0
        verdict = ("IMPROVED" if improved
                   else "REGRESSED" if regressed
                   else "no significant difference")
        verdicts.append((label, verdict, comparison))
        print(f"  {label} vs {comparison['reference']}: "
              f"delta={comparison['success_delta_percent']:+.2f}pp "
              f"CI95=[{low:+.2f}, {high:+.2f}] "
              f"p={comparison['mcnemar_exact_p']} "
              f"ref_only={comparison['reference_only_success']} "
              f"cand_only={comparison['candidate_only_success']} "
              f"both_fail={comparison['both_fail']} -> {verdict}")

    print("\ngate verdict")
    if not verdicts:
        print("  cannot decide: a paired comparison against an in-process "
              "baseline is required")
        return 1
    for label, verdict, _ in verdicts:
        print(f"  {label}: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
