#!/usr/bin/env python3
"""Evaluate every phase-2 checkpoint on a fixed ascending-stair suite."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def summarize(path):
    data = json.loads(path.read_text())
    records = data["records"]
    success = []
    falls = []
    upright = []
    for record in records:
        _, _, fell, distance, elevation, max_rp, *_ = record
        success.append((not fell) and distance >= 2.0 and elevation >= 0.06)
        falls.append(bool(fell))
        upright.append((not fell) and max_rp < 0.7)
    count = len(records)
    success_pct = 100.0 * sum(success) / count
    fall_pct = 100.0 * sum(falls) / count
    upright_pct = 100.0 * sum(upright) / count
    return {
        "episodes": count,
        "success_pct": round(success_pct, 3),
        "fall_pct": round(fall_pct, 3),
        "upright_pct": round(upright_pct, 3),
        "score": round(success_pct - 1.5 * fall_pct, 3),
    }


def write_summary(rows, evaluation_dir):
    rows = sorted(rows, key=lambda row: row["iteration"])
    (evaluation_dir / "validation_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))
    with (evaluation_dir / "validation_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "iteration", "episodes", "success_pct", "fall_pct",
            "upright_pct", "score", "evaluated_at",
        ])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--start", type=int, default=59000)
    parser.add_argument("--end", type=int, default=62999)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    run = Path(args.run).resolve()
    checkpoint_dir = run / "checkpoints"
    evaluation_dir = run / "evaluations" / "fixed_upstairs"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    status_path = evaluation_dir / "monitor_status.json"
    rows = []
    summary_path = evaluation_dir / "validation_summary.json"
    if summary_path.exists():
        rows = json.loads(summary_path.read_text())
    completed = {int(row["iteration"]) for row in rows}

    while True:
        checkpoints = []
        for checkpoint in checkpoint_dir.glob("ac_weights_[0-9]*.pt"):
            match = re.fullmatch(r"ac_weights_(\d+)\.pt", checkpoint.name)
            if match:
                iteration = int(match.group(1))
                if iteration >= args.start and (iteration % 250 == 0 or iteration >= args.end):
                    checkpoints.append((iteration, checkpoint))

        for iteration, checkpoint in sorted(checkpoints):
            if iteration in completed:
                continue
            result_path = evaluation_dir / f"up_fixed_{iteration:06d}.json"
            log_path = evaluation_dir / f"up_fixed_{iteration:06d}.log"
            command = [
                sys.executable,
                str(run.parents[2] / "scripts" / "eval_stairs_cpu.py"),
                "--run", str(run), "--iteration", str(iteration),
                "--vx", "0.5", "--terrain-type", "2",
                "--num-envs", "24", "--episodes-per-env", "2",
                "--out", str(result_path),
            ]
            with log_path.open("w") as log_handle:
                result = subprocess.run(
                    command, stdout=log_handle, stderr=subprocess.STDOUT,
                    timeout=600)
            if result.returncode != 0 or not result_path.exists():
                status_path.write_text(json.dumps({
                    "state": "evaluation_failed", "iteration": iteration,
                    "returncode": result.returncode, "log": str(log_path),
                }, ensure_ascii=False, indent=2))
                time.sleep(args.poll_seconds)
                continue

            row = {"iteration": iteration, **summarize(result_path),
                   "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            rows.append(row)
            completed.add(iteration)
            write_summary(rows, evaluation_dir)

            best = max(rows, key=lambda item: (item["score"], item["success_pct"],
                                                -item["fall_pct"], item["iteration"]))
            best_target = checkpoint_dir / f"ac_weights_{best['iteration']:06d}.pt"
            best_link = checkpoint_dir / "ac_weights_best_fixed.pt"
            temporary_link = checkpoint_dir / ".ac_weights_best_fixed.tmp"
            temporary_link.unlink(missing_ok=True)
            temporary_link.symlink_to(best_target.name)
            os.replace(temporary_link, best_link)
            status_path.write_text(json.dumps({
                "state": "running", "last_evaluated": iteration,
                "best": best, "best_checkpoint": str(best_target),
            }, ensure_ascii=False, indent=2))

        alive = process_alive(args.training_pid)
        latest = max((iteration for iteration, _ in checkpoints), default=args.start - 1)
        if not alive:
            state = "complete" if latest >= args.end else "training_stopped_early"
            best = max(rows, key=lambda item: (item["score"], item["success_pct"],
                                                -item["fall_pct"], item["iteration"])) if rows else None
            status_path.write_text(json.dumps({
                "state": state, "last_checkpoint": latest, "best": best,
            }, ensure_ascii=False, indent=2))
            return 0 if state == "complete" else 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
