#!/usr/bin/env python3
"""Validate each Phase-4 checkpoint in MuJoCo on complete staircases."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import yaml

HEIGHTS = (0.08, 0.10, 0.12)
SPEEDS = (0.30, 0.40)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_success(metrics, height, cfg):
    x, y, z = metrics["base_xyz"]
    stair_end = float(cfg["stairs_x_start"]) + int(cfg["stairs_num_steps"]) * float(cfg["stairs_tread_depth"])
    top_height = int(cfg["stairs_num_steps"]) * height
    return (
        not metrics["fell"]
        and x >= stair_end - 0.15
        and abs(y) <= 0.8
        and z >= top_height + 0.20
    )


def write_rows(rows, output_dir):
    rows = sorted(rows, key=lambda row: row["iteration"])
    (output_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        fields = ["iteration", "success_8cm", "success_10cm", "success_12cm", "overall_success", "score", "evaluated_at"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--start", type=int, default=59500)
    parser.add_argument("--end", type=int, default=60249)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    run = Path(args.run).resolve()
    project = run.parents[2]
    checkpoint_dir = run / "checkpoints"
    output_dir = run / "evaluations" / "sim2sim_stairs"
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "monitor_status.json"
    summary_path = output_dir / "summary.json"
    rows = json.loads(summary_path.read_text()) if summary_path.exists() else []
    completed = {int(row["iteration"]) for row in rows}
    python = sys.executable
    base_cfg = yaml.safe_load((project / "deploy_cpp/config/robots/mybot_v3_cse_sim.yaml").read_text())

    while True:
        available = []
        for weight in checkpoint_dir.glob("ac_weights_[0-9]*.pt"):
            match = re.fullmatch(r"ac_weights_(\d+)\.pt", weight.name)
            if not match:
                continue
            iteration = int(match.group(1))
            body = checkpoint_dir / f"body_{iteration:06d}.jit"
            adaptation = checkpoint_dir / f"adaptation_module_{iteration:06d}.jit"
            if iteration >= args.start and body.exists() and adaptation.exists():
                available.append((iteration, weight, body, adaptation))

        for iteration, weight, body, adaptation in sorted(available):
            if iteration in completed:
                continue
            records = []
            checkpoint_output = output_dir / f"checkpoint_{iteration:06d}"
            checkpoint_output.mkdir(exist_ok=True)
            for height in HEIGHTS:
                cfg = dict(base_cfg)
                cfg.update(terrain_mode="stairs", stairs_step_height=height, mujoco_xml_relpath=str(project / "deploy_cpp/robot/mybot_v3/xml/mybot_v3.xml"))
                config_path = checkpoint_output / f"config_{int(height * 100):02d}cm.yaml"
                config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                for speed in SPEEDS:
                    metrics_path = checkpoint_output / f"h{int(height * 100):02d}_v{int(speed * 100):02d}.json"
                    log_path = checkpoint_output / f"h{int(height * 100):02d}_v{int(speed * 100):02d}.log"
                    command = [
                        python, str(project / "scripts/sim2sim_mujoco.py"),
                        "--config", str(config_path),
                        "--body", str(body), "--adaptation", str(adaptation),
                        "--terrain-mode", "stairs", "--vx", str(speed),
                        "--duration", "12", "--headless", "--stop-on-fall",
                        "--metrics-json", str(metrics_path),
                    ]
                    with log_path.open("w") as log_handle:
                        result = subprocess.run(command, cwd=project, stdout=log_handle, stderr=subprocess.STDOUT, timeout=120)
                    if result.returncode != 0 or not metrics_path.exists():
                        status_path.write_text(json.dumps({
                            "state": "evaluation_failed", "iteration": iteration,
                            "height": height, "speed": speed, "log": str(log_path),
                        }, ensure_ascii=False, indent=2) + "\n")
                        records.append({"height": height, "speed": speed, "success": False, "error": True})
                        continue
                    metrics = json.loads(metrics_path.read_text())
                    records.append({
                        "height": height, "speed": speed,
                        "success": is_success(metrics, height, cfg),
                        "metrics": metrics,
                    })

            rates = {}
            for height in HEIGHTS:
                selected = [record for record in records if record["height"] == height]
                rates[height] = 100.0 * sum(record["success"] for record in selected) / len(selected)
            overall = sum(rates.values()) / len(rates)
            score = 0.2 * rates[0.08] + 0.3 * rates[0.10] + 0.5 * rates[0.12]
            row = {
                "iteration": iteration,
                "success_8cm": round(rates[0.08], 3),
                "success_10cm": round(rates[0.10], 3),
                "success_12cm": round(rates[0.12], 3),
                "overall_success": round(overall, 3),
                "score": round(score, 3),
                "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "records": records,
            }
            rows.append(row)
            completed.add(iteration)
            write_rows(rows, output_dir)

            best = max(rows, key=lambda item: (item["score"], item["success_12cm"], item["iteration"]))
            best_iteration = int(best["iteration"])
            links = {
                "ac_weights_best_sim2sim.pt": f"ac_weights_{best_iteration:06d}.pt",
                "body_best_sim2sim.jit": f"body_{best_iteration:06d}.jit",
                "adaptation_module_best_sim2sim.jit": f"adaptation_module_{best_iteration:06d}.jit",
            }
            for link_name, target_name in links.items():
                link = checkpoint_dir / link_name
                temporary = checkpoint_dir / f".{link_name}.tmp"
                temporary.unlink(missing_ok=True)
                temporary.symlink_to(target_name)
                os.replace(temporary, link)
            status_path.write_text(json.dumps({
                "state": "running", "last_evaluated": iteration,
                "best": {key: value for key, value in best.items() if key != "records"},
            }, ensure_ascii=False, indent=2) + "\n")

        latest = max((item[0] for item in available), default=args.start - 1)
        if not alive(args.training_pid):
            state = "complete" if latest >= args.end else "training_stopped_early"
            best = max(rows, key=lambda item: (item["score"], item["success_12cm"], item["iteration"])) if rows else None
            status_path.write_text(json.dumps({
                "state": state, "last_checkpoint": latest,
                "best": ({key: value for key, value in best.items() if key != "records"} if best else None),
            }, ensure_ascii=False, indent=2) + "\n")
            return 0 if state == "complete" else 2
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
