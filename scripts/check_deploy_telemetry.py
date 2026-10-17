#!/usr/bin/env python3
"""Check short ROS2 deployment telemetry for names, values, stamps and gaps.

Run with the fake motor driver or MuJoCo bridge. The output is a small JSON
summary; it does not claim any hardware feedback or physical calibration.
"""
import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

TOPICS = ("/joint_states", "/policy_joint_targets")


class TelemetryCheck(Node):
    def __init__(self, expected_names):
        super().__init__("robocon_telemetry_check")
        self.expected_names = expected_names
        self.rows = {topic: [] for topic in TOPICS}
        self.subs = [self.create_subscription(
            JointState, topic, lambda msg, topic=topic: self.receive(topic, msg), 10)
            for topic in TOPICS]

    def receive(self, topic, msg):
        received_ns = time.time_ns()
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        expected = self.expected_names
        names = list(msg.name)
        values = list(msg.position) + list(msg.velocity) + list(msg.effort)
        self.rows[topic].append({
            "received_ns": received_ns,
            "stamp_ns": stamp_ns,
            "names_ok": names == expected,
            "position_count": len(msg.position),
            "velocity_count": len(msg.velocity),
            "effort_count": len(msg.effort),
            "finite": all(math.isfinite(value) for value in values),
            "age_ms": (received_ns - stamp_ns) / 1e6,
        })


def summarize(rows, topic):
    stamps = [row["stamp_ns"] for row in rows]
    gaps = [(b-a)/1e6 for a,b in zip(stamps, stamps[1:])]
    return {
        "messages": len(rows),
        "joint_names_valid": all(row["names_ok"] for row in rows),
        "positions_12": all(row["position_count"] == 12 for row in rows),
        "velocities": sorted(set(row["velocity_count"] for row in rows)),
        "effort_counts": sorted(set(row["effort_count"] for row in rows)),
        "all_finite": all(row["finite"] for row in rows),
        "positive_monotonic_stamps": all(x > 0 for x in stamps) and
            all(b > a for a,b in zip(stamps, stamps[1:])),
        "max_stamp_gap_ms": round(max(gaps), 3) if gaps else None,
        "max_abs_age_ms": round(max(abs(row["age_ms"]) for row in rows), 3) if rows else None,
        "missing": len(rows) == 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--names-json", type=Path, default=None,
                    help="JSON array of expected joint names; default is MyBot V3 policy order")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.duration <= 0:
        ap.error("duration must be positive")
    expected = (
        json.loads(a.names_json.read_text()) if a.names_json else
        [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
         for joint in ("hip", "thigh", "calf")]
    )
    if len(expected) != 12:
        ap.error("expected 12 joint names")
    rclpy.init()
    node = TelemetryCheck(expected)
    try:
        end = time.monotonic() + a.duration
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        result = {
            "duration_s": a.duration,
            "topics": {topic: summarize(node.rows[topic], topic) for topic in TOPICS},
            "note": "Header age uses the local wall clock; hardware sample age requires a source timestamp.",
        }
        node.destroy_node()
        rclpy.shutdown()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if any(block["missing"] or not block["joint_names_valid"] or
           not block["positions_12"] or not block["all_finite"] or
           not block["positive_monotonic_stamps"]
           for block in result["topics"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
