#!/usr/bin/env python3
"""Validate RoboCon 2027 rules, score totals, and hardware training gates."""

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from robodog_gym.envs.robocon_2027 import (
    high_risk_gate_status,
    load_hardware_capability,
    load_rules,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules", type=Path, default=None)
    parser.add_argument("--hardware", type=Path, default=None)
    args = parser.parse_args()

    rules = load_rules(args.rules) if args.rules else load_rules()
    hardware = load_hardware_capability(args.hardware) if args.hardware else load_hardware_capability()
    result = {
        "valid": True,
        "rulebook": rules["rulebook"],
        "duration_s": rules["match"]["duration_s"],
        "obstacle_count": len(rules["obstacles"]),
        "maximum_scores": {
            mode: values["maximum"] for mode, values in rules["scoring"].items()
        },
        "high_risk_training_gate": high_risk_gate_status(hardware),
        "clarifications_required": rules.get("clarifications_required", []),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
