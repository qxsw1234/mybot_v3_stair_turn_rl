"""Load and validate the single-source RoboCon 2027 obstacle rules."""

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_RULES_PATH = CONFIG_DIR / "rules.yaml"
DEFAULT_HARDWARE_PATH = CONFIG_DIR / "hardware_capability.yaml"

ORDINARY_OBSTACLES = (
    "slalom",
    "low_frame",
    "wall",
    "brick_wood_pit",
    "tire_array",
    "plum_posts",
    "beam",
)
ALL_OBSTACLES = ORDINARY_OBSTACLES[:3] + ("gully",) + ORDINARY_OBSTACLES[3:]
HARDWARE_STATUSES = {"unmeasured", "pass", "fail", "blocked"}


class RulesConfigError(ValueError):
    """Raised when a rules or hardware configuration is internally inconsistent."""


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise RulesConfigError(f"expected a mapping in {path}")
    return data


def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str], context: str) -> None:
    missing = sorted(set(keys) - set(mapping))
    if missing:
        raise RulesConfigError(f"{context} is missing keys: {missing}")


def load_rules(path: Path = DEFAULT_RULES_PATH) -> Dict[str, Any]:
    rules = _load_yaml(Path(path))
    _require_keys(rules, ("schema_version", "match", "scoring", "obstacles"), "rules")
    if rules["schema_version"] != 1:
        raise RulesConfigError("unsupported rules schema_version")
    if tuple(rules["obstacles"].keys()) != ALL_OBSTACLES:
        raise RulesConfigError(
            f"obstacles must be ordered exactly as {ALL_OBSTACLES}, "
            f"got {tuple(rules['obstacles'].keys())}"
        )
    if int(rules["match"]["obstacle_count"]) != len(ALL_OBSTACLES):
        raise RulesConfigError("match.obstacle_count does not match obstacle definitions")

    for mode in ("autonomous", "manual"):
        score = rules["scoring"][mode]
        computed = (
            len(ORDINARY_OBSTACLES) * int(score["ordinary_obstacle"])
            + int(score["gully_ascent"])
            + int(score["gully_gap"])
            + int(score["gully_descent"])
            + int(score["return_to_start"])
        )
        if computed != int(score["maximum"]):
            raise RulesConfigError(
                f"{mode} maximum is {score['maximum']}, but components sum to {computed}"
            )
    return rules


def load_hardware_capability(path: Path = DEFAULT_HARDWARE_PATH) -> Dict[str, Any]:
    hardware = _load_yaml(Path(path))
    _require_keys(hardware, ("schema_version", "capabilities", "high_risk_training_gate"), "hardware")
    if hardware["schema_version"] != 1:
        raise RulesConfigError("unsupported hardware schema_version")
    for name, capability in hardware["capabilities"].items():
        status = capability.get("status")
        if status not in HARDWARE_STATUSES:
            raise RulesConfigError(f"capability {name!r} has invalid status {status!r}")
    return hardware


def high_risk_gate_status(hardware: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    capabilities = hardware["capabilities"]
    result = {}
    for obstacle, required in hardware["high_risk_training_gate"]["required_capabilities"].items():
        statuses = {name: capabilities[name]["status"] for name in required}
        result[obstacle] = {
            "ready": all(status == "pass" for status in statuses.values()),
            "capabilities": statuses,
        }
    return result
