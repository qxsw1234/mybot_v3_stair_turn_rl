"""RoboCon 2027 rule configuration and cross-engine evaluators."""

from .evaluator import AttemptResult, ObstacleAttempt, ScoreLedger, half_or_more_feet
from .rules import high_risk_gate_status, load_hardware_capability, load_rules

__all__ = [
    "AttemptResult",
    "ObstacleAttempt",
    "ScoreLedger",
    "half_or_more_feet",
    "high_risk_gate_status",
    "load_hardware_capability",
    "load_rules",
]
