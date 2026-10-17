"""Rule-level event ledger and scoring for RoboCon 2027 obstacle attempts.

The evaluator deliberately consumes semantic events instead of simulator contact
objects.  Isaac Gym, MuJoCo and real-robot log adapters can therefore share the
same rule decisions without sharing a physics-engine API.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from .rules import ORDINARY_OBSTACLES, load_rules


GULLY_SUBTASKS = ("gully_ascent", "gully_gap", "gully_descent")
SLALOM_SEQUENCE = (
    "start", "pole_1", "pole_2", "pole_3", "pole_4",
    "corner", "pole_5", "pole_6", "finish",
)


def half_or_more_feet(total_feet: int, contact_feet: Iterable[str]) -> bool:
    """Return the exact rule threshold; for a quadruped, two feet is failure."""
    unique = set(contact_feet)
    if total_feet <= 0:
        raise ValueError("total_feet must be positive")
    if len(unique) > total_feet:
        raise ValueError("contact feet cannot exceed total feet")
    return 2 * len(unique) >= total_feet


@dataclass(frozen=True)
class AttemptResult:
    obstacle: str
    passed: bool
    failed: bool
    reason: str
    completed_subtasks: Tuple[str, ...] = ()
    metrics: Dict[str, Any] = field(default_factory=dict)


class ObstacleAttempt:
    """Collect one-shot contacts and rule events for one obstacle attempt."""

    def __init__(self, obstacle: str, required_surfaces: Iterable[str] = (), total_feet: int = 4):
        if obstacle not in load_rules()["obstacles"]:
            raise ValueError(f"unknown obstacle: {obstacle}")
        self.obstacle = obstacle
        self.total_feet = int(total_feet)
        self.required_surfaces = set(required_surfaces)
        self.touched_once: Set[str] = set()
        self.reached_zones: Set[str] = set()
        self.slalom_events = []
        self.slalom_pole_offsets_m = {}
        self.flags: Set[str] = set()
        self.outside_ground_feet: Set[str] = set()
        self.frame_contacts = 0
        self.max_frame_impulse_ns = 0.0
        self.elapsed_s: Optional[float] = None

    def touch_surface(self, surface: str) -> bool:
        """Record an idempotent contact and return True only on first contact."""
        first = surface not in self.touched_once
        self.touched_once.add(surface)
        return first

    def reach_zone(self, zone: str) -> bool:
        if self.obstacle == "slalom":
            if zone not in {"start", "corner", "finish"}:
                raise ValueError(f"unknown slalom zone: {zone}")
            return self._slalom_event(zone)
        first = zone not in self.reached_zones
        self.reached_zones.add(zone)
        return first

    def _slalom_event(self, event: str) -> bool:
        if event in self.slalom_events:
            return False
        next_index = len(self.slalom_events)
        if next_index >= len(SLALOM_SEQUENCE) or event != SLALOM_SEQUENCE[next_index]:
            self.flags.add("slalom_sequence_error")
            return False
        self.slalom_events.append(event)
        if event in {"start", "corner", "finish"}:
            self.reached_zones.add(event)
        return True

    def pass_slalom_pole(self, index: int, side: str, offset_m: float) -> bool:
        """Record a projected centre-line crossing and its signed clearance."""
        if self.obstacle != "slalom" or index not in range(1, 7):
            raise ValueError("slalom pole index must be 1..6")
        if side not in {"left", "right", "center"}:
            raise ValueError("slalom side must be left, right, or center")
        event = f"pole_{index}"
        if event in self.slalom_events:
            return False
        expected = "left" if index % 2 else "right"
        if side != expected:
            self.flags.add("wrong_pole_side")
        accepted = self._slalom_event(event)
        if accepted:
            self.slalom_pole_offsets_m[event] = float(offset_m)
        return accepted

    def mark(self, event: str) -> None:
        self.flags.add(event)

    def record_frame_contact(self, impulse_ns: float = 0.0) -> None:
        """Measure allowed contact without turning it into a rule failure."""
        self.frame_contacts += 1
        self.max_frame_impulse_ns = max(self.max_frame_impulse_ns, float(impulse_ns))

    def record_outside_ground_contacts(self, feet: Iterable[str]) -> None:
        self.outside_ground_feet = set(feet)

    @property
    def outside_ground_failure(self) -> bool:
        return half_or_more_feet(self.total_feet, self.outside_ground_feet)

    def _metrics(self) -> Dict[str, Any]:
        return {
            "elapsed_s": self.elapsed_s,
            "frame_contacts": self.frame_contacts,
            "max_frame_impulse_ns": self.max_frame_impulse_ns,
            "outside_ground_feet": sorted(self.outside_ground_feet),
            "touched_once": sorted(self.touched_once),
            "reached_zones": sorted(self.reached_zones),
            "slalom_events": list(self.slalom_events),
            "slalom_pole_offsets_m": dict(self.slalom_pole_offsets_m),
        }

    def finalize(self) -> AttemptResult:
        if self.obstacle == "slalom":
            return self._finalize_slalom()
        if self.obstacle == "low_frame":
            return self._finalize_low_frame()
        if self.obstacle == "gully":
            return self._finalize_gully()
        if self.obstacle in {"brick_wood_pit", "tire_array", "plum_posts", "beam"}:
            return self._finalize_complete_only()
        passed = "crossed" in self.flags and "fell" not in self.flags
        return AttemptResult(
            self.obstacle, passed, not passed, "success" if passed else "not_crossed", metrics=self._metrics()
        )

    def _finalize_slalom(self) -> AttemptResult:
        required = set(load_rules()["obstacles"]["slalom"]["required_zones"])
        if "pole_contact" in self.flags:
            reason = "pole_contact"
        elif "pole_overflight" in self.flags:
            reason = "pole_overflight"
        elif "wrong_pole_side" in self.flags:
            reason = "wrong_pole_side"
        elif "slalom_sequence_error" in self.flags:
            reason = "slalom_sequence_error"
        elif len(self.slalom_events) != len(SLALOM_SEQUENCE):
            reason = "missing_or_unordered_pole_pass"
        elif not required.issubset(self.reached_zones):
            reason = "missing_required_zones"
        elif "crossed" not in self.flags:
            reason = "not_crossed"
        else:
            reason = "success"
        passed = reason == "success"
        return AttemptResult(self.obstacle, passed, not passed, reason, metrics=self._metrics())

    def _finalize_low_frame(self) -> AttemptResult:
        if "violent_collision" in self.flags:
            reason = "violent_collision"
        elif "frame_displacement" in self.flags:
            reason = "frame_displacement"
        elif "outside_gate" in self.flags:
            reason = "outside_gate"
        elif "crossed" in self.flags:
            reason = "success"
        elif "fell" in self.flags:
            reason = "fell"
        else:
            reason = "not_crossed"
        passed = reason == "success"
        return AttemptResult(self.obstacle, passed, not passed, reason, metrics=self._metrics())

    def _finalize_gully(self) -> AttemptResult:
        completed = tuple(name for name in GULLY_SUBTASKS if name in self.flags)
        if self.outside_ground_failure:
            reason = "half_or_more_feet_on_ground"
        elif "fell" in self.flags:
            reason = "fell"
        elif set(GULLY_SUBTASKS).issubset(self.flags):
            reason = "success"
        else:
            reason = "partial"
        passed = reason == "success"
        failed = reason in {"half_or_more_feet_on_ground", "fell"}
        return AttemptResult(self.obstacle, passed, failed, reason, completed, self._metrics())

    def _finalize_complete_only(self) -> AttemptResult:
        if self.outside_ground_failure:
            reason = "half_or_more_feet_on_ground"
        elif "side_fall" in self.flags or "fell" in self.flags:
            reason = "side_fall"
        elif self.obstacle in {"brick_wood_pit", "tire_array", "beam"} and (
            "entered_short_side" not in self.flags or "exited_short_side" not in self.flags
        ):
            reason = "invalid_entry_or_exit"
        elif self.obstacle == "plum_posts" and not self.required_surfaces.issubset(self.touched_once):
            reason = "missing_required_top_contacts"
        elif "crossed" not in self.flags:
            reason = "not_crossed"
        else:
            reason = "success"
        passed = reason == "success"
        return AttemptResult(self.obstacle, passed, not passed, reason, metrics=self._metrics())


class ScoreLedger:
    """Idempotent match score ledger for one fixed control mode."""

    def __init__(
        self,
        control_mode: str,
        rules: Optional[Dict[str, Any]] = None,
        retain_gully_subscores_on_failed_attempt: Optional[bool] = None,
    ):
        self.rules = rules or load_rules()
        if control_mode not in self.rules["scoring"]:
            raise ValueError("control_mode must be 'autonomous' or 'manual'")
        self.control_mode = control_mode
        self.retain_gully_subscores_on_failed_attempt = retain_gully_subscores_on_failed_attempt
        self.completed: Set[str] = set()
        self.returned_to_start = False
        self.penalty_points = 0

    def award(self, result: AttemptResult) -> None:
        if result.obstacle == "gully":
            if result.failed and result.completed_subtasks:
                if self.retain_gully_subscores_on_failed_attempt is None:
                    raise RuntimeError(
                        "official clarification required: whether completed gully "
                        "subscores survive a failed attempt"
                    )
                if not self.retain_gully_subscores_on_failed_attempt:
                    return
            self.completed.update(result.completed_subtasks)
        elif result.passed and result.obstacle in ORDINARY_OBSTACLES:
            self.completed.add(result.obstacle)

    def mark_returned_to_start(self) -> None:
        if set(ORDINARY_OBSTACLES).issubset(self.completed) and set(GULLY_SUBTASKS).issubset(self.completed):
            self.returned_to_start = True

    def add_penalty(self, points: int) -> None:
        if points < 0:
            raise ValueError("penalty points must be non-negative")
        self.penalty_points += int(points)

    @property
    def total(self) -> int:
        scoring = self.rules["scoring"][self.control_mode]
        ordinary = len(set(ORDINARY_OBSTACLES) & self.completed) * int(scoring["ordinary_obstacle"])
        gully = sum(int(scoring[name]) for name in GULLY_SUBTASKS if name in self.completed)
        returned = int(scoring["return_to_start"]) if self.returned_to_start else 0
        return max(0, ordinary + gully + returned - self.penalty_points)
