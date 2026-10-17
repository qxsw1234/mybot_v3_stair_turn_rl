import unittest

from scripts.eval_robocon_2027_baselines import (
    aggregate_contact_snapshots,
    slalom_projected_passes,
)

from robodog_gym.envs.robocon_2027 import (
    ObstacleAttempt,
    ScoreLedger,
    half_or_more_feet,
    high_risk_gate_status,
    load_hardware_capability,
    load_rules,
)


class RulesTest(unittest.TestCase):
    def test_score_totals_match_rulebook(self):
        rules = load_rules()
        self.assertEqual(rules["scoring"]["autonomous"]["maximum"], 1500)
        self.assertEqual(rules["scoring"]["manual"]["maximum"], 1000)
        self.assertEqual(rules["obstacles"]["low_frame"]["inner_height_m"], 0.29)
        self.assertEqual(rules["obstacles"]["beam"]["width_m"], 0.40)
        self.assertEqual(rules["obstacles"]["slalom"]["pole_count"], 6)

    def test_half_or_more_threshold_for_quadruped(self):
        self.assertFalse(half_or_more_feet(4, ["FL"]))
        self.assertTrue(half_or_more_feet(4, ["FL", "FR"]))

    def test_hardware_gates_start_closed(self):
        gates = high_risk_gate_status(load_hardware_capability())
        self.assertFalse(gates["wall"]["ready"])
        self.assertFalse(gates["gully"]["ready"])
        self.assertFalse(gates["low_frame"]["ready"])


class EvaluatorTest(unittest.TestCase):
    def test_low_frame_light_contact_is_allowed(self):
        attempt = ObstacleAttempt("low_frame")
        attempt.record_frame_contact(impulse_ns=0.2)
        attempt.mark("crossed")
        result = attempt.finalize()
        self.assertTrue(result.passed)
        self.assertEqual(result.metrics["frame_contacts"], 1)

    def test_low_frame_displacement_fails(self):
        attempt = ObstacleAttempt("low_frame")
        attempt.mark("crossed")
        attempt.mark("frame_displacement")
        self.assertEqual(attempt.finalize().reason, "frame_displacement")

    def test_low_frame_outside_gate_fails_even_if_crossed(self):
        attempt = ObstacleAttempt("low_frame")
        attempt.mark("outside_gate")
        attempt.mark("crossed")
        result = attempt.finalize()
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "outside_gate")

    def test_low_frame_fall_before_crossing_reports_fell(self):
        attempt = ObstacleAttempt("low_frame")
        attempt.mark("fell")
        result = attempt.finalize()
        self.assertFalse(result.passed)
        self.assertEqual(result.reason, "fell")

    def test_low_frame_completion_is_not_undone_by_later_free_locomotion(self):
        attempt = ObstacleAttempt("low_frame")
        attempt.mark("crossed")
        # Once the obstacle attempt is complete, a later fall outside the
        # frame belongs to subsequent locomotion rather than this attempt.
        attempt.mark("fell")
        self.assertTrue(attempt.finalize().passed)

    def test_slalom_order_and_alternating_sides(self):
        attempt = ObstacleAttempt("slalom")
        attempt.reach_zone("start")
        for index in range(1, 7):
            if index == 5:
                attempt.reach_zone("corner")
            attempt.pass_slalom_pole(
                index, "left" if index % 2 else "right", 0.39 if index % 2 else -0.39)
        attempt.reach_zone("finish")
        attempt.mark("crossed")
        self.assertTrue(attempt.finalize().passed)
        self.assertEqual(len(attempt.finalize().metrics["slalom_events"]), 9)

    def test_slalom_skipped_pole_and_early_corner_fail(self):
        attempt = ObstacleAttempt("slalom")
        attempt.reach_zone("start")
        attempt.pass_slalom_pole(1, "left", 0.39)
        attempt.pass_slalom_pole(3, "left", 0.39)
        attempt.reach_zone("corner")
        attempt.mark("crossed")
        self.assertEqual(attempt.finalize().reason, "slalom_sequence_error")

    def test_slalom_wrong_side_and_reset(self):
        attempt = ObstacleAttempt("slalom")
        attempt.reach_zone("start")
        attempt.pass_slalom_pole(1, "right", -0.39)
        self.assertEqual(attempt.finalize().reason, "wrong_pole_side")
        fresh = ObstacleAttempt("slalom")
        self.assertEqual(fresh.slalom_events, [])
        self.assertEqual(fresh.flags, set())

    def test_slalom_contact_overrides_completed_path(self):
        attempt = ObstacleAttempt("slalom")
        attempt.reach_zone("start")
        for index in range(1, 7):
            if index == 5:
                attempt.reach_zone("corner")
            attempt.pass_slalom_pole(
                index, "left" if index % 2 else "right", 0.39 if index % 2 else -0.39)
        attempt.reach_zone("finish")
        attempt.mark("crossed")
        attempt.mark("pole_contact")
        self.assertEqual(attempt.finalize().reason, "pole_contact")

    def test_short_substep_pole_contact_is_retained(self):
        def snapshot(bodies, floor_feet):
            return {
                "floor_feet": set(floor_feet),
                "surface_feet": set(),
                "surface_contact_points": [],
                "obstacle_bodies": set(bodies),
                "obstacle_geoms": {"pole_1"} if bodies else set(),
                "max_obstacle_force_n": 12.0 if bodies else 0.0,
                "total_obstacle_force_n": 12.0 if bodies else 0.0,
                "max_contacting_robot_top_z": 0.0,
            }
        snapshots = [
            snapshot([], ["FL"]),
            snapshot(["FR_foot"], ["FL", "FR"]),
            snapshot([], ["FR"]),
            snapshot([], []),
        ]
        combined = aggregate_contact_snapshots(snapshots, 0.005)
        self.assertIn("FR_foot", combined["obstacle_bodies"])
        self.assertEqual(combined["max_simultaneous_floor_feet"], {"FL", "FR"})
        self.assertAlmostEqual(combined["obstacle_impulse_ns"], 0.06)

    def test_slalom_projection_catches_skipped_and_wrong_side(self):
        projected = slalom_projected_passes((-0.1, -0.02), (1.0, -0.02))
        self.assertEqual([item[0] for item in projected], [1, 2])
        self.assertEqual(projected[0][1], "center")
        self.assertEqual(projected[1][1], "center")

    def test_tire_hole_is_not_rule_failure(self):
        attempt = ObstacleAttempt("tire_array")
        for event in ("foot_in_hole", "entered_short_side", "exited_short_side", "crossed"):
            attempt.mark(event)
        self.assertTrue(attempt.finalize().passed)

    def test_tire_two_outside_feet_fail(self):
        attempt = ObstacleAttempt("tire_array")
        for event in ("entered_short_side", "exited_short_side", "crossed"):
            attempt.mark(event)
        attempt.record_outside_ground_contacts(["FL", "FR"])
        self.assertEqual(attempt.finalize().reason, "half_or_more_feet_on_ground")

    def test_plum_post_contacts_are_idempotent(self):
        attempt = ObstacleAttempt("plum_posts", required_surfaces=("p1", "p2"))
        self.assertTrue(attempt.touch_surface("p1"))
        self.assertFalse(attempt.touch_surface("p1"))
        attempt.mark("crossed")
        self.assertEqual(attempt.finalize().reason, "missing_required_top_contacts")
        attempt.touch_surface("p2")
        self.assertTrue(attempt.finalize().passed)

    def test_gully_partial_scores_and_maximum(self):
        rules = load_rules()
        ledger = ScoreLedger("autonomous", rules)
        attempt = ObstacleAttempt("gully")
        attempt.mark("gully_ascent")
        ledger.award(attempt.finalize())
        self.assertEqual(ledger.total, 75)

        attempt.mark("gully_gap")
        attempt.mark("gully_descent")
        ledger.award(attempt.finalize())
        self.assertEqual(ledger.total, 300)

        for obstacle in (
            "slalom", "low_frame", "wall", "brick_wood_pit", "tire_array", "plum_posts", "beam"
        ):
            ledger.completed.add(obstacle)
        ledger.mark_returned_to_start()
        self.assertEqual(ledger.total, 1500)

    def test_gully_failed_attempt_requires_official_clarification(self):
        attempt = ObstacleAttempt("gully")
        attempt.mark("gully_ascent")
        attempt.mark("fell")
        result = attempt.finalize()
        with self.assertRaises(RuntimeError):
            ScoreLedger("autonomous").award(result)

        conservative = ScoreLedger(
            "autonomous", retain_gully_subscores_on_failed_attempt=False
        )
        conservative.award(result)
        self.assertEqual(conservative.total, 0)


if __name__ == "__main__":
    unittest.main()
