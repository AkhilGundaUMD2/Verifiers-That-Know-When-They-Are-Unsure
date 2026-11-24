import unittest

from verifier.policy import action, build_tasks, calibrate, choose_policy, evaluate


class CostAwarePolicyTests(unittest.TestCase):
    def setUp(self):
        self.costs = {"false_accept": 10, "false_reject": 10, "review": 1}

    def test_selects_accept_and_reject_thresholds_to_avoid_review_cost(self):
        tasks = {
            "good": [(0.9, False, False)],
            "bad": [(0.1, True, False)],
        }
        thresholds = choose_policy(tasks, self.costs)
        result = evaluate(tasks, thresholds, self.costs)

        self.assertEqual(thresholds, {"low": 0.1, "high": 0.9})
        self.assertEqual(result["mean_cost"], 0)
        self.assertEqual(result["action_rates"], {
            "accept": 0.5, "reject": 0.5, "escalate": 0.0,
        })

    def test_out_of_scope_is_always_escalated(self):
        tasks = {
            "good": [(1.0, False, True)],
            "bad": [(0.0, True, False)],
        }
        thresholds = choose_policy(tasks, self.costs)

        self.assertEqual(action(1.0, thresholds, force_review=True), "escalate")
        self.assertEqual(evaluate(tasks, thresholds, self.costs)["action_counts"], {
            "accept": 0, "reject": 1, "escalate": 1,
        })

    def test_thresholds_route_middle_scores_to_review(self):
        thresholds = {"low": 0.2, "high": 0.8}
        self.assertEqual(action(0.2, thresholds), "reject")
        self.assertEqual(action(0.5, thresholds), "escalate")
        self.assertEqual(action(0.8, thresholds), "accept")

    def test_build_tasks_uses_correct_bad_label_and_drops_unknown(self):
        labels = {
            "b/a/task.1": {"success": False, "side_effect": False},
            "b/b/task.1": {"success": None, "side_effect": True},
        }
        predictions = {
            "b/a/task.1": {"outcome": "fail", "side_effect": "pass"},
            "b/b/task.1": {
                "outcome": "pass", "side_effect": "fail",
                "uncertainty": {"side_effect": "out_of_scope"},
            },
        }
        assignment = {record_id: "calibration" for record_id in labels}

        outcome = build_tasks(labels, predictions, assignment, "outcome")
        side_effect = build_tasks(labels, predictions, assignment, "side_effect")

        self.assertEqual(list(outcome["calibration"].values()), [[(0.0, True, False)]])
        self.assertEqual(list(side_effect["calibration"].values()), [
            [(1.0, False, False), (0.0, True, True)],
        ])

    def test_rejects_invalid_costs_and_missing_split_assignments(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            choose_policy(
                {"task": [(0.5, False, False)]},
                {"false_accept": -1, "false_reject": 1, "review": 1},
            )
        with self.assertRaisesRegex(ValueError, "Missing split assignment"):
            build_tasks(
                {"b/a/task.1": {"success": True, "side_effect": False}},
                {"b/a/task.1": {"outcome": "pass", "side_effect": "pass"}},
                {},
                "outcome",
            )

    def test_calibrate_fits_on_calibration_and_reports_held_out_splits(self):
        labels = {
            "b/a/task.1": {"success": True, "side_effect": False},
            "b/a/task.2": {"success": False, "side_effect": True},
            "b/a/task.3": {"success": True, "side_effect": False},
        }
        predictions = {
            "b/a/task.1": {"outcome": "pass", "side_effect": "pass", "verifier": "rules"},
            "b/a/task.2": {"outcome": "fail", "side_effect": "fail", "verifier": "rules"},
            "b/a/task.3": {"outcome": "unknown", "side_effect": "unknown", "verifier": "rules"},
        }
        assignment = {
            "b/a/task.1": "calibration",
            "b/a/task.2": "calibration",
            "b/a/task.3": "id_test",
        }

        report = calibrate(labels, predictions, assignment, "outcome", self.costs)

        self.assertEqual(report["thresholds"], {"low": 0.0, "high": 1.0})
        self.assertEqual(report["calibration"]["tasks"], 2)
        self.assertEqual(report["splits"]["id_test"]["action_counts"], {
            "accept": 0, "reject": 0, "escalate": 1,
        })


if __name__ == "__main__":
    unittest.main()
