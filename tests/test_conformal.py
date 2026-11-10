import math
import random
import unittest

from verifier.conformal import (INF, build_tasks, crc_threshold, empirical_risk, evaluate,
                                group_of, hb_pvalue, ltt_threshold, score_of)


def tasks_from(rows):
    return {f"t{i}": runs for i, runs in enumerate(rows)}


class HelperTests(unittest.TestCase):
    def test_group_of_matches_loader_group_ids(self):
        self.assertEqual(group_of("webarena/agent-x/webarena.improved.12"), "agentrewardbench/webarena.12")

    def test_score_falls_back_to_status(self):
        self.assertEqual(score_of({"outcome": "unknown"}, "outcome"), 0.5)
        self.assertEqual(score_of({"outcome": "fail", "scores": {"outcome": 0.8}}, "outcome"), 0.8)

    def test_build_tasks_drops_unknown_labels_and_flags_bad(self):
        labels = {"b/a/t.1": {"success": True, "side_effect": True},
                  "b/c/t.1": {"success": None, "side_effect": False}}
        predictions = {"b/a/t.1": {"outcome": "pass", "side_effect": "pass"},
                       "b/c/t.1": {"outcome": "fail", "side_effect": "pass"}}
        self.assertEqual(build_tasks(labels, predictions, "outcome"),
                         {"agentrewardbench/t.1": [(1.0, False)]})
        self.assertEqual(build_tasks(labels, predictions, "side_effect"),
                         {"agentrewardbench/t.1": [(1.0, True), (1.0, False)]})

    def test_risk_is_task_mean_and_monotone(self):
        tasks = tasks_from([[(0.9, True), (0.2, False)], [(0.5, True)]])
        self.assertEqual(empirical_risk(tasks, INF), 0)
        self.assertEqual(empirical_risk(tasks, 0.9), 0.25)
        self.assertEqual(empirical_risk(tasks, 0.5), 0.75)

    def test_evaluate_counts_run_level_reviews(self):
        tasks = tasks_from([[(0.9, False), (0.2, False)]])
        self.assertEqual(evaluate(tasks, 0.5)["review_rate"], 0.5)


class ThresholdTests(unittest.TestCase):
    def test_crc_picks_most_permissive_feasible_lambda(self):
        # 19 good tasks at 0.9, one bad task at 0.1; n = 20.
        tasks = tasks_from([[(0.9, False)]] * 19 + [[(0.1, True)]])
        # Accepting everything costs 20/21 * 0.05 + 1/21 ~ 0.095.
        self.assertEqual(crc_threshold(tasks, 0.10), 0.1)
        self.assertEqual(crc_threshold(tasks, 0.06), 0.9)
        # 1/(n+1) > alpha: only escalating everything is safe.
        self.assertEqual(crc_threshold(tasks, 0.04), INF)

    def test_hb_pvalue_behaves(self):
        self.assertEqual(hb_pvalue(0.2, 100, 0.1), 1.0)
        self.assertLess(hb_pvalue(0.0, 100, 0.1), 0.001)
        self.assertGreater(hb_pvalue(0.09, 100, 0.1), hb_pvalue(0.01, 100, 0.1))

    def test_ltt_is_no_looser_than_crc_here(self):
        rng = random.Random(1)
        tasks = tasks_from([[(s := rng.random(), rng.random() > s)] for _ in range(300)])
        self.assertGreaterEqual(ltt_threshold(tasks, 0.1, 0.1), crc_threshold(tasks, 0.1))

    def test_crc_controls_expected_risk_on_synthetic_data(self):
        # Score s ~ U(0,1); a run is bad with probability 1 - s. True risk at
        # lambda is integral_lambda^1 (1-s) ds = (1-lambda)^2 / 2.
        rng = random.Random(0)
        alpha, risks = 0.05, []
        for _ in range(400):
            cal = tasks_from([[(s := rng.random(), rng.random() > s)] for _ in range(100)])
            lam = crc_threshold(cal, alpha)
            risks.append(0.0 if math.isinf(lam) else (1 - lam) ** 2 / 2)
        self.assertLessEqual(sum(risks) / len(risks), alpha)


if __name__ == "__main__":
    unittest.main()
