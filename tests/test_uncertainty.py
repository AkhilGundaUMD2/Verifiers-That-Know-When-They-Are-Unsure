import json
import unittest
from types import SimpleNamespace

from baselines.score import score
from verifier.hybrid import aggregate_evidence, score_items
from verifier.uncertainty import classify_axis, classify_item, strongest


def llm(status, cause="none"):
    return {"item_id": "R1", "status": status, "cause": cause, "reason": ""}


def state(status):
    return {"item_id": "R1", "status": status, "reason": ""}


class ClassifyItemTests(unittest.TestCase):
    def test_decided_item_has_no_uncertainty(self):
        self.assertIsNone(classify_item("pass", llm("pass"), []))
        self.assertIsNone(classify_item("fail", llm("fail"), [state("fail")]))

    def test_llm_cause_maps_to_type(self):
        self.assertEqual(classify_item("unknown", llm("unknown", "missing_evidence"), []), "epistemic")
        self.assertEqual(classify_item("unknown", llm("unknown", "ambiguous_criterion"), []), "aleatoric")
        self.assertEqual(classify_item("unknown", llm("unknown", "unverifiable"), []), "out_of_scope")

    def test_unknown_without_cause_or_llm_is_epistemic(self):
        self.assertEqual(classify_item("unknown", llm("unknown"), []), "epistemic")
        self.assertEqual(classify_item("unknown", {"item_id": "R1", "status": "unknown"}, []), "epistemic")
        self.assertEqual(classify_item("unknown", None, []), "epistemic")
        self.assertEqual(classify_item("unknown", None, [state("unknown")]), "epistemic")

    def test_conflicting_sources_are_epistemic_even_when_decided(self):
        self.assertEqual(classify_item("fail", llm("pass"), [state("fail")]), "epistemic")

    def test_least_reducible_type_wins(self):
        # An unverifiable LLM verdict outranks a state predicate that could not run.
        self.assertEqual(
            classify_item("unknown", llm("unknown", "unverifiable"), [state("unknown")]),
            "out_of_scope")
        self.assertEqual(strongest(["epistemic", None, "aleatoric"]), "aleatoric")
        self.assertIsNone(strongest([None]))


def row(status, uncertainty):
    return {"status": status, "uncertainty": uncertainty}


class ClassifyAxisTests(unittest.TestCase):
    def test_axis_without_items_is_out_of_scope(self):
        self.assertEqual(classify_axis("unknown", []), "out_of_scope")

    def test_pass_is_certain(self):
        self.assertIsNone(classify_axis("pass", [row("pass", None)]))

    def test_one_clean_failure_makes_fail_certain(self):
        self.assertIsNone(classify_axis("fail", [row("fail", "epistemic"), row("fail", None)]))

    def test_fail_resting_only_on_conflicts_is_epistemic(self):
        self.assertEqual(classify_axis("fail", [row("fail", "epistemic"), row("pass", None)]), "epistemic")

    def test_unknown_axis_takes_strongest_unknown_item(self):
        items = [row("pass", None), row("unknown", "epistemic"), row("unknown", "aleatoric")]
        self.assertEqual(classify_axis("unknown", items), "aleatoric")


ITEMS = [
    {"id": "R1", "kind": "required", "description": "The answer is sent"},
    {"id": "R2", "kind": "required", "description": "The answer names the cheapest gym"},
]


class HybridIntegrationTests(unittest.TestCase):
    def test_aggregate_reports_item_and_axis_types(self):
        result = aggregate_evidence(
            ITEMS,
            [{"item_id": "R1", "status": "pass", "cause": "none", "reason": ""},
             {"item_id": "R2", "status": "unknown", "cause": "unverifiable", "reason": ""}],
            [],
        )
        self.assertEqual(result["outcome"], "unknown")
        self.assertEqual(result["uncertainty"],
                         {"outcome": "out_of_scope", "side_effect": "out_of_scope"})
        self.assertEqual([i["uncertainty"] for i in result["checklist_evidence"]],
                         [None, "out_of_scope"])

    def test_invalid_cause_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cause"):
            aggregate_evidence(ITEMS, [{"item_id": "R1", "status": "unknown", "cause": "vibes"}], [])

    def test_model_failure_is_epistemic(self):
        refusal = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=None, refusal="no"), finish_reason="stop")])
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: refusal)))
        record = {"id": "r", "instruction": "x", "steps": []}
        scored = score_items(record, ITEMS, client)
        result = aggregate_evidence(ITEMS, scored["items"], [])
        self.assertEqual(result["uncertainty"]["outcome"], "epistemic")

    def test_schema_requires_cause(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            content = json.dumps({"items": [
                {"item_id": "R1", "status": "pass", "cause": "none", "reason": ""},
                {"item_id": "R2", "status": "unknown", "cause": "ambiguous_criterion", "reason": ""}]})
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=None), finish_reason="stop")])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        scored = score_items({"id": "r", "instruction": "x", "steps": []}, ITEMS, client)
        schema = captured["response_format"]["json_schema"]["schema"]["properties"]["items"]["items"]
        self.assertIn("cause", schema["required"])
        self.assertEqual(scored["items"][1]["cause"], "ambiguous_criterion")


class ScoreByTypeTests(unittest.TestCase):
    def test_reviews_are_split_by_type(self):
        labels = {"a": {"success": False, "side_effect": False, "looping": False},
                  "b": {"success": True, "side_effect": False, "looping": False},
                  "c": {"success": False, "side_effect": False, "looping": False}}
        predictions = {
            "a": {"outcome": "unknown", "side_effect": "pass", "looping": "pass",
                  "uncertainty": {"outcome": "aleatoric", "side_effect": None}},
            "b": {"outcome": "unknown", "side_effect": "pass", "looping": "pass",
                  "uncertainty": {"outcome": "aleatoric", "side_effect": None}},
            "c": {"outcome": "fail", "side_effect": "pass", "looping": "pass",
                  "uncertainty": {"outcome": None, "side_effect": None}},
        }
        report = score(labels, predictions)
        self.assertEqual(report["outcome"]["review_by_type"], {"aleatoric": {"count": 2, "bad": 1}})
        self.assertEqual(report["side_effect"]["review_by_type"], {})
        self.assertNotIn("review_by_type", report["looping"])

    def test_untyped_predictions_keep_old_report(self):
        labels = {"a": {"success": True, "side_effect": False, "looping": False}}
        predictions = {"a": {"outcome": "pass", "side_effect": "pass", "looping": "pass"}}
        self.assertNotIn("review_by_type", score(labels, predictions)["outcome"])


if __name__ == "__main__":
    unittest.main()
