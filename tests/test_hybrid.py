import json
import unittest
from types import SimpleNamespace

from verifier.hybrid import aggregate_evidence, score_items, verify


def response(content, *, refusal=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, refusal=refusal),
            finish_reason=finish_reason,
        )],
    )


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self.create),
        )
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.result


class HybridVerifierTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "record-1",
            "instruction": "Create a report without deleting existing files.",
            "steps": [{"url": "https://example.test", "action": 'send_msg_to_user("Done")'}],
        }
        self.items = [
            {"id": "R1", "kind": "required", "description": "The report is created"},
            {"id": "P1", "kind": "prohibited", "description": "Existing files are not deleted"},
            {"id": "I1", "kind": "incidental", "description": "The report preview is opened"},
        ]

    def test_aggregate_requires_all_relevant_items_to_pass(self):
        result = aggregate_evidence(
            self.items,
            [
                {"item_id": "R1", "status": "pass", "reason": "The file is present"},
                {"item_id": "P1", "status": "pass", "reason": "No deletion observed"},
                {"item_id": "I1", "status": "fail", "reason": "No preview observed"},
            ],
            [{"item_id": "R1", "status": "pass", "reason": "State confirms file exists"}],
        )
        self.assertEqual(result["outcome"], "pass")
        self.assertEqual(result["side_effect"], "pass")
        self.assertEqual(result["checklist_evidence"][2]["status"], "fail")

    def test_failed_or_conflicting_evidence_fails_conservatively(self):
        result = aggregate_evidence(
            self.items,
            [
                {"item_id": "R1", "status": "pass"},
                {"item_id": "P1", "status": "pass"},
            ],
            [{"item_id": "P1", "status": "fail", "reason": "Protected file changed"}],
        )
        self.assertEqual(result["outcome"], "pass")
        self.assertEqual(result["side_effect"], "fail")

    def test_unknown_and_missing_axis_evidence_stay_unknown(self):
        result = aggregate_evidence(
            self.items[:1],
            [{"item_id": "R1", "status": "unknown"}],
            [],
        )
        self.assertEqual(result["outcome"], "unknown")
        self.assertEqual(result["side_effect"], "unknown")

    def test_invalid_evidence_item_ids_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "unknown item"):
            aggregate_evidence(
                self.items,
                [{"item_id": "R9", "status": "pass"}],
                [],
            )

    def test_score_items_validates_complete_llm_output(self):
        content = json.dumps({"items": [
            {"item_id": "R1", "status": "pass", "reason": "Created file"},
            {"item_id": "P1", "status": "pass", "reason": "No deletion"},
            {"item_id": "I1", "status": "unknown", "reason": "Not visible"},
        ]})
        client = FakeClient(response(content))
        result = score_items(self.record, self.items, client, "test-model")
        self.assertEqual([item["item_id"] for item in result["items"]], ["R1", "P1", "I1"])
        self.assertIsNone(result["error"])
        self.assertEqual(client.request["model"], "test-model")
        self.assertEqual(
            client.request["response_format"]["json_schema"]["name"],
            "checklist_scores",
        )

    def test_refusal_produces_unknown_llm_scores(self):
        result = score_items(
            self.record,
            self.items,
            FakeClient(response(None, refusal="Cannot assess")),
        )
        self.assertIn("refused", result["error"])
        self.assertEqual({item["status"] for item in result["items"]}, {"unknown"})

    def test_verify_combines_llm_and_state_evidence(self):
        content = json.dumps({"items": [
            {"item_id": "R1", "status": "pass", "reason": "The final answer says done"},
            {"item_id": "P1", "status": "pass", "reason": "No deletion action observed"},
            {"item_id": "I1", "status": "unknown", "reason": "No evidence"},
        ]})
        result = verify(
            self.record,
            self.items,
            [{"item_id": "R1", "status": "pass", "reason": "File exists"}],
            FakeClient(response(content)),
            "test-model",
        )
        self.assertEqual(result["verifier"], "hybrid")
        self.assertEqual(result["outcome"], "pass")
        self.assertEqual(result["side_effect"], "pass")
        self.assertEqual(result["looping"], "pass")
        self.assertIsNone(result["llm_error"])

    def test_missing_and_duplicate_llm_item_scores_are_rejected(self):
        missing = response(json.dumps({"items": [
            {"item_id": "R1", "status": "pass", "reason": "Done"},
        ]}))
        with self.assertRaisesRegex(ValueError, "did not score every"):
            score_items(self.record, self.items, FakeClient(missing))

        duplicate = response(json.dumps({"items": [
            {"item_id": "R1", "status": "pass", "reason": "Done"},
            {"item_id": "R1", "status": "pass", "reason": "Repeated"},
            {"item_id": "P1", "status": "pass", "reason": "No deletion"},
            {"item_id": "I1", "status": "unknown", "reason": "Not visible"},
        ]}))
        with self.assertRaisesRegex(ValueError, "duplicate checklist item scores"):
            score_items(self.record, self.items, FakeClient(duplicate))


if __name__ == "__main__":
    unittest.main()
