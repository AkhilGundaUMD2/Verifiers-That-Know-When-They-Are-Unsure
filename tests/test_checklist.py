import json
import unittest
from types import SimpleNamespace

from verifier.__main__ import unique_tasks
from verifier.checklist import build_prompt, generate, number_items


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


class ChecklistTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "group_id": "agentrewardbench/webarena.1",
            "dataset": "agentrewardbench",
            "domain": "webarena",
            "instruction": "Close issue #4.\n",
            "steps": [{"url": "https://git.test", "action": 'click("1")'}],
        }

    def test_build_prompt_includes_instruction_domain_and_start_page(self):
        prompt = build_prompt(self.record)
        self.assertIn("Instruction: Close issue #4.", prompt)
        self.assertIn("Environment: webarena", prompt)
        self.assertIn("Starting page: https://git.test", prompt)

    def test_build_prompt_works_without_steps(self):
        record = dict(self.record, steps=None)
        self.assertNotIn("Starting page", build_prompt(record))

    def test_number_items_assigns_ids_per_kind_and_drops_blanks(self):
        items = number_items({
            "required": [{"description": "Issue #4 is closed"}, {"description": " "}],
            "prohibited": [{"description": "Other issues stay open"}],
            "incidental": [],
        })
        self.assertEqual(
            [(item["id"], item["kind"]) for item in items],
            [("R1", "required"), ("P1", "prohibited")],
        )

    def test_generate_returns_numbered_items(self):
        content = json.dumps({
            "required": [{"description": "Issue #4 is closed"}],
            "prohibited": [{"description": "No other issue is closed"}],
            "incidental": [{"description": "Viewing the issue list"}],
        })
        client = FakeClient(response(content))
        result = generate(self.record, client, model="test-model")
        self.assertEqual(result["group_id"], "agentrewardbench/webarena.1")
        self.assertEqual([item["id"] for item in result["items"]], ["R1", "P1", "I1"])
        self.assertIsNone(result["error"])
        self.assertEqual(client.request["model"], "test-model")
        self.assertEqual(client.request["response_format"]["json_schema"]["name"], "checklist")

    def test_generate_records_refusal_and_truncation(self):
        refused = generate(self.record, FakeClient(response(None, refusal="no")))
        self.assertIsNone(refused["items"])
        self.assertIn("refused", refused["error"])
        cut = generate(self.record, FakeClient(response("{", finish_reason="length")))
        self.assertIsNone(cut["items"])
        self.assertIn("length", cut["error"])

    def test_unique_tasks_keeps_first_record_per_group(self):
        records = [{"group_id": "a", "id": 1}, {"group_id": "a", "id": 2},
                   {"group_id": "b", "id": 3}]
        self.assertEqual([r["id"] for r in unique_tasks(records)], [1, 3])


if __name__ == "__main__":
    unittest.main()
