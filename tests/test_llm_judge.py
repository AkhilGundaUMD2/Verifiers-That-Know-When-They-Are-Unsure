import json
import unittest
from types import SimpleNamespace

from baselines.llm_judge import (
    MAX_TREE_CHARS,
    build_prompt,
    judge,
    unknown,
)


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


class LlmJudgeTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "id": "record-1",
            "instruction": "Find the answer.",
            "steps": [
                {"url": "https://example.test", "action": 'click("1")'},
                {"url": "https://example.test", "action": 'send_msg_to_user("42")',
                 "last_action_error": "Click failed\nextra details",
                 "axtree": "Final tree"},
            ],
        }

    def test_build_prompt_includes_instruction_actions_error_and_final_tree(self):
        prompt = build_prompt(self.record)
        self.assertIn("Instruction: Find the answer.", prompt)
        self.assertIn("1. [https://example.test] click", prompt)
        self.assertIn("error: Click failed", prompt)
        self.assertNotIn("extra details", prompt)
        self.assertIn("Final tree", prompt)

    def test_build_prompt_notes_missing_final_answer(self):
        record = {"instruction": "Task", "steps": [{"action": "click()"}]}
        self.assertIn(
            "stopped without sending a final answer",
            build_prompt(record),
        )

    def test_build_prompt_truncates_long_final_tree(self):
        record = {
            "instruction": "Task",
            "steps": [{"axtree": "x" * (MAX_TREE_CHARS + 10)}],
        }
        prompt = build_prompt(record)
        self.assertIn("... (truncated)", prompt)
        self.assertNotIn("x" * (MAX_TREE_CHARS + 1), prompt)

    def test_judge_maps_verdict_and_sends_expected_prompt(self):
        client = FakeClient(response(json.dumps({
            "reasoning": "Evidence supports completion.",
            "success": "yes",
            "side_effect": "no",
            "looping": "unsure",
            "confidence": 0.9,
        })))
        result = judge(self.record, client, "test-model")
        self.assertEqual(result["outcome"], "pass")
        self.assertEqual(result["side_effect"], "pass")
        self.assertEqual(result["looping"], "unknown")
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["model"], "test-model")
        self.assertIn("Instruction: Find the answer.", client.request["messages"][1]["content"])

    def test_refusal_returns_unknown_verdict(self):
        client = FakeClient(response(None, refusal="Cannot judge this."))
        result = judge(self.record, client, "test-model")
        self.assertEqual(result["outcome"], "unknown")
        self.assertEqual(result["side_effect"], "unknown")
        self.assertIn("Cannot judge", result["reasoning"])

    def test_incomplete_response_returns_unknown_verdict(self):
        client = FakeClient(response('{"partial": true}', finish_reason="length"))
        result = judge(self.record, client, "test-model")
        self.assertEqual(result["outcome"], "unknown")
        self.assertIn("Incomplete response", result["reasoning"])

    def test_unknown_helper_has_no_confidence(self):
        result = unknown(self.record, "test-model", "Unavailable")
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["reasoning"], "Unavailable")
        self.assertEqual(result["verifier"], "llm")


if __name__ == "__main__":
    unittest.main()
