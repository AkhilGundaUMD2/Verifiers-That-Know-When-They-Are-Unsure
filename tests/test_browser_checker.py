import unittest

from baselines.browser_checker import (
    check_final_answer,
    check_looping,
    check_risky_actions,
    find_target,
    parse_calls,
    verify,
)


class BrowserCheckerTests(unittest.TestCase):
    def test_parse_calls_extracts_simple_and_nested_calls(self):
        self.assertEqual(
            parse_calls('click("12"); send_msg_to_user("Done")'),
            [("click", "12"), ("send_msg_to_user", "Done")],
        )

    def test_parse_calls_ignores_invalid_action(self):
        self.assertEqual(parse_calls("not valid python("), [])

    def test_find_target_matches_bid_without_matching_prefix(self):
        axtree = "[12] button Delete\n[123] button Keep"
        self.assertEqual(find_target(axtree, "12"), "[12] button Delete")
        self.assertIsNone(find_target(axtree, "1"))

    def test_final_answer_passes_when_agent_answers(self):
        result = check_final_answer({
            "steps": [{"action": 'send_msg_to_user("Finished")'}],
        })
        self.assertEqual(result["status"], "pass")
        self.assertIn("Finished", result["reason"])

    def test_infeasible_report_is_unknown(self):
        result = check_final_answer({
            "steps": [{"action": 'report_infeasible("Cannot access")'}],
        })
        self.assertEqual(result["status"], "unknown")

    def test_missing_final_answer_fails(self):
        result = check_final_answer({
            "steps": [],
            "metadata": {"summary_info": {"truncated": True}},
        })
        self.assertEqual(result["status"], "fail")
        self.assertIn("step limit", result["reason"])

    def test_looping_detects_repeated_action_on_same_url(self):
        step = {"url": "https://example.test", "action": 'click("1")'}
        result = check_looping({"steps": [step] * 4})
        self.assertEqual(result["status"], "fail")

    def test_looping_does_not_combine_different_urls(self):
        steps = [
            {"url": f"https://example.test/{i}", "action": 'click("1")'}
            for i in range(4)
        ]
        self.assertEqual(check_looping({"steps": steps})["status"], "pass")

    def test_unrequested_risky_click_fails(self):
        result = check_risky_actions({
            "instruction": "Find the order status.",
            "steps": [{
                "action": 'click("7")',
                "axtree": "[7] button Delete account",
            }],
        })
        self.assertEqual(result["status"], "fail")
        self.assertIn("Delete account", result["reason"])

    def test_requested_risky_click_passes(self):
        result = check_risky_actions({
            "instruction": "Delete the old draft.",
            "steps": [{
                "action": 'click("7")',
                "axtree": "[7] button Delete draft",
            }],
        })
        self.assertEqual(result["status"], "pass")

    def test_missing_click_target_is_unknown(self):
        result = check_risky_actions({
            "instruction": "Find the order status.",
            "steps": [{"action": 'click("7")', "axtree": "[8] button Keep"}],
        })
        self.assertEqual(result["status"], "unknown")

    def test_verify_aggregates_checks_into_expected_axes(self):
        result = verify({
            "id": "record-1",
            "instruction": "Find the order status.",
            "steps": [{
                "action": 'click("7")',
                "axtree": "[7] button Delete account",
                "url": "https://example.test",
            }],
        })
        self.assertEqual(result["verifier"], "rules")
        self.assertEqual(result["outcome"], "fail")
        self.assertEqual(result["side_effect"], "fail")
        self.assertEqual(result["looping"], "pass")


if __name__ == "__main__":
    unittest.main()
