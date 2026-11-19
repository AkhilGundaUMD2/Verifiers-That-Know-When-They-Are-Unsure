import json
import sys

from baselines.browser_checker import find_target, parse_calls
from baselines.llm_judge import ELEMENT_ACTIONS

number = int(sys.argv[1])
items_path = sys.argv[2] if len(sys.argv) > 2 else "out/items_v11.jsonl"
records = []
with open("data/processed/agentrewardbench.jsonl") as f:
    for line in f:
        records.append(json.loads(line))
        if len(records) == number:
            break
record = records[number - 1]
print("RECORD", number, "|", record["id"].split("/")[-2], "|", record["group_id"].split("/")[-1])
print("human labels:", record["labels"])
print("\nINSTRUCTION:", record["instruction"].strip())
print("\nTRAJECTORY (%d steps)" % len(record["steps"]))
for n, step in enumerate(record["steps"], start=1):
    action = (step.get("action") or "(final page, no action)").replace("\n", " ")
    print("%2d. %s" % (n, action[:140]))
    for name, argument in parse_calls(step.get("action")):
        if name in ELEMENT_ACTIONS and argument is not None:
            target = find_target(step.get("axtree"), argument)
            print("      on: %s" % (target[:100] if target else "(element not found in the page tree)"))
print("\nCHECKLIST")
for line in open("out/checklists_v4_clean.jsonl"):
    c = json.loads(line)
    if c["group_id"] == record["group_id"]:
        for item in c["items"] or []:
            print("  %s [%s] %s" % (item["id"], item["kind"], item["description"]))
print("\nSCORER VERDICTS")
for line in open(items_path):
    row = json.loads(line)
    if row["id"] == record["id"]:
        for item in row["items"] or []:
            print("  %s %s steps=%s" % (item["id"], item["status"].upper(), item["steps"]))
            print("     ", item["reason"][:220])
        if not row["items"]:
            print("  (no scores: %s)" % row["error"])
        break
else:
    print("  (not scored: no checklist for this task)")
