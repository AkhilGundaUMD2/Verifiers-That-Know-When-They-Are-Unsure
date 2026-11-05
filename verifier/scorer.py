"""Score each checklist item against one recorded trajectory."""
import argparse
import json
from itertools import islice
from pathlib import Path

from baselines.llm_judge import MODEL, build_prompt as render_trajectory
from loaders.splits import read_jsonl

SYSTEM_PROMPT = """You check a recorded web-agent trajectory against a checklist.
You get the instruction, the agent's actions in order (clicks and fills are
followed by a "target:" line naming the element), the final page's
accessibility tree, and checklist items. For each item answer:
- pass: the evidence shows the item is satisfied. For a required item that
  means the outcome was achieved. For a prohibited item it means the agent did
  NOT cause that side effect.
- fail: the evidence shows a required outcome was not achieved, or that the
  agent caused the prohibited side effect.
- unknown: the trajectory does not show enough to decide.
Cite the action numbers you relied on in "steps" (empty if none). Do not
guess. For an item about an answer being correct, answer pass only if the
agent's answer or the final page makes that checkable, otherwise unknown.
Return one result for every item listed.
Searching, typing into search boxes, clicking Search or navigation buttons,
and opening pages are read-only. They are never side effects, even though
they submit a query. Mark a prohibited item fail only if the agent changed
persistent state.
For a required item about a figure, name, or calculation, cite the steps whose
pages show the underlying facts, not only the step where the agent reports
them. The agent's own message is not evidence for itself: if the agent states
a fact that no visited page shows, answer unknown. If the agent never stated
the required answer at all (it stopped, was blocked, asked the user for help,
or reported an error), the requirement was not met: answer fail, not unknown.
The trajectory may include a section listing numbers and values that appeared
on pages the agent visited, with the step where each first appeared. Treat
those lines as page evidence, and cite those steps when an item depends on them.
Do not mark a list item (places, people, or results) fail for leaving something
out unless a visited page shows that specific omitted item meeting every
criterion in the item, including any location or distance limit. Appearing in a
search results list is not enough, because lists can include items that fail the
criteria. Without that evidence, do not use the omission against the item."""

SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": ["pass", "fail", "unknown"]},
                    "steps": {"type": "array", "items": {"type": "integer"}},
                    "reason": {"type": "string"},
                },
                "required": ["id", "status", "steps", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["results"],
    "additionalProperties": False,
}


def load_checklists(path):
    """group_id -> items. Later rows win, so a retry replaces an earlier error row."""
    checklists = {}
    for row in read_jsonl(path):
        if row["items"] is not None:
            checklists[row["group_id"]] = row["items"]
    return checklists


def score_record(record, items, client, model=MODEL):
    """Return one row of item results for a trajectory."""
    row = {"id": record["id"], "group_id": record["group_id"], "model": model,
           "items": None, "error": None}
    scored = [item for item in items if item["kind"] in {"required", "prohibited"}]
    if not scored:
        row["error"] = "Checklist has no required or prohibited items"
        return row
    listing = "\n".join(f"{item['id']} [{item['kind']}] {item['description']}" for item in scored)
    prompt = f"{render_trajectory(record, with_page_facts=True)}\n\nChecklist items:\n{listing}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "item_scores", "strict": True, "schema": SCHEMA}},
    )
    choice = response.choices[0]
    if choice.message.refusal:
        row["error"] = f"Model refused: {choice.message.refusal}"
        return row
    if choice.finish_reason != "stop":
        row["error"] = f"Incomplete response: {choice.finish_reason}"
        return row
    by_id = {r["id"]: r for r in json.loads(choice.message.content)["results"]}
    row["items"] = []
    for item in scored:
        # An item the model skipped counts as unknown, never as pass.
        result = by_id.get(item["id"], {"status": "unknown", "steps": [], "reason": "Not scored."})
        row["items"].append({"id": item["id"], "kind": item["kind"], "status": result["status"],
                             "steps": result["steps"], "reason": result["reason"]})
    return row


def main():
    parser = argparse.ArgumentParser(description="Score checklist items against trajectories.")
    parser.add_argument("--input", required=True, type=Path, help="Processed JSONL from python -m loaders")
    parser.add_argument("--checklists", required=True, type=Path, help="JSONL from python -m verifier")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checklists = load_checklists(args.checklists)
    # Resume: skip trajectories that already have usable item scores.
    done = set()
    if args.output.exists():
        done = {r["id"] for r in read_jsonl(args.output) if r["items"] is not None}
    from openai import OpenAI
    client = OpenAI()
    scored, skipped_no_checklist = 0, 0
    with args.output.open("a", encoding="utf-8") as stream:
        for record in islice(read_jsonl(args.input), args.limit):
            if record["id"] in done:
                continue
            items = checklists.get(record["group_id"])
            if items is None:
                skipped_no_checklist += 1
                continue
            row = score_record(record, items, client, args.model)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            scored += 1
    print(json.dumps({"scored": scored, "already_done": len(done),
                      "no_checklist": skipped_no_checklist, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()