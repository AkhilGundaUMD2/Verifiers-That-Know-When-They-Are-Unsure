# Generate one checklist per task from processed JSONL.
import argparse
import json
from pathlib import Path

from loaders.splits import read_jsonl
from .checklist import MODEL, generate


def unique_tasks(records):
    # Every agent's attempt at a task shares its group_id and instruction.
    seen = set()
    for record in records:
        if record["group_id"] not in seen:
            seen.add(record["group_id"])
            yield record


def main():
    parser = argparse.ArgumentParser(description="Generate task checklists.")
    parser.add_argument("--input", required=True, type=Path, help="Processed JSONL from python -m loaders")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Only generate the first N new checklists")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Resume: skip tasks that already have a usable checklist.
    done = set()
    if args.output.exists():
        done = {row["group_id"] for row in read_jsonl(args.output) if row["items"] is not None}
    from openai import OpenAI
    client = OpenAI()
    total = 0
    with args.output.open("a", encoding="utf-8") as stream:
        for record in unique_tasks(read_jsonl(args.input)):
            if args.limit is not None and total >= args.limit:
                break
            if record["group_id"] in done:
                continue
            result = generate(record, client, args.model)
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            total += 1
            if total % 50 == 0:
                print(f"Generated {total} checklists", flush=True)
    print(json.dumps({"count": total, "skipped": len(done), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
