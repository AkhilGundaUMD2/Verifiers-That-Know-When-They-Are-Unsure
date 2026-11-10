# Run a baseline verifier over processed AgentRewardBench JSONL.
import argparse
import json
from itertools import islice
from pathlib import Path

from loaders.splits import read_jsonl
from .browser_checker import verify
from .llm_judge import MODEL, judge
from verifier.hybrid import verify as hybrid_verify


def main():
    parser = argparse.ArgumentParser(description="Run a verifier over processed trajectories.")
    parser.add_argument("verifier", choices=["rules", "llm", "hybrid"])
    parser.add_argument("--input", required=True, type=Path, help="Processed JSONL from python -m loaders")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Only verify the first N records")
    parser.add_argument("--model", default=MODEL, help="Model for the llm verifier")
    parser.add_argument("--checklists", type=Path,
                        help="Checklist JSONL from python -m verifier (required for hybrid)")
    parser.add_argument("--state-evidence", type=Path,
                        help="Per-record state predicate results JSONL (required for hybrid)")
    args = parser.parse_args()
    if args.verifier == "hybrid" and (args.checklists is None or args.state_evidence is None):
        parser.error("hybrid requires both --checklists and --state-evidence")

    checklists, state_evidence = {}, {}
    if args.verifier == "hybrid":
        for row in read_jsonl(args.checklists):
            group_id = row["group_id"]
            if group_id in checklists:
                raise ValueError(f"Duplicate checklist for task: {group_id}")
            if row.get("items") is None:
                raise ValueError(f"Checklist unavailable for {group_id}: {row.get('error')}")
            checklists[group_id] = row["items"]
        for row in read_jsonl(args.state_evidence):
            record_id = row["id"]
            if record_id in state_evidence:
                raise ValueError(f"Duplicate state evidence for record: {record_id}")
            if not isinstance(row.get("items"), list):
                raise ValueError(f"State evidence items must be a list for {record_id}")
            state_evidence[record_id] = row["items"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = None
    if args.verifier in {"llm", "hybrid"}:
        from openai import OpenAI
        client = OpenAI()
    total = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for record in islice(read_jsonl(args.input), args.limit):
            # OSWorld task definitions have no recorded steps to verify.
            if record.get("steps") is None:
                raise ValueError(f"Record has no trajectory: {record['id']}")
            if args.verifier == "rules":
                result = verify(record)
            elif args.verifier == "llm":
                result = judge(record, client, args.model)
            else:
                group_id = record["group_id"]
                if group_id not in checklists:
                    raise ValueError(f"No checklist found for task: {group_id}")
                if record["id"] not in state_evidence:
                    raise ValueError(f"No state evidence found for record: {record['id']}")
                result = hybrid_verify(
                    record,
                    checklists[group_id],
                    state_evidence[record["id"]],
                    client,
                    args.model,
                )
            # Flush each verdict so an interrupted LLM run keeps its progress.
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            total += 1
            if total % 50 == 0:
                print(f"Verified {total} records", flush=True)
    print(json.dumps({"verifier": args.verifier, "count": total, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
