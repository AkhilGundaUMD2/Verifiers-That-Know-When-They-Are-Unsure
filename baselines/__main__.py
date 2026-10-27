# Run a baseline verifier over processed AgentRewardBench JSONL.
import argparse
import json
from itertools import islice
from pathlib import Path

from loaders.splits import read_jsonl
from .browser_checker import verify
from .llm_judge import MODEL, judge


def main():
    parser = argparse.ArgumentParser(description="Run a baseline verifier over processed trajectories.")
    parser.add_argument("verifier", choices=["rules", "llm"])
    parser.add_argument("--input", required=True, type=Path, help="Processed JSONL from python -m loaders")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Only verify the first N records")
    parser.add_argument("--model", default=MODEL, help="Model for the llm verifier")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = None
    if args.verifier == "llm":
        from openai import OpenAI
        client = OpenAI()
    total = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for record in islice(read_jsonl(args.input), args.limit):
            # OSWorld task definitions have no recorded steps to verify.
            if record.get("steps") is None:
                raise ValueError(f"Record has no trajectory: {record['id']}")
            result = (verify(record) if args.verifier == "rules"
                      else judge(record, client, args.model))
            # Flush each verdict so an interrupted LLM run keeps its progress.
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            total += 1
            if total % 50 == 0:
                print(f"Verified {total} records", flush=True)
    print(json.dumps({"verifier": args.verifier, "count": total, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
