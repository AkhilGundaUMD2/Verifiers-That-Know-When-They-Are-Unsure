"""Copy each record's ID and human labels into a small JSONL.

Processed records carry full step histories, so reading them for labels alone
is slow. The scorer accepts this file in place of the processed JSONL.
"""
import argparse
import json
from pathlib import Path

from loaders.splits import read_jsonl


def label_rows(records):
    for record in records:
        yield {"id": record["id"], "labels": record["labels"]}


def main():
    parser = argparse.ArgumentParser(description="Extract human labels from processed records.")
    parser.add_argument("--input", required=True, type=Path, help="Processed JSONL from python -m loaders")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for row in label_rows(read_jsonl(args.input)):
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            total += 1
    print(json.dumps({"count": total, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
