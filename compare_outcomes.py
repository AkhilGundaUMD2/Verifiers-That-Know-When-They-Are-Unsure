import json
import sys
from collections import Counter


def aggregate(items, prefix):
    statuses = [i["status"] for i in (items or []) if i["id"].startswith(prefix)]
    if "fail" in statuses:
        return "fail"
    if not statuses or "unknown" in statuses:
        return "unknown"
    return "pass"


def load_rows(path):
    rows = {}
    for line in open(path):
        row = json.loads(line)
        rows[row["id"]] = row
    return rows


def outcomes(path, records):
    rows = load_rows(path)
    result = []
    for record in records:
        row = rows.get(record["id"])
        if row is None:
            result.append("unknown")
        elif "items" in row:
            result.append(aggregate(row["items"], "R"))
        else:
            result.append(row["outcome"])
    return result


def main(paths, limit=47):
    records = []
    with open("data/processed/agentrewardbench.jsonl") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) == limit:
                break
    all_outcomes = {}
    for path in paths:
        got = outcomes(path, records)
        all_outcomes[path] = got
        counts = Counter((r["labels"]["success"], o) for r, o in zip(records, got))
        false_accepts = counts[(False, "pass")]
        false_rejects = counts[(True, "fail")]
        reviews = sum(n for (h, o), n in counts.items() if o == "unknown")
        print("\n" + path)
        for key in sorted(counts, key=str):
            print("  human success=%s  verifier=%s: %d" % (key[0], key[1], counts[key]))
        print("  records=%d reviews=%d false_accepts=%d false_rejects=%d"
              % (len(records), reviews, false_accepts, false_rejects))
    for before, after in zip(paths, paths[1:]):
        print("\nchanged: %s -> %s" % (before, after))
        changed = False
        for number, record in enumerate(records, start=1):
            old, new = all_outcomes[before][number - 1], all_outcomes[after][number - 1]
            if old != new:
                changed = True
                print("  record %d (%s) human=%s: %s -> %s"
                      % (number, record["group_id"].split("/")[-1], record["labels"]["success"], old, new))
        if not changed:
            print("  none")


if __name__ == "__main__":
    main(sys.argv[1:])
