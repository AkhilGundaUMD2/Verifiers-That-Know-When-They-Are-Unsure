import json
import re
import sys
from collections import Counter

QUOTED = [r"(?<![A-Za-z])'([^']{3,60})'(?![A-Za-z])", r'"([^"]{3,60})"', "\u201c([^\u201d]{3,60})\u201d"]
NAMES = r"\b[A-Z][A-Za-z&]+(?: [A-Z][A-Za-z&]+)+\b"
NUMBERS = r"\d[\d,]*\.?\d*"


def norm(text):
    return re.sub(r"\s+", " ", (text or "").lower().replace(",", ""))


def claims(reason):
    found = []
    for pattern in QUOTED:
        found += re.findall(pattern, reason)
    found += re.findall(NAMES, reason)
    found += [n for n in re.findall(NUMBERS, reason) if len(n.rstrip(".,")) >= 2]
    return list(dict.fromkeys(c.strip(".") for c in found))


def step_text(steps, numbers):
    parts = []
    for n in numbers:
        if 1 <= n <= len(steps):
            s = steps[n - 1]
            parts += [s.get("action"), s.get("url"), s.get("axtree")]
    return norm(" ".join(p for p in parts if p))


def main(items_path, data_path="data/processed/agentrewardbench.jsonl", limit=47):
    records = {}
    with open(data_path) as f:
        for line in f:
            r = json.loads(line)
            records[r["id"]] = r
            if len(records) == limit:
                break
    rows = {}
    for line in open(items_path):
        row = json.loads(line)
        rows[row["id"]] = row
    totals, flagged = Counter(), []
    for rid, row in rows.items():
        record = records.get(rid)
        if not record or not row["items"]:
            continue
        steps = record["steps"]
        everything = norm(record["instruction"]) + " " + step_text(steps, range(1, len(steps) + 1))
        for item in row["items"]:
            if item["status"] not in ("pass", "fail"):
                continue
            totals["items"] += 1
            cited = item["steps"]
            bad = [n for n in cited if not 1 <= n <= len(steps)]
            if bad:
                totals["out_of_range"] += 1
                flagged.append((rid, item, "step out of range: %s" % bad))
            if not cited:
                totals["uncited"] += 1
            evidence = step_text(steps, cited)
            for claim in claims(item["reason"]):
                key = norm(claim)
                if not key:
                    continue
                totals["claims"] += 1
                if key in evidence:
                    totals["in_cited_steps"] += 1
                elif key in everything:
                    totals["elsewhere_only"] += 1
                    flagged.append((rid, item, "not in cited steps, found elsewhere: %r" % claim))
                else:
                    totals["nowhere"] += 1
                    flagged.append((rid, item, "not in the trajectory at all: %r" % claim))
    print(json.dumps(dict(totals), indent=2))
    for rid, item, why in flagged:
        print("\n%s  %s [%s] %s steps=%s\n  %s\n  reason: %s"
              % (rid[-40:], item["id"], item["kind"], item["status"], item["steps"], why, item["reason"][:200]))


if __name__ == "__main__":
    main(sys.argv[1])
