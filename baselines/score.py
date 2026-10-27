"""Score verifier predictions against AgentRewardBench human labels.

Each axis is scored as a review decision. A prediction of "unknown" means the
trajectory goes to a human, so it counts toward review_rate and never as an error.

- false_accept_rate: share of bad trajectories the verifier let through
  (on outcome this is the false-success rate; on side_effect, missed side effects).
- false_reject_rate: share of good trajectories the verifier flagged.
- accuracy: agreement with humans on the trajectories the verifier decided.

Records whose human label is unknown (null) are counted but not scored.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from loaders.splits import read_jsonl

# Prediction axis -> human label field.
AXES = {"outcome": "success", "side_effect": "side_effect", "looping": "looping"}


def is_good(axis, label):
    # Success is good; a side effect or looping is bad.
    return label if axis == "outcome" else not label


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def score_axis(axis, pairs):
    """pairs: (human label, predicted status) for one axis."""
    counts = defaultdict(int)
    for label, status in pairs:
        if label is None:
            counts["unlabeled"] += 1
            continue
        good = is_good(axis, label)
        counts["good" if good else "bad"] += 1
        if status == "unknown":
            counts["review"] += 1
        elif (status == "pass") == good:
            counts["correct"] += 1
        elif status == "pass":
            counts["false_accept"] += 1
        else:
            counts["false_reject"] += 1
    labeled = counts["good"] + counts["bad"]
    return {"labeled": labeled, "unlabeled": counts["unlabeled"],
            "good": counts["good"], "bad": counts["bad"],
            "review_rate": ratio(counts["review"], labeled),
            "accuracy": ratio(counts["correct"], labeled - counts["review"]),
            "false_accept_rate": ratio(counts["false_accept"], counts["bad"]),
            "false_reject_rate": ratio(counts["false_reject"], counts["good"])}


def score(labels, predictions):
    """labels: id -> human labels; predictions: id -> verifier result."""
    return {axis: score_axis(axis, [(labels[i][field], predictions[i][axis]) for i in predictions])
            for axis, field in AXES.items()}


def main():
    parser = argparse.ArgumentParser(description="Score verifier predictions against human labels.")
    parser.add_argument("--records", required=True, type=Path, help="Processed JSONL from python -m loaders")
    parser.add_argument("--predictions", required=True, type=Path, help="JSONL from python -m baselines")
    parser.add_argument("--splits", type=Path, help="Split JSON from python -m loaders.splits")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Keep only the labels; full records carry large step histories.
    labels = {r["id"]: r["labels"] for r in read_jsonl(args.records)}
    predictions = {}
    for result in read_jsonl(args.predictions):
        if result["id"] not in labels:
            raise ValueError(f"Prediction for unknown record: {result['id']}")
        if result["id"] in predictions:
            raise ValueError(f"Duplicate prediction: {result['id']}")
        predictions[result["id"]] = result
    if not predictions:
        raise ValueError("No predictions to score")
    verifiers = sorted({p["verifier"] for p in predictions.values()})
    report = {"verifier": verifiers[0] if len(verifiers) == 1 else verifiers,
              "predictions": len(predictions), "records": len(labels),
              "all": score(labels, predictions)}
    if args.splits:
        # Score each split separately; OOD is where calibration is expected to break.
        assignment = json.loads(args.splits.read_text(encoding="utf-8"))["assignment"]
        by_split = defaultdict(dict)
        for record_id, result in predictions.items():
            by_split[assignment[record_id]][record_id] = result
        report["splits"] = {split: score(labels, subset) for split, subset in sorted(by_split.items())}
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
