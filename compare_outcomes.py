import argparse
import json
from collections import defaultdict
from pathlib import Path

from baselines.score import score
from loaders.splits import read_jsonl


AXES = ("outcome", "side_effect", "looping")


def load_labels(path, limit=None):
    labels = {}
    for index, row in enumerate(read_jsonl(path)):
        if limit and index >= limit:
            break
        if "labels" not in row:
            raise ValueError(f"Record has no human labels: {row.get('id')}")
        labels[row["id"]] = row["labels"]
    return labels


def load_predictions(path):
    predictions = {}
    verifiers = set()
    for row in read_jsonl(path):
        record_id = row["id"]
        if record_id in predictions:
            raise ValueError(f"Duplicate prediction: {record_id}")
        missing = [axis for axis in AXES if axis not in row]
        if missing:
            raise ValueError(f"Prediction {record_id} is missing axes: {', '.join(missing)}")
        for axis in AXES:
            if row[axis] not in {"pass", "fail", "unknown"}:
                raise ValueError(f"Invalid {axis} status for {record_id}: {row[axis]!r}")
        verifiers.add(row.get("verifier", Path(path).stem))
        predictions[record_id] = row
    if not predictions:
        raise ValueError(f"No predictions in {path}")
    if len(verifiers) != 1:
        raise ValueError(f"Prediction file mixes verifiers: {sorted(verifiers)}")
    return predictions


def split_predictions(predictions, assignment):
    by_split = defaultdict(dict)
    for record_id, prediction in predictions.items():
        if record_id not in assignment:
            raise ValueError(f"Record is absent from split assignment: {record_id}")
        by_split[assignment[record_id]][record_id] = prediction
    return dict(sorted(by_split.items()))


def format_rate(value):
    return "-" if value is None else f"{value:.3f}"


def print_report(name, report):
    print(f"\n{name}")
    print("axis          labeled  review  accuracy  false accept  false reject")
    for axis in AXES:
        result = report[axis]
        print(
            f"{axis:<13} {result['labeled']:>7} "
            f"{format_rate(result['review_rate']):>7} "
            f"{format_rate(result['accuracy']):>9} "
            f"{format_rate(result['false_accept_rate']):>13} "
            f"{format_rate(result['false_reject_rate']):>13}"
        )


def print_changes(before_name, before, after_name, after, records):
    print(f"\nchanged: {before_name} -> {after_name}")
    changed = []
    for record_id in sorted(set(before) & set(after)):
        old = before[record_id]["outcome"]
        new = after[record_id]["outcome"]
        if old != new:
            changed.append((record_id, old, new))
    if not changed:
        print("  none")
        return
    for record_id, old, new in changed:
        record = records.get(record_id, {})
        labels = record.get("labels", {})
        human = labels.get("success")
        group_id = record.get("group_id", record_id)
        print(f"  {record_id} ({group_id.rsplit('/', 1)[-1]}) human={human}: {old} -> {new}")


def compare(records_path, prediction_paths, splits_path=None, output_path=None, limit=None):
    labels = load_labels(records_path, limit=limit)
    all_splits = None
    if splits_path:
        all_splits = json.loads(Path(splits_path).read_text(encoding="utf-8"))["assignment"]

    reports = {}
    predictions_by_name = {}
    records = {}
    for index, row in enumerate(read_jsonl(records_path)):
        if limit and index >= limit:
            break
        records[row["id"]] = row

    for path in prediction_paths:
        predictions = load_predictions(path)
        predictions = {record_id: prediction for record_id, prediction in predictions.items()
                       if record_id in labels}
        unknown_predictions = sorted(set(predictions) - set(labels))
        if unknown_predictions:
            raise ValueError(
                f"Prediction for unknown record in {path}: {unknown_predictions[0]}"
            )
        name = next(iter({p.get("verifier", Path(path).stem) for p in predictions.values()}))
        predictions_by_name[name] = predictions
        reports[name] = {"all": score(labels, predictions)}
        if all_splits:
            reports[name]["splits"] = {}
            for split, subset in split_predictions(predictions, all_splits).items():
                reports[name]["splits"][split] = score(labels, subset)

    for name, report in reports.items():
        print_report(name, report["all"])
        for split, split_report in report.get("splits", {}).items():
            print_report(f"{name} [{split}]", split_report)

    names = list(reports)
    for before_name, after_name in zip(names, names[1:]):
        print_changes(
            before_name,
            predictions_by_name[before_name],
            after_name,
            predictions_by_name[after_name],
            records,
        )

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(
            {"predictions": len(prediction_paths), "comparisons": reports},
            indent=2,
            sort_keys=True,
        )
        output.write_text(text + "\n", encoding="utf-8")
        print(f"\nWrote JSON report: {output}")
    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Compare single-judge, rule-based, and hybrid verifier predictions."
    )
    parser.add_argument(
        "prediction_paths",
        nargs="*",
        help="Prediction JSONL files (for example rules and llm or hybrid results)",
    )
    parser.add_argument(
        "--records",
        default=Path("data/processed/agentrewardbench.jsonl"),
        type=Path,
        help="Processed JSONL or labels JSONL",
    )
    parser.add_argument("--splits", type=Path, help="Split assignment JSON")
    parser.add_argument("--output", type=Path, help="Write the comparison as JSON")
    parser.add_argument(
        "--limit",
        type=int,
        default=47,
        help="Compare only the first N records (use 0 for all records)",
    )
    args = parser.parse_args()
    compare(
        args.records,
        args.prediction_paths,
        args.splits,
        args.output,
        limit=args.limit or None,
    )


if __name__ == "__main__":
    main()
