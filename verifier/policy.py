"""Fit a cost-sensitive accept/reject/escalate policy on calibration tasks."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from loaders.splits import read_jsonl

from .conformal import AXES, group_of, score_of

COST_FIELDS = ("false_accept", "false_reject", "review")


def build_tasks(labels, predictions, assignment, axis):
    """Group labeled runs by task as (score, bad, force_review) tuples."""
    field, bad_when = AXES[axis]
    tasks = defaultdict(lambda: defaultdict(list))
    for record_id, prediction in predictions.items():
        if record_id not in labels:
            raise ValueError(f"Missing human labels for prediction {record_id!r}")
        if record_id not in assignment:
            raise ValueError(f"Missing split assignment for prediction {record_id!r}")
        if assignment[record_id] not in {"calibration", "id_test", "ood_test"}:
            raise ValueError(f"Invalid split for {record_id!r}: {assignment[record_id]!r}")
        label = labels[record_id][field]
        if label is None:
            continue
        if not isinstance(label, bool):
            raise ValueError(f"Invalid human label for {record_id!r} on {axis}: {label!r}")
        score = score_of(prediction, axis)
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score for {record_id!r} on {axis}")
        uncertainty = prediction.get("uncertainty") or {}
        if not isinstance(uncertainty, dict):
            raise ValueError(f"Invalid uncertainty mapping for {record_id!r}")
        force_review = uncertainty.get(axis) == "out_of_scope"
        tasks[assignment[record_id]][group_of(record_id)].append(
            (score, label == bad_when, force_review)
        )
    return {split: dict(groups) for split, groups in tasks.items()}


def action(score, thresholds, force_review=False):
    """Apply inclusive reject/accept thresholds, with out-of-scope precedence."""
    if force_review:
        return "escalate"
    low, high = thresholds["low"], thresholds["high"]
    if low is not None and high is not None and low >= high:
        raise ValueError("The reject threshold must be lower than the accept threshold")
    if low is not None and score <= low:
        return "reject"
    if high is not None and score >= high:
        return "accept"
    return "escalate"


def _validate_costs(costs):
    if set(costs) != set(COST_FIELDS):
        raise ValueError(f"Costs must define exactly: {', '.join(COST_FIELDS)}")
    for name, value in costs.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} cost must be a finite non-negative number")
    if not any(costs.values()):
        raise ValueError("At least one cost must be greater than zero")


def choose_policy(tasks, costs):
    """Minimize mean task-level calibration cost over all threshold pairs."""
    _validate_costs(costs)
    if not tasks:
        raise ValueError("No labeled tasks to calibrate on")

    task_count = len(tasks)
    bins = {}
    total_mass = 0.0
    for runs in tasks.values():
        if not runs:
            raise ValueError("Cannot calibrate on an empty task")
        weight = 1 / (task_count * len(runs))
        for score, bad, force_review in runs:
            total_mass += weight
            if force_review:
                continue
            counts = bins.setdefault(score, [0.0, 0.0, 0, 0])
            if bad:
                counts[0] += weight
                counts[2] += 1
            else:
                counts[1] += weight
                counts[3] += 1

    scores = sorted(bins)
    prefix_bad, prefix_good = [0.0], [0.0]
    prefix_count = [0]
    for score in scores:
        bad_mass, good_mass, bad_count, good_count = bins[score]
        prefix_bad.append(prefix_bad[-1] + bad_mass)
        prefix_good.append(prefix_good[-1] + good_mass)
        prefix_count.append(prefix_count[-1] + bad_count + good_count)

    total_bad = prefix_bad[-1]
    total_good = prefix_good[-1]
    total_eligible = prefix_count[-1]
    best_key = None
    best_thresholds = None
    for low_index, low in [(-1, None)] + list(enumerate(scores)):
        rejected_bad = prefix_bad[low_index + 1]
        rejected_good = prefix_good[low_index + 1]
        rejected_count = prefix_count[low_index + 1]
        for high_index, high in list(enumerate(scores)) + [(len(scores), None)]:
            if low_index >= high_index:
                continue
            accepted_bad = total_bad - prefix_bad[high_index]
            accepted_good = total_good - prefix_good[high_index]
            accepted_count = total_eligible - prefix_count[high_index]
            escalated_mass = total_mass - rejected_bad - rejected_good - accepted_bad - accepted_good
            mean_cost = (
                costs["false_accept"] * accepted_bad
                + costs["false_reject"] * rejected_good
                + costs["review"] * escalated_mass
            )
            false_accept_risk = accepted_bad
            false_reject_risk = rejected_good
            auto_count = accepted_count + rejected_count
            key = (mean_cost, auto_count, false_accept_risk, false_reject_risk)
            if best_key is None or key < best_key:
                best_key = key
                best_thresholds = {"low": low, "high": high}
    if best_thresholds is None:
        raise ValueError("No valid threshold pair")
    return best_thresholds


def evaluate(tasks, thresholds, costs):
    """Report task-weighted costs/risks and run-weighted action rates."""
    _validate_costs(costs)
    if not tasks:
        raise ValueError("No labeled tasks to evaluate")

    task_costs, false_accept_risks, false_reject_risks = [], [], []
    counts = {"accept": 0, "reject": 0, "escalate": 0}
    for runs in tasks.values():
        if not runs:
            raise ValueError("Cannot evaluate an empty task")
        run_costs, run_false_accepts, run_false_rejects = [], [], []
        for score, bad, force_review in runs:
            decision = action(score, thresholds, force_review)
            counts[decision] += 1
            run_costs.append(
                costs["false_accept"] if decision == "accept" and bad else
                costs["false_reject"] if decision == "reject" and not bad else
                costs["review"] if decision == "escalate" else 0
            )
            run_false_accepts.append(decision == "accept" and bad)
            run_false_rejects.append(decision == "reject" and not bad)
        task_costs.append(sum(run_costs) / len(runs))
        false_accept_risks.append(sum(run_false_accepts) / len(runs))
        false_reject_risks.append(sum(run_false_rejects) / len(runs))

    total_runs = sum(counts.values())
    return {
        "tasks": len(tasks),
        "runs": total_runs,
        "mean_cost": sum(task_costs) / len(task_costs),
        "false_accept_risk": sum(false_accept_risks) / len(false_accept_risks),
        "false_reject_risk": sum(false_reject_risks) / len(false_reject_risks),
        "action_counts": counts,
        "action_rates": {name: count / total_runs for name, count in counts.items()},
    }


def calibrate(labels, predictions, assignment, axis, costs):
    """Fit on calibration tasks only and evaluate unchanged thresholds elsewhere."""
    tasks_by_split = build_tasks(labels, predictions, assignment, axis)
    calibration = tasks_by_split.get("calibration", {})
    thresholds = choose_policy(calibration, costs)
    return {
        "axis": axis,
        "costs": costs,
        "thresholds": thresholds,
        "calibration": evaluate(calibration, thresholds, costs),
        "splits": {
            split: evaluate(tasks, thresholds, costs)
            for split, tasks in sorted(tasks_by_split.items())
            if split != "calibration"
        },
    }


def _parse_cost_entry(entry):
    axis, separator, raw_costs = entry.partition("=")
    if not separator or axis not in AXES:
        raise ValueError(f"Expected AXIS=FALSE_ACCEPT,FALSE_REJECT,REVIEW: {entry!r}")
    parts = raw_costs.split(",")
    if len(parts) != len(COST_FIELDS):
        raise ValueError(f"Expected three comma-separated costs: {entry!r}")
    try:
        values = [float(part) for part in parts]
    except ValueError as error:
        raise ValueError(f"Invalid numeric cost in {entry!r}") from error
    costs = dict(zip(COST_FIELDS, values))
    _validate_costs(costs)
    return axis, costs


def main():
    parser = argparse.ArgumentParser(
        description="Fit cost-aware accept/reject/escalate thresholds on calibration tasks."
    )
    parser.add_argument("--records", required=True, type=Path,
                        help="Labels JSONL from python -m baselines.labels")
    parser.add_argument("--predictions", required=True, type=Path, help="Verifier JSONL")
    parser.add_argument("--splits", required=True, type=Path, help="Split JSON from python -m loaders.splits")
    parser.add_argument(
        "--cost", action="append", required=True, metavar="AXIS=FA,FR,REVIEW",
        help="Costs per axis; repeat, e.g. outcome=10,1,0.2",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    costs_by_axis = {}
    try:
        for entry in args.cost:
            axis, costs = _parse_cost_entry(entry)
            if axis in costs_by_axis:
                parser.error(f"Cost for {axis!r} was provided more than once")
            costs_by_axis[axis] = costs
    except ValueError as error:
        parser.error(str(error))

    labels = {row["id"]: row["labels"] for row in read_jsonl(args.records)}
    predictions = {row["id"]: row for row in read_jsonl(args.predictions)}
    assignment = json.loads(args.splits.read_text(encoding="utf-8"))["assignment"]
    report = {
        "predictions": args.predictions.as_posix(),
        "verifiers": sorted({row["verifier"] for row in predictions.values()}),
        "selection": "minimum mean task-level calibration cost",
        "guarantee": "none; thresholds are cost-minimized on calibration data",
        "axes": [
            calibrate(labels, predictions, assignment, axis, costs)
            for axis, costs in costs_by_axis.items()
        ],
    }
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
