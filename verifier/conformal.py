"""Choose review thresholds with conformal risk control on the calibration split.

Policy for one axis: accept a trajectory when its score s >= lambda, otherwise
escalate it to a human. The loss for a trajectory is 1 when it is accepted
and humans labeled it bad on that axis, else 0. Raising lambda only removes
accepted runs, so the loss never increases with lambda.

Runs of the same task by different agents are correlated, so the
exchangeable unit is the task (group_id): a task's loss is the mean loss over
its labeled runs, which stays in [0, 1].

Two ways to pick lambda:
- crc: conformal risk control (Angelopoulos et al., 2022). The smallest lambda
  with n/(n+1) * Rhat(lambda) + 1/(n+1) <= alpha. Guarantees expected test
  risk <= alpha over draws of the calibration set.
- ltt: Learn then Test (Angelopoulos et al., 2021). Fixed-sequence testing of
  H0: risk(lambda) > alpha from strict to loose lambda, with Hoeffding-Bentkus
  p-values. Guarantees risk <= alpha with probability >= 1 - delta.

Escalating everything (lambda = inf) has zero loss and is always allowed.
Verifiers without numeric scores fall back to pass = 1, unknown = 0.5,
fail = 0.
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from loaders.benchmarks import normalize_task_id
from loaders.splits import read_jsonl

INF = math.inf
STATUS_SCORE = {"pass": 1.0, "unknown": 0.5, "fail": 0.0}
# Prediction axis -> human label field, and whether a True label is bad.
AXES = {"outcome": ("success", False), "side_effect": ("side_effect", True)}


def group_of(record_id):
    """Task group for an AgentRewardBench ID benchmark/agent/task_id."""
    return "agentrewardbench/" + normalize_task_id(record_id.rsplit("/", 1)[-1])


def score_of(prediction, axis):
    """A prediction's score on an axis: its numeric score if any, else its status."""
    scores = prediction.get("scores") or {}
    if axis in scores and scores[axis] is not None:
        return float(scores[axis])
    return STATUS_SCORE[prediction[axis]]


def build_tasks(labels, predictions, axis):
    """Group labeled runs by task: group_id -> [(score, bad)].

    Runs whose human label is unknown are dropped; a task with no labeled run
    is dropped with them.
    """
    field, bad_when = AXES[axis]
    tasks = defaultdict(list)
    for record_id, prediction in predictions.items():
        label = labels[record_id][field]
        if label is None:
            continue
        tasks[group_of(record_id)].append((score_of(prediction, axis), label == bad_when))
    return dict(tasks)


def task_loss(runs, lam):
    """Mean of 1{accepted and bad} over one task's runs."""
    return sum(bad and score >= lam for score, bad in runs) / len(runs)


def empirical_risk(tasks, lam):
    return sum(task_loss(runs, lam) for runs in tasks.values()) / len(tasks)


def candidates(tasks):
    """Every distinct score, strictest first, after inf (escalate everything)."""
    scores = {score for runs in tasks.values() for score, _ in runs}
    return [INF] + sorted(scores, reverse=True)


def crc_threshold(tasks, alpha):
    """Smallest lambda whose CRC-adjusted risk is within alpha (inf if none)."""
    n = len(tasks)
    if n == 0:
        raise ValueError("No labeled tasks to calibrate on")
    chosen = INF
    # Risk only grows as lambda falls, so stop at the first violation.
    for lam in candidates(tasks):
        if n / (n + 1) * empirical_risk(tasks, lam) + 1 / (n + 1) > alpha:
            break
        chosen = lam
    return chosen


def _kl(a, b):
    # Bernoulli KL divergence h(a, b), with 0 log 0 = 0.
    def term(x, y):
        return 0.0 if x == 0 else x * math.log(x / y)
    return term(a, b) + term(1 - a, 1 - b)


def _binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def hb_pvalue(rhat, n, alpha):
    """Hoeffding-Bentkus p-value for H0: risk > alpha (Bates et al., 2021)."""
    if rhat >= alpha:
        return 1.0
    hoeffding = math.exp(-n * _kl(rhat, alpha))
    bentkus = math.e * _binom_cdf(math.ceil(n * rhat), n, alpha)
    return min(hoeffding, bentkus, 1.0)


def ltt_threshold(tasks, alpha, delta):
    """Smallest lambda certified by fixed-sequence testing at level delta."""
    n = len(tasks)
    if n == 0:
        raise ValueError("No labeled tasks to calibrate on")
    chosen = INF
    for lam in candidates(tasks)[1:]:
        if hb_pvalue(empirical_risk(tasks, lam), n, alpha) > delta:
            break
        chosen = lam
    return chosen


def evaluate(tasks, lam):
    """Task-level risk and run-level review rate of a threshold on held-out tasks."""
    runs = [run for task in tasks.values() for run in task]
    return {"tasks": len(tasks), "runs": len(runs),
            "risk": empirical_risk(tasks, lam),
            "review_rate": sum(score < lam for score, _ in runs) / len(runs)}


def bootstrap(tasks, lam, reps=1000, seed=0):
    """95% percentile interval for risk and review rate, resampling tasks."""
    rng = random.Random(seed)
    keys = list(tasks)
    risks, reviews = [], []
    for _ in range(reps):
        sample = [tasks[rng.choice(keys)] for _ in keys]
        resampled = {index: runs for index, runs in enumerate(sample)}
        result = evaluate(resampled, lam)
        risks.append(result["risk"])
        reviews.append(result["review_rate"])

    def interval(values):
        values.sort()
        return [values[int(0.025 * reps)], values[int(0.975 * reps) - 1]]
    return {"risk_ci95": interval(risks), "review_rate_ci95": interval(reviews)}


def resplit_check(tasks, alpha, reps=1000, seed=0):
    """Re-split pooled tasks in half many times; fit CRC on one half, test on the other."""
    rng = random.Random(seed)
    keys = sorted(tasks)
    risks, reviews = [], []
    for _ in range(reps):
        rng.shuffle(keys)
        half = len(keys) // 2
        calibration = {k: tasks[k] for k in keys[:half]}
        test = {k: tasks[k] for k in keys[half:]}
        result = evaluate(test, crc_threshold(calibration, alpha))
        risks.append(result["risk"])
        reviews.append(result["review_rate"])
    risks.sort()
    return {"reps": reps, "mean_risk": sum(risks) / reps,
            "risk_p95": risks[int(0.95 * reps) - 1],
            "share_above_alpha": sum(r > alpha for r in risks) / reps,
            "mean_review_rate": sum(reviews) / reps}


def calibrate(labels, predictions, assignment, axis, alpha, delta, reps=1000, seed=0):
    """Fit both thresholds on calibration and report them on every split."""
    by_split = defaultdict(dict)
    for record_id, prediction in predictions.items():
        by_split[assignment[record_id]][record_id] = prediction
    tasks = {split: build_tasks(labels, subset, axis) for split, subset in by_split.items()}
    cal = tasks["calibration"]
    report = {"axis": axis, "alpha": alpha, "delta": delta,
              "calibration_tasks": len(cal),
              "calibration_bad_runs": sum(bad for runs in cal.values() for _, bad in runs),
              "risk_curve": [{"lambda": lam, "risk": empirical_risk(cal, lam)}
                             for lam in candidates(cal)]}
    for method, lam in (("crc", crc_threshold(cal, alpha)),
                        ("ltt", ltt_threshold(cal, alpha, delta))):
        splits = {}
        for split, split_tasks in sorted(tasks.items()):
            result = evaluate(split_tasks, lam)
            if split != "calibration":
                result.update(bootstrap(split_tasks, lam, reps, seed))
            splits[split] = result
        report[method] = {"lambda": lam, "splits": splits}
    pooled = {**cal, **tasks.get("id_test", {})}
    report["resplit_check"] = resplit_check(pooled, alpha, reps, seed)
    return report


def _json_safe(value):
    # JSON has no infinity; write inf as the string "inf".
    if isinstance(value, float) and math.isinf(value):
        return "inf"
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description="Fit review thresholds on the calibration split.")
    parser.add_argument("--records", required=True, type=Path,
                        help="Labels JSONL from python -m baselines.labels (or processed JSONL)")
    parser.add_argument("--predictions", required=True, type=Path, help="Verifier JSONL")
    parser.add_argument("--splits", required=True, type=Path, help="Split JSON from python -m loaders.splits")
    parser.add_argument("--alpha", action="append", required=True, metavar="AXIS=ALPHA",
                        help="Target risk per axis, e.g. outcome=0.05; repeat per axis")
    parser.add_argument("--delta", type=float, default=0.1, help="LTT failure probability")
    parser.add_argument("--reps", type=int, default=1000, help="Bootstrap and re-split repetitions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    alphas = {}
    for entry in args.alpha:
        axis, _, value = entry.partition("=")
        if axis not in AXES:
            parser.error(f"Unknown axis {axis!r}; choose from {sorted(AXES)}")
        alphas[axis] = float(value)
    labels = {r["id"]: r["labels"] for r in read_jsonl(args.records)}
    predictions = {p["id"]: p for p in read_jsonl(args.predictions)}
    assignment = json.loads(args.splits.read_text(encoding="utf-8"))["assignment"]
    verifiers = sorted({p["verifier"] for p in predictions.values()})
    report = {"predictions": args.predictions.as_posix(), "verifier": verifiers,
              "axes": [calibrate(labels, predictions, assignment, axis, alpha,
                                 args.delta, args.reps, args.seed)
                       for axis, alpha in alphas.items()]}
    text = json.dumps(_json_safe(report), indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
