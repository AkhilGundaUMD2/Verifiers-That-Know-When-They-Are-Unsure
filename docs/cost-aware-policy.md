# Cost-aware accept / reject / escalate policy

The policy uses the **selective classification / learning-to-defer** setup:
the verifier accepts a good result, rejects a bad result, or sends an uncertain
case for human review. The relative costs are explicit inputs rather than
implicit assumptions.

Code: `verifier/policy.py`. Tests: `tests/test_policy.py`.

## Decision rule

For each axis (`outcome` and `side_effect`) independently, a higher score means
more likely good. Scores use `scores[axis]` when available, otherwise the
existing fallback `pass = 1`, `unknown = 0.5`, `fail = 0`.

Given a lower threshold `low` and an upper threshold `high`:

| Condition | Action |
|---|---|
| Score `<= low` | Reject |
| Score `>= high` | Accept |
| Otherwise | Escalate |

A missing `low` or `high` disables that automatic decision. A prediction
typed `out_of_scope` on the axis always escalates, regardless of score.

The policy minimizes mean **task-level** calibration cost. Each labeled run
has cost:

- false accept (accepting a human-labeled bad result): `false_accept`
- false reject (rejecting a human-labeled good result): `false_reject`
- escalation to a human: `review`
- correct automatic decisions: zero

Runs from the same task are averaged within the task, so tasks—not correlated
runs—have equal weight. Unlabeled examples are excluded. The thresholds are
selected by an exhaustive search over score cut points on the calibration
split; those same thresholds are then evaluated, unchanged, on ID and OOD
splits.

## Running it

First create the lightweight labels file if it does not exist:

```bash
python -m baselines.labels --input data/processed/agentrewardbench.jsonl \
  --output data/processed/agentrewardbench.labels.jsonl
```

Then fit and evaluate the policy. Costs are relative units and must be chosen
for the application; the values below are illustrative, not recommendations:

```bash
python -m verifier.policy \
  --records data/processed/agentrewardbench.labels.jsonl \
  --predictions results/rules_agentrewardbench.jsonl \
  --splits data/splits/agentrewardbench.json \
  --cost outcome=10,1,0.2 \
  --cost side_effect=10,1,0.2 \
  --output results/policy_rules.json
```

Each `--cost` value is `AXIS=FALSE_ACCEPT,FALSE_REJECT,REVIEW`. Repeat it for
each axis to be evaluated. Select costs before examining held-out results; if
the application has no agreed cost ratios, report sensitivity across several
predeclared ratios instead of treating one as definitive.

## Interpretation and limitations

The output reports calibration cost and the held-out mean task-level cost,
false-accept risk, false-reject risk, and run-level action counts/rates. These
are empirical estimates, **not statistical risk guarantees**. In particular,
the existing `verifier/conformal.py` guarantee only applies to its one-sided
accept/escalate loss. It does not certify the reject region or this joint
threshold search. The OOD split also violates the exchangeability assumption
needed for conformal guarantees.

If guarantees for both automatic decisions are required, extend this
experiment with risk constraints for both false accepts and false rejects,
including correction for the threshold-pair search. Do not select costs or
thresholds using the test splits.

## Research basis

- Madras et al., [*Predict Responsibly: Improving Fairness and Accuracy by
  Learning to Defer*](https://arxiv.org/abs/1711.06664), NeurIPS 2018:
  models the decision to defer in a human-plus-model system.
- Geifman and El-Yaniv, [*Selective Classification for Deep Neural
  Networks*](https://arxiv.org/abs/1705.08500), 2017: studies the
  risk/coverage trade-off under a reject option.
- Angelopoulos et al., [*Conformal Risk Control*](https://arxiv.org/abs/2208.02814),
  2022: provides the one-sided expected-risk calibration framework already
  implemented in this project; extending it to paired errors requires
  additional risk-control work.
