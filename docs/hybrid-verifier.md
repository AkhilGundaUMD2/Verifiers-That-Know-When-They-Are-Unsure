# Hybrid checklist verifier

The hybrid verifier scores a task checklist with an LLM, combines those scores
with explicit state-predicate evidence, and reports task outcome and side-effect
labels separately. It also reports the existing trajectory looping check.

## Evidence format

Generate one checklist per task using `python -m verifier`. Each checklist item
contains:

- `id`: stable checklist item ID
- `kind`: `required`, `prohibited`, or `incidental`
- `description`: the condition to assess

Prepare a state-evidence JSONL row per trajectory. Each row has an `id`
matching the trajectory ID and an `items` list containing `item_id`, `status`
(`pass`, `fail`, or `unknown`), and an optional `reason`. The state status
describes whether that checklist condition is satisfied. For example, a
prohibited side effect is `pass` when a state predicate confirms it was
avoided. Use the predicates in `baselines.state_checker` to produce evidence;
the hybrid CLI does not infer environment state from a trajectory.

Include a row with `"items": []` when no state predicate applies. Item IDs must
match the task checklist. Missing or unknown evidence remains unknown.
Conflicting pass/fail evidence is treated conservatively as fail. Any failed
required item fails the outcome axis; any failed prohibited item fails the
side-effect axis. Incidental items are included in evidence but do not change
either label.

## Uncertainty types

`verifier/uncertainty.py` gives every checklist item and every axis label
(`outcome`, `side_effect`) an uncertainty type, or `null` when the result is
confident. The hybrid output carries the axis types in `uncertainty` and the
item types in each `checklist_evidence` row.

| Type | Meaning | Signals |
|---|---|---|
| `epistemic` | Reducible: better observation of the same run would settle it | LLM cause `missing_evidence`; LLM refusal or truncation; a state predicate returned `unknown`; LLM and state evidence disagree |
| `aleatoric` | The task itself is ambiguous; reviewers with full evidence could disagree | LLM cause `ambiguous_criterion` |
| `out_of_scope` | No browser trajectory could settle it | LLM cause `unverifiable`; the checklist has no item for the axis |

The LLM scorer returns a `cause` for every item (`none` when it passes or
fails). An unknown item with no cause defaults to `epistemic`. When several
types apply, the least reducible wins: `out_of_scope`, then `aleatoric`, then
`epistemic`.

An axis that passes is confident. An axis that fails is confident if at least
one failing item is confident; if every failing item rests on conflicting
evidence, the axis is typed `epistemic`. An unknown axis takes the strongest
type among its unknown items.

The scorer reports `review_by_type` for each typed axis: how many reviewed
(unknown) trajectories fall under each type, and how many of them humans
labeled bad.

One way to check the aleatoric signal: 106 AgentRewardBench trajectories have
two annotators, who disagree on success for 13 of them. Those 13 should be
typed `aleatoric` more often than the rest.

## Run

```bash
python -m verifier --input data/processed/agentrewardbench.jsonl --output data/checklists/agentrewardbench.jsonl
python -m baselines hybrid --input data/processed/agentrewardbench.jsonl --checklists data/checklists/agentrewardbench.jsonl --state-evidence data/state-evidence/agentrewardbench.jsonl --output results/hybrid.jsonl
```
