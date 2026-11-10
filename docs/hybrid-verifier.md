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

## Run

```bash
python -m verifier --input data/processed/agentrewardbench.jsonl --output data/checklists/agentrewardbench.jsonl
python -m baselines hybrid --input data/processed/agentrewardbench.jsonl --checklists data/checklists/agentrewardbench.jsonl --state-evidence data/state-evidence/agentrewardbench.jsonl --output results/hybrid.jsonl
```
