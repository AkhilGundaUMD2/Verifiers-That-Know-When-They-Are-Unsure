"""Type the uncertainty behind each undecided checklist item and axis label.

Every item or axis that the verifier cannot settle confidently gets one type:

- epistemic: reducible. Better observation of the same run would settle it:
  the deciding page was not recorded or was truncated, the model call failed,
  a state predicate could not run, or two evidence sources disagree.
- aleatoric: irreducible ambiguity in the task. The instruction or checklist
  item has more than one reasonable reading, so even full evidence would leave
  reviewers disagreeing.
- out_of_scope: the verifier cannot check it by design. No browser trajectory
  would contain the evidence (hidden server state, effects outside the
  browser, real-world facts no page shows), or the checklist has no item for
  the axis at all.

Confident results carry no type (None). When several types apply, the least
reducible wins: out_of_scope, then aleatoric, then epistemic. Re-running or
observing more cannot fix an out-of-scope or ambiguous item, so an escalation
policy should not treat it as if it could.
"""

TYPES = ("epistemic", "aleatoric", "out_of_scope")
# Most to least reducible. The highest-ranked type present wins.
RANK = {"epistemic": 0, "aleatoric": 1, "out_of_scope": 2}

# Causes the LLM scorer may give for an unknown item, mapped to types.
CAUSES = {
    "none": None,
    "missing_evidence": "epistemic",
    "ambiguous_criterion": "aleatoric",
    "unverifiable": "out_of_scope",
}

# Checklist kind that decides each axis.
AXIS_KINDS = {"outcome": "required", "side_effect": "prohibited"}


def strongest(types):
    """Least reducible type among types, or None when none is given."""
    present = [t for t in types if t is not None]
    return max(present, key=RANK.__getitem__) if present else None


def classify_item(status, llm, state):
    """Type one checklist item from its combined status and evidence.

    llm is the LLM evidence entry (or None if missing); state is the list of
    state-predicate entries for this item.
    """
    sources = ([llm] if llm else []) + list(state)
    statuses = {entry["status"] for entry in sources}
    if {"pass", "fail"} <= statuses:
        # Sources disagree. The aggregate fails conservatively, but the
        # disagreement itself is evidence that more observation is needed.
        return "epistemic"
    if status != "unknown":
        return None
    types = []
    if llm is None:
        types.append("epistemic")
    elif llm["status"] == "unknown":
        # An unknown with no stated cause is treated as reducible.
        types.append(CAUSES.get(llm.get("cause", "none")) or "epistemic")
    if any(entry["status"] == "unknown" for entry in state):
        types.append("epistemic")
    return strongest(types) or "epistemic"


def classify_axis(status, items):
    """Type one axis label from the checklist items that decide it.

    items are evidence rows with "status" and "uncertainty" keys, already
    filtered to the axis's checklist kind.
    """
    if not items:
        # No criterion covers this axis, so the verifier cannot judge it.
        return "out_of_scope" if status == "unknown" else None
    if status == "pass":
        return None
    if status == "fail":
        failing = [item for item in items if item["status"] == "fail"]
        # One clean failure decides the axis regardless of the others.
        if any(item["uncertainty"] is None for item in failing):
            return None
        return strongest(item["uncertainty"] for item in failing)
    undecided = [item for item in items if item["status"] == "unknown"]
    return strongest(item["uncertainty"] for item in undecided) or "epistemic"


def classify(axis_statuses, checklist_evidence):
    """Type every axis in axis_statuses ({"outcome": ..., "side_effect": ...})."""
    return {
        axis: classify_axis(
            status,
            [item for item in checklist_evidence if item["kind"] == AXIS_KINDS[axis]],
        )
        for axis, status in axis_statuses.items()
    }
