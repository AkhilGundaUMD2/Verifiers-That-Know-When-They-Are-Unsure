# Review thresholds with conformal risk control

This is D4 item 2: choosing, on the calibration split, when the verifier may
accept a trajectory on its own and when it must escalate to a human. The
choice comes with a statistical guarantee on how often a bad trajectory gets
through.

Code: `verifier/conformal.py`. Tests: `tests/test_conformal.py`. Result
for the rules baseline: `results/thresholds_rules.json`.

> **Status.** The procedure is implemented, tested, and has been run on the
> only scored verifier output in the repo, the rule-based baseline. The rules
> baseline gives three-valued verdicts, not graded scores, so its thresholds
> mostly show how the method behaves on a weak score. The hybrid verifier
> needs an end-to-end run (checklists, state evidence, LLM scoring) before
> its thresholds can be fitted. The command is the same; see
> [Rerunning on another verifier](#rerunning-on-another-verifier).

## 1. Problem setup

For each axis (`outcome`, `side_effect`) separately:

| Piece | Choice |
|---|---|
| Score s(x) | Higher means more likely good. Uses a numeric `scores[axis]` on the prediction if present, otherwise maps the verdict: pass = 1, unknown = 0.5, fail = 0 |
| Policy | Accept if s(x) ≥ λ, otherwise escalate to a human. λ = ∞ escalates everything |
| Loss | L = 1 if the trajectory is **accepted and humans labeled it bad** on that axis, else 0. Bad means `success = false` for outcome and `side_effect = true` for side effect |
| Risk | Expected loss: the share of all trajectories where a bad run got through |
| Target | Risk ≤ α, with α chosen per axis before looking at test data |

**Why this loss.** The loss never increases as λ rises, because raising λ only
removes accepted runs. That monotonicity is the one condition conformal risk
control needs. The obvious alternative, "share of bad runs accepted," divides
by the number of bad runs, which is random and tiny for side effects (28 bad
runs in calibration), so the plain bound does not apply to it.

**One-sided policy.** This rule either accepts or escalates. A verdict of
`fail` falls below every λ > 0, so it is escalated, not auto-rejected. Adding a
reject band (accept / reject / escalate) brings in a second risk and belongs
to the cost-aware policy, D4 item 3 (see [Extensions](#extensions)).

## 2. Sampling unit: the task, not the run

The guarantee assumes calibration and test samples are exchangeable. Runs of
the same task by different agents are correlated: same instruction, same site,
often the same failure. The splits already keep every task in one split, so
the exchangeable unit is the **task** (`group_id`).

- A task's loss is the mean loss over its labeled runs, which stays in [0, 1].
- n is the number of tasks: **116** in calibration, 117 in `id_test`, 118 in
  `ood_test`.
- `group_id` is rebuilt from the record ID with the loader's own
  `normalize_task_id`. This reproduces the split file's task counts exactly.

Treating the 419 calibration runs as independent would shrink the 1/(n+1)
term by about 3.6× and overstate how strong the guarantee is.

**Unknown human labels** are dropped: 7 for success and 3 for side effect in
calibration. A task with no labeled run on an axis drops out of that axis.
Counting unknowns as bad would be the conservative alternative; at these
counts it barely moves the risk.

## 3. Empirical risk curve

For every candidate λ (each distinct score, plus ∞):

    R̂(λ) = (1/n) Σ_tasks  mean over the task's runs of 1{s ≥ λ and bad}

The calibration curves for the rules baseline:

| λ | Accepts | Outcome R̂ | Side-effect R̂ |
|---|---|---|---|
| ∞ | nothing | 0 | 0 |
| 1.0 | `pass` | (no `pass` verdicts) | 0.0532 |
| 0.5 | `pass` + `unknown` | 0.2507 | 0.0560 |
| 0.0 | everything | 0.6573 | 0.0747 |

## 4. Choosing λ

Two estimators are computed. They answer different questions.

### Conformal risk control (CRC): expected-risk guarantee

Pick the most permissive λ such that

    n/(n+1) · R̂(λ) + B/(n+1) ≤ α,   B = 1 (maximum loss)

Then **E[risk on a new task] ≤ α**, averaged over draws of the calibration set
(Angelopoulos et al., 2022, *Conformal Risk Control*). With n = 116 the
finite-sample penalty is 1/117 ≈ 0.0085, so for α = 0.05 the calibration risk
must be at most about 0.042.

Because R̂ only grows as λ falls, the code scans candidates from strict to
loose and stops at the first one that breaks the bound.

### Learn then Test (LTT): high-probability guarantee

CRC says nothing about how often one particular fitted λ ends up above α. If
the claim must hold for the single threshold you deploy, use LTT
(Angelopoulos et al., 2021, *Learn then Test*):

- For each λ, test H₀: risk(λ) > α with a Hoeffding–Bentkus p-value
  (Bates et al., 2021):
  p = min( exp(−n·h(R̂, α)),  e·P[Binomial(n, α) ≤ ⌈nR̂⌉] ),
  where h is the Bernoulli KL divergence.
- Test λ from strict to loose (fixed-sequence testing, which needs no
  multiplicity correction) and stop at the first p > δ.
- Then **P(risk(λ̂) ≤ α) ≥ 1 − δ**. The run here uses δ = 0.1.

LTT is never looser than CRC, and on 116 tasks it is much stricter.

## 5. Validation

On each held-out split, the code reports:

- **Realized risk and review rate** at the fitted λ. Risk is per task; review
  rate is the share of runs escalated.
- **Bootstrap 95% intervals** from resampling **tasks**, 1,000 reps, seed 0.
- **Re-split check.** Calibration and `id_test` tasks are pooled (233 tasks),
  split in half at random 1,000 times, CRC is fitted on one half and tested on
  the other. The mean realized risk should be ≤ α. This is the empirical check
  of the guarantee.
- **OOD.** The same λ is applied to `ood_test` (workarena), where
  exchangeability does not hold. The excess over α is the D5 measure of where
  guarantees break.

## 6. Results: rules baseline

Fitted thresholds, `results/thresholds_rules.json`:

| Axis | α | CRC λ | LTT λ (δ = 0.1) | id_test risk [95% CI] | id_test review rate | OOD risk [95% CI] |
|---|---|---|---|---|---|---|
| outcome | 0.05 | ∞ | ∞ | 0 | 100% | 0 |
| side_effect | 0.10 | 0.0 | ∞ | 0.083 [0.053, 0.119] | 0% | 0.047 [0.021, 0.076] |

The side-effect row is CRC's λ; LTT escalates everything.

Sweep over α (CRC λ / LTT λ, and the re-split check for CRC):

| α | Outcome λ | Side-effect λ | Side-effect re-split: mean risk, share of reps with risk > α, mean review rate |
|---|---|---|---|
| 0.01 | ∞ / ∞ | ∞ / ∞ | 0, 0%, 100% |
| 0.02 | ∞ / ∞ | ∞ / ∞ | 0, 0%, 100% |
| 0.05 | ∞ / ∞ | ∞ / ∞ | 0.009, **13.2%**, 88% |
| 0.10 | ∞ / ∞ | 0.0 / ∞ | 0.077, 2.9%, 1% |
| 0.20 | ∞ / ∞ | 0.0 / 0.0 | 0.079, 0%, 0% |

To reproduce a row:

```bash
python -m verifier.conformal --records data/processed/agentrewardbench.labels.jsonl \
  --predictions results/rules_agentrewardbench.jsonl \
  --splits data/splits/agentrewardbench.json \
  --alpha outcome=0.05 --alpha side_effect=0.05
```

### What the numbers say

1. **The rules baseline can never be trusted to accept on outcome.** It never
   outputs `pass` for outcome (calibration verdicts: 231 `fail`, 188
   `unknown`), so the most permissive useful choice is accepting `unknown`.
   That lets bad runs through on 25% of tasks. Every α up to 0.20 therefore
   escalates everything. This is the correct answer for a verifier that
   cannot confirm success, not a bug.
2. **Its side-effect score barely separates good from bad.** Accepting only
   `pass` gives risk 0.053, against 0.075 for accepting everything. The
   baseline's 66% missed-side-effect rate shows up here as a score with almost
   no discriminative power.
3. **When α is above the base rate, the threshold is trivial.** Side effects
   occur on about 7.5% of tasks, so at α = 0.10 CRC accepts everything,
   including `fail` verdicts. That satisfies the guarantee (id_test risk 0.083
   is within α) but means the verifier is not being used. **α should be set
   below the base rate of bad runs**, or the policy is meaningless.
4. **CRC controls risk on average, not in every draw.** At α = 0.05 the
   re-split mean risk is 0.009, well within α, but 13% of individual splits
   come out above α. That is exactly the difference between CRC and LTT. If
   the report claims that "the deployed threshold keeps risk under 5%," it
   needs LTT, which escalates everything here.
5. **The finite-sample penalty is large at this size.** 1/(n+1) ≈ 0.0085
   uses up most of α = 0.01 before any risk is counted. For α = 0.01 to be
   reachable, the score must accept a large share of tasks with almost no bad
   ones. The side-effect axis will need a very sharp score from the hybrid
   verifier.
6. **OOD.** On workarena the side-effect risk at λ = 0 is lower (0.047) than
   in distribution, because workarena has fewer side effects. A guarantee can
   look fine OOD for the wrong reason. D5 should compare base rates, not only
   risks.

## Rerunning on another verifier

Any verifier JSONL with `id`, `verifier`, `outcome`, and `side_effect` works.
Run the same command with `--predictions` pointing at the new file. For a
graded score, add a `scores` object to each prediction, for example
`{"outcome": 0.83, "side_effect": 0.97}`. It overrides the verdict mapping,
and every distinct score becomes a candidate threshold.

The hybrid verifier writes verdicts and uncertainty types but no numeric
score yet. Without one it falls back to the three-level mapping above, which
leaves CRC only three thresholds to choose from. Candidate scores, from
cheapest:

1. **Fraction of required items passing**, with unknown items weighted by
   their uncertainty type.
2. **A confidence per item from the LLM**, added to the scoring schema.
3. **Agreement across k LLM samples.** The most principled option, at k times
   the cost; a good D5 ablation.

**Out-of-scope trajectories** (D4 item 1) should be escalated before
calibrating, with their share reported separately. Otherwise they dilute the
calibration set with cases the policy never accepts.

## Extensions

- **Accept / reject / escalate.** Two thresholds control two risks (false
  accept, false reject). Use LTT over a grid of (λ_lo, λ_hi) with a correction
  for testing many hypotheses, or control the false-accept risk with CRC and
  choose λ_lo by cost. This is D4 item 3.
- **Joint guarantee across axes.** Calibrating each axis at α controls each
  one separately. To guarantee that neither axis lets a bad run through, split
  α between the axes (Bonferroni) or define a combined loss.

## References

- Angelopoulos, Bates, Fisch, Lei, Schuster. *Conformal Risk Control.* 2022.
  arXiv:2208.02814.
- Angelopoulos, Bates, Candès, Jordan, Lei. *Learn then Test: Calibrating
  Predictive Algorithms to Achieve Risk Control.* 2021. arXiv:2110.01052.
- Bates, Angelopoulos, Lei, Malik, Jordan. *Distribution-Free,
  Risk-Controlling Prediction Sets.* 2021. arXiv:2101.02703.
