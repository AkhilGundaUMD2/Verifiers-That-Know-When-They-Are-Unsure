# Verifiers That Know When They Are Unsure

ML capstone project investigating two-axis verification of computer
agents: task success and harmful side effects.

The verifier will use checklist-based checks, trajectory evidence,
uncertainty labels, and cost-aware calibration to accept, reject,
or escalate a result for human review.

## D1 — Minimum (Setup)

- [x] Create public shared repository and add all teammates
- [x] Set up GitHub Actions for tests and reproducibility checks
- [ ] Complete literature matrix
- [x] Build AgentRewardBench and OSWorld data loaders
- [x] Document calibration, in-distribution test, and OOD test splits

## D2 — Baseline

- [x] Build rule-based browser verifier
- [x] Build single-LLM judge (GPT-6 Luna)
- [x] Score verifiers against human labels
- [ ] Add unit tests for the baselines
- [ ] Run baselines on the full AgentRewardBench data

## D3 — Core Study

- [ ] Generate task checklists (required items, prohibited side effects, incidental changes)
- [ ] Score checklist items with an LLM
- [ ] Aggregate evidence into outcome and side-effect labels
- [ ] Build hybrid state-plus-LLM verifier

## D4 — Research Extension

- [ ] Classify uncertainty as epistemic, aleatoric, or out-of-scope
- [ ] Choose review thresholds with conformal risk control on the calibration split
- [ ] Add cost-aware accept / reject / escalate policy

## D5 — Full Result

- [ ] Compare against single-judge and rule-based baselines
- [ ] Run ablations on uncertainty typing and cost-aware thresholds
- [ ] Map where guarantees break on the OOD split
- [ ] Build adversarial test cases
- [ ] Write experiment report
