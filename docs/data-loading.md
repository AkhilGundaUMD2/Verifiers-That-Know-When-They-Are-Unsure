# Data loading

Run commands from the project root with Python 3.11+. Loading local files uses only the Python standard library. Downloading AgentRewardBench additionally requires huggingface_hub.

## 1. AgentRewardBench

Install the download dependency:

```bash
python -m pip install huggingface_hub
```

Download the pinned official snapshot. Only cleaned JSON files are downloaded; screenshots are optional and are not included here.

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="McGill-NLP/agent-reward-bench",
    repo_type="dataset",
    revision="b6d17e646009d6cb63d5dd7be78807b680693f61",
    local_dir="data/raw/agentrewardbench",
    allow_patterns=["cleaned/**/*.json"],
)
PY
```

Get human annotations from the matching benchmark code repository:

```bash
mkdir -p data/raw/agentrewardbench
curl --fail --location https://raw.githubusercontent.com/McGill-NLP/agent-reward-bench/05899fcfe52c925978944a23920373b7a9c63740/agent_reward_bench/data/annotations.csv -o data/raw/agentrewardbench/annotations.csv
```

Load trajectories and join human annotations:

```bash
python -m loaders agentrewardbench --root data/raw/agentrewardbench --annotations data/raw/agentrewardbench/annotations.csv --output data/processed/agentrewardbench.jsonl --expected-count 1302
```

Annotations join on benchmark, agent name (CSV model_name), and normalized task ID. Normalization removes .improved and .resized, following upstream code. All annotators are retained. Consensus labels are true/false only when all available annotators agree; Unsure, absence, or disagreement produces null. Unknown label strings cause an error. Cumulative environment rewards remain metadata, not human labels. Multiple JSON trajectories that collide on an annotation join key fail rather than silently reuse labels.

The CSV has 1,408 rows representing 1,302 trajectories. The loader yields one record per trajectory, not per annotator. No API key, LLM, or running browser environment is required.

## 2. OSWorld (original 369-task benchmark)

```bash
git clone https://github.com/xlang-ai/OSWorld.git data/raw/osworld
git -C data/raw/osworld checkout b138d348256078fa634fc3b73567a7337c793e6b
python -m loaders osworld --root data/raw/osworld --output data/processed/osworld.jsonl --expected-count 369
```

The loader follows evaluation_examples/test_all.json and reads only referenced examples/<domain>/<id>.json files. Missing, duplicate, or mismatched task IDs fail. Task setup and evaluator specifications are preserved as data, never executed.

OSWorld records are TASK DEFINITIONS: steps and success/side-effect labels are null. The trajectory field in upstream JSON is a directory reference, not an executed run. Actual trajectories, state snapshots, and side-effect annotations must be collected separately for verifier evaluation. This loader targets the original JSON benchmark, not OSWorld-V2's task implementation classes.

## Output and reproducibility

Each command produces JSONL (one record per line) and a .manifest.json containing counts by domain, SHA-256 hashes of input files and output, and annotation/index hashes. Files are sorted deterministically. If ingestion fails, an existing output is preserved. Raw and processed datasets are gitignored; commit loaders, tests, documentation, and source revision IDs instead.

Shared fields: dataset, record_type, id, task_id, group_id, domain, agent, instruction, steps, labels, source_path, metadata. AgentRewardBench also includes raw annotations and validity metadata; OSWorld includes evaluator and setup config.

Keep the same group_id together when creating splits. This groups different agents' trajectories for the same task. Split generation is a separate next step; these loaders do not assign calibration or test partitions.

```bash
python -m unittest discover -s tests -v
```

Validation performed: 8 unit tests; all 369 live OSWorld task definitions; one live AgentRewardBench trajectory joined with its official annotation. Full 1,302-trajectory ingestion has not yet been run. Count checks are explicit so a changed or incomplete source cannot silently pass.

## Sources

- https://github.com/McGill-NLP/agent-reward-bench
- https://huggingface.co/datasets/McGill-NLP/agent-reward-bench
- https://github.com/xlang-ai/OSWorld
