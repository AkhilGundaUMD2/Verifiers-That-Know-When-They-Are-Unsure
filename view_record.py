# Use the shared loader so the viewer shows the same records as an import.
import argparse
from itertools import islice
from pprint import pprint

from loaders.benchmarks import load_agentrewardbench

# Choose a one-based record number and optionally display its recorded actions.
parser = argparse.ArgumentParser(description="View an AgentRewardBench record.")
parser.add_argument("number", type=int, help="Record number, starting at 1")
parser.add_argument("--steps", action="store_true", help="Show recorded steps")
args = parser.parse_args()

# Reject invalid positions before reading any dataset files.
if args.number < 1:
    parser.error("Record number must be at least 1.")

# Read from the project's local copy of AgentRewardBench and its annotations.
records = load_agentrewardbench(
    "data/raw/agentrewardbench",
    "data/raw/agentrewardbench/annotations.csv",
)

# Skip to the requested record without building a list of the whole dataset.
# next returns None when the requested position is beyond the available records.
record = next(islice(records, args.number - 1, args.number), None)

if record is None:
    parser.error(f"Record {args.number} does not exist.")

# Keep the default display focused on the instruction, labels, and annotations.
pprint({
    "task_id": record["task_id"],
    "instruction": record["instruction"],
    "number_of_steps": len(record["steps"]),
    "labels": record["labels"],
    "annotations": record["annotations"],
})

# With --steps, show each action in order and handle steps without an action field.
if args.steps:
    for number, step in enumerate(record["steps"], start=1):
        print(f"\nStep {number}:")
        print(step.get("action", "(No action recorded)"))
