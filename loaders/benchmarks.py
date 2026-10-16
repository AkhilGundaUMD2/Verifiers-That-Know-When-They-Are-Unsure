"""Load official benchmark files into provenance-preserving JSON records."""
import csv
import json
from collections import defaultdict
from pathlib import Path


def normalize_task_id(task_id):
    # Match alternate trajectory filenames to their shared annotation task ID.
    return task_id.replace('.improved', '').replace('.resized', '')


def read_json(path):
    # Read all benchmark JSON consistently as UTF-8.
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def consensus(rows, column, mapping):
    """Disagreement, Unsure, and absence remain unknown, never negative."""
    # A single distinct mapped value means all annotators agree.
    # Empty rows or conflicting values produce None; unanimous Unsure does too.
    values = {mapping[row[column]] for row in rows}
    return values.pop() if len(values) == 1 else None


def load_agentrewardbench(root, annotations):
    # The cleaned directory contains recorded trajectories, not tasks to execute.
    root = Path(root)
    cleaned = root / 'cleaned'
    if not cleaned.is_dir():
        raise FileNotFoundError(f'Missing cleaned trajectory directory: {cleaned}')
    # Index CSV rows by benchmark, agent, and normalized task ID for the join.
    grouped = defaultdict(list)
    with Path(annotations).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        # Fail early if the CSV cannot supply the join keys or expected labels.
        required = {'benchmark', 'model_name', 'task_id', 'trajectory_success',
                    'trajectory_side_effect', 'trajectory_looping'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Annotations require columns: {sorted(required)}')
        for row in reader:
            key = (row['benchmark'], row['model_name'], normalize_task_id(row['task_id']))
            grouped[key].append(row)
    # Preserve uncertainty with None instead of treating it as a failed task.
    success = {'Successful': True, 'Unsuccessful': False, 'Unsure': None}
    yes_no = {'Yes': True, 'No': False, 'Unsure': None}
    # Track both full record IDs and normalized keys to reject ambiguous joins.
    seen = set()
    seen_keys = set()
    # Sorting makes the emitted record order reproducible across imports.
    for path in sorted(cleaned.rglob('*.json')):
        raw = read_json(path)
        task_id = path.stem
        benchmark, agent = raw['benchmark'], raw['agent']
        key = (benchmark, agent, normalize_task_id(task_id))
        record_id = '/'.join((benchmark, agent, task_id))
        if record_id in seen:
            raise ValueError(f'Duplicate trajectory ID: {record_id}')
        seen.add(record_id)
        if not isinstance(raw['goal'], str) or not isinstance(raw['steps'], list):
            raise ValueError(f'Invalid goal or steps: {path}')
        if key in seen_keys:
            raise ValueError(f'Duplicate annotation join key: {key}')
        seen_keys.add(key)
        # Consume matching annotations; leftovers reveal unmatched CSV entries.
        rows = grouped.pop(key, [])
        # Emit a common record schema while preserving steps, annotations, and
        # extra source fields. group_id links variants of the same task.
        yield {'dataset': 'agentrewardbench', 'record_type': 'trajectory',
               'id': record_id, 'task_id': task_id,
               'group_id': 'agentrewardbench/' + normalize_task_id(task_id),
               'domain': benchmark, 'agent': agent, 'instruction': raw['goal'],
               'steps': raw['steps'], 'valid': raw.get('valid'),
               'labels': {'success': consensus(rows, 'trajectory_success', success),
                          'side_effect': consensus(rows, 'trajectory_side_effect', yes_no),
                          'looping': consensus(rows, 'trajectory_looping', yes_no)},
               'annotations': rows, 'source_path': path.relative_to(root).as_posix(),
               'metadata': {k: v for k, v in raw.items() if k not in {'goal', 'steps'}}}
    # These final checks run when the generator has been fully consumed.
    if not seen:
        raise ValueError('No trajectory JSON files found')
    if grouped:
        raise ValueError(f'{len(grouped)} annotated trajectories have no matching JSON files')


def load_osworld(root, manifest='test_all.json'):
    # Accept either the repository root or evaluation_examples itself.
    root = Path(root)
    base = root / 'evaluation_examples' if (root / 'evaluation_examples').is_dir() else root
    # The manifest selects task IDs by domain; it does not contain agent runs.
    index = read_json(base / manifest)
    if not isinstance(index, dict) or not index:
        raise ValueError('OSWorld manifest must map domains to task ID lists')
    seen = set()
    for domain, ids in sorted(index.items()):
        if not isinstance(ids, list):
            raise ValueError(f'Invalid task list for {domain}')
        for task_id in sorted(ids):
            # Require single path components before using manifest values in paths.
            if not isinstance(task_id, str) or Path(task_id).name != task_id or Path(domain).name != domain:
                raise ValueError('Unsafe task ID or domain')
            # Task IDs must be unique even when listed under different domains.
            if task_id in seen:
                raise ValueError(f'Duplicate OSWorld task ID: {task_id}')
            seen.add(task_id)
            path = base / 'examples' / domain / f'{task_id}.json'
            raw = read_json(path)
            if raw['id'] != task_id or not isinstance(raw['instruction'], str):
                raise ValueError(f'Invalid task ID or instruction: {path}')
            # Task definitions have no observed steps or outcome labels.
            # Preserve evaluator/configuration data for later use without running it.
            yield {'dataset': 'osworld', 'record_type': 'task_definition',
                   'id': task_id, 'task_id': task_id, 'group_id': 'osworld/' + task_id,
                   'domain': domain, 'agent': None, 'instruction': raw['instruction'],
                   'steps': None, 'labels': {'success': None, 'side_effect': None, 'looping': None},
                   'evaluator': raw['evaluator'], 'config': raw.get('config', []),
                   'source_path': path.relative_to(base).as_posix(),
                   'metadata': {k: v for k, v in raw.items()
                                if k not in {'instruction', 'evaluator', 'config'}}}
