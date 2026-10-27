"""Assign processed records to calibration, in-distribution, and OOD test splits."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ('calibration', 'id_test', 'ood_test')


def group_rank(group_id, seed):
    # Hash with the seed so ordering is reproducible yet unrelated to record order.
    return hashlib.sha256(f'{seed}:{group_id}'.encode('utf-8')).hexdigest()


def assign_splits(records, ood_domains, calibration_fraction=0.5, seed=0):
    """Map record ID to split; every record of a group_id lands in one split."""
    if not 0 < calibration_fraction < 1:
        raise ValueError('calibration_fraction must be between 0 and 1')
    ood_domains = set(ood_domains)
    # Collect each group's records and domains before deciding anything.
    groups, domains, seen = defaultdict(list), defaultdict(set), set()
    for record in records:
        if record['id'] in seen:
            raise ValueError(f"Duplicate record ID: {record['id']}")
        seen.add(record['id'])
        groups[record['group_id']].append(record['id'])
        domains[record['group_id']].add(record['domain'])
    if not groups:
        raise ValueError('No records to split')
    # Held-out domains must exist, and must leave in-distribution data behind.
    present = set().union(*domains.values())
    if missing := ood_domains - present:
        raise ValueError(f'OOD domains not present in records: {sorted(missing)}')
    if not present - ood_domains:
        raise ValueError('OOD domains cover every record; nothing left in distribution')
    assignment, in_distribution = {}, []
    for group_id, group_domains in domains.items():
        # A group straddling the OOD boundary would leak held-out tasks.
        if group_domains & ood_domains and group_domains - ood_domains:
            raise ValueError(f'Group spans OOD and in-distribution domains: {group_id}')
        if group_domains & ood_domains:
            for record_id in groups[group_id]:
                assignment[record_id] = 'ood_test'
        else:
            in_distribution.append(group_id)
    # Order groups by seeded hash and cut at an exact group count.
    in_distribution.sort(key=lambda group_id: group_rank(group_id, seed))
    cut = round(len(in_distribution) * calibration_fraction)
    if cut == 0 or cut == len(in_distribution):
        raise ValueError('Calibration fraction leaves calibration or id_test empty')
    for index, group_id in enumerate(in_distribution):
        split = 'calibration' if index < cut else 'id_test'
        for record_id in groups[group_id]:
            assignment[record_id] = split
    return assignment


def read_jsonl(path):
    # Yield one processed record per non-empty line.
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Processed JSONL from python -m loaders')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ood-domain', required=True, action='append', dest='ood_domains',
                        help='Domain held out as OOD test; repeat to hold out several')
    parser.add_argument('--calibration-fraction', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    # Keep only the fields needed for splitting and summary counts.
    records = [{k: r[k] for k in ('id', 'group_id', 'domain', 'labels')} for r in read_jsonl(args.input)]
    assignment = assign_splits(records, args.ood_domains, args.calibration_fraction, args.seed)
    # Summarize sizes, domains, and unknown labels per split for documentation.
    summary = {split: {'records': 0, 'groups': set(), 'domains': Counter(), 'unknown_labels': Counter()}
               for split in SPLITS}
    for record in records:
        entry = summary[assignment[record['id']]]
        entry['records'] += 1
        entry['groups'].add(record['group_id'])
        entry['domains'][record['domain']] += 1
        entry['unknown_labels'].update(k for k, v in record['labels'].items() if v is None)
    summary = {split: {'records': e['records'], 'groups': len(e['groups']),
                       'domains': dict(e['domains']), 'unknown_labels': dict(e['unknown_labels'])}
               for split, e in summary.items()}
    # Hash in 1 MiB chunks; processed inputs can be larger than available memory.
    digest = hashlib.sha256()
    with args.input.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    output = {'config': {'input': args.input.as_posix(),
                         'input_sha256': digest.hexdigest(),
                         'ood_domains': sorted(set(args.ood_domains)),
                         'calibration_fraction': args.calibration_fraction, 'seed': args.seed},
              'summary': summary, 'assignment': dict(sorted(assignment.items()))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
