# Parse command-line options, serialize records, and track source-file integrity.
import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from .benchmarks import load_agentrewardbench, load_osworld


def main():
    # Select dataset and inputs. Exactly one output mode is required.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=['agentrewardbench', 'osworld'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--manifest', default='test_all.json')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--validate-only', action='store_true',
                      help='Read original files and report counts without writing files')
    parser.add_argument('--expected-count', type=int)
    args = parser.parse_args()
    # AgentRewardBench needs separate CSV to  attach human annotations.
    if args.dataset == 'agentrewardbench' and not args.annotations:
        parser.error('--annotations is required for agentrewardbench')
    # Both loaders return generators, so records are read one at a time.
    records = (load_agentrewardbench(args.root, args.annotations) if args.dataset == 'agentrewardbench'
               else load_osworld(args.root, args.manifest))
    if args.validate_only:
        # Consume the loader to run checks w/o creating output files.
        # Count missing or uncertain labels separately for each label type.
        counts, unknown, total = Counter(), Counter(), 0
        for record in records:
            total += 1
            counts[record['domain']] += 1
            for label, value in record['labels'].items():
                if value is None:
                    unknown[label] += 1
            del record
            # Report progress during large imports; flushing makes it visible immediately.
            if total % 100 == 0:
                print(f'Checked {total} records', flush=True)
        # An optional expected size catches incomplete or unexpected inputs.
        if args.expected_count is not None and total != args.expected_count:
            raise ValueError(f'Expected {args.expected_count} records; loaded {total}')
        print(json.dumps({'count': total, 'domains': dict(counts),
                          'unknown_labels': dict(unknown),
                          'mode': 'validate-only', 'files_written': 0}, indent=2))
        return
    # Prepare the destination and resolve the base for relative source paths.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts, sources, total = Counter(), {}, 0
    base = args.root
    if args.dataset == 'osworld' and (base / 'evaluation_examples').is_dir():
        base = base / 'evaluation_examples'
    def hash_file(path):
        # Hash in 1 MiB chunks so large files need little working memory.
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    # Stage JSONL beside the destination so replacement stays on one filesystem.
    # The existing output is replaced only after loading and count checks succeed.
    fd, temporary = tempfile.mkstemp(dir=args.output.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            for record in records:
                # Write one complete JSON object per line and record its source hash.
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
                total += 1
                counts[record['domain']] += 1
                sources[record['source_path']] = hash_file(base / record['source_path'])
        if args.expected_count is not None and total != args.expected_count:
            raise ValueError(f'Expected {args.expected_count} records; loaded {total}')
        # Save counts and hashes in a companion manifest for reproducibility.
        metadata = {'dataset': args.dataset, 'count': total, 'domains': dict(counts),
                    'source_sha256': sources, 'output_sha256': hash_file(Path(temporary))}
        if args.annotations:
            metadata['annotations_sha256'] = hash_file(args.annotations)
        if args.dataset == 'osworld':
            metadata['manifest_sha256'] = hash_file(base / args.manifest)
        # Atomically publish the records, then write their companion manifest.
        os.replace(temporary, args.output)
        args.output.with_suffix('.manifest.json').write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        print(json.dumps({'count': total, 'domains': dict(counts), 'output': str(args.output)}, indent=2))
    finally:
        # Remove an unfinished temporary file
        Path(temporary).unlink(missing_ok=True)


if __name__ == '__main__':
    main()
