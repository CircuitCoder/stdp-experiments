#!/usr/bin/env python3
"""Collect completed and failed pilot records without discarding provenance."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = []
    for directory in sorted(args.root.iterdir()):
        if directory.name.startswith(('train_', 'probe_')):
            for result_path in sorted(directory.glob('*/result.json')):
                record = {'group': directory.name, 'case': result_path.parent.name,
                          'artifact_path': str(result_path.parent),
                          'result': json.loads(result_path.read_text())}
                manifest_path = result_path.with_name('manifest.json')
                if manifest_path.is_file():
                    record['manifest'] = json.loads(manifest_path.read_text())
                records.append(record)
        elif directory.name.startswith('morrison_') and directory.is_dir():
            path = directory / 'result.json'
            if path.is_file():
                record = {'group': 'morrison', 'case': directory.name,
                          'artifact_path': str(directory), 'result': json.loads(path.read_text()),
                          'manifest': json.loads((directory / 'manifest.json').read_text())}
                memory = directory / 'allocated_gpu_memory.json'
                if memory.is_file():
                    record['gpu_memory'] = json.loads(memory.read_text())
                records.append(record)
    evidence = {}
    for name in ('local_machine.json', 'remote_machine.json', 'execution_initial.json',
                 'cifar_download.json', 'transfer_note.json', 'fashion_gpu_activity_comparison.json',
                 'cifar_gpu_activity_comparison.json', 'cifar_gpu_repeat_comparison.json'):
        path = args.root / name
        if path.is_file():
            evidence[name] = json.loads(path.read_text())
    snapshots = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(args.root.glob('source_v*.tar.gz'))}
    with args.output.open('x') as stream:
        json.dump({'schema': 'workload-pilots-summary-v1', 'runs': records,
                   'evidence': evidence, 'source_archive_sha256': snapshots},
                  stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    for record in records:
        result = record['result']
        print(record['group'], record['case'], result['status'],
              'accuracy', result.get('evaluation', {}).get('accuracy_percent'),
              'us/step', result.get('us_per_step'))


if __name__ == '__main__':
    main()
