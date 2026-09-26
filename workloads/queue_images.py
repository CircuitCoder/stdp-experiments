#!/usr/bin/env python3
"""Execute a recorded image experiment plan with one active case at a time."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def run_plan(plan_path):
    plan = json.loads(plan_path.read_text())
    root = Path(plan['artifact_root'])
    root.mkdir(parents=True, exist_ok=True)
    source = Path(plan['source_root'])
    names = [case['name'] for case in plan['cases']]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate case names')
    stages = [case['stage'] for case in plan['cases']]
    if any(stage not in ('cifar_pilot', 'fashion_long') for stage in stages):
        raise ValueError('Unknown experiment stage')
    if stages != sorted(stages, key=lambda stage: stage == 'fashion_long'):
        raise ValueError('All CIFAR pilots must precede every Fashion long run')
    for case in plan['cases']:
        if Path(case['output']).exists() or Path(case['log']).exists():
            raise FileExistsError(f"Case output/log already exists: {case['name']}")
    for relative, expected in plan['source_sha256'].items():
        if sha256(source / relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')

    lock_path = Path(plan['lock_path'])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another image experiment queue holds the lock') from exc
        write_new(root / 'queue_started.json',
                  {'created_utc': now(), 'plan_sha256': sha256(plan_path), 'case_count': len(names)})
        (root / 'case_results').mkdir(exist_ok=False)
        (root / 'logs').mkdir(exist_ok=True)
        results = []

        def event(value):
            value = {'utc': now(), **value}
            with (root / 'queue_events.jsonl').open('a') as stream:
                stream.write(json.dumps(value, allow_nan=False) + '\n')
            print(json.dumps(value, allow_nan=False), flush=True)

        event({'event': 'queue_started', 'case_count': len(names)})
        try:
            for position, case in enumerate(plan['cases']):
                if (case['stage'] == 'fashion_long'
                        and not (root / 'pilot_summary.json').exists()):
                    write_new(root / 'pilot_summary.json',
                              {'completed_utc': now(), 'cases': results.copy()})
                    event({'event': 'cifar_pilots_finished', 'case_count': len(results)})
                # A case's estimate includes all its checkpoints and evaluations.
                # Preserve a reserve on both the artifact and compilation filesystems.
                for directory, required in ((root, case.get('required_artifact_bytes', 0)),
                                             (Path(plan['build_root']), case.get('required_build_bytes', 0))):
                    directory.mkdir(parents=True, exist_ok=True)
                    if shutil.disk_usage(directory).free < required + plan.get('disk_reserve_bytes', 0):
                        raise RuntimeError(f'Insufficient free space before {case["name"]}: {directory}')
                event({'event': 'case_started', 'position': position + 1,
                       'case': case['name'], 'stage': case['stage'], 'command': case['command']})
                started = time.monotonic()
                with Path(case['log']).open('x') as log:
                    child = subprocess.run(case['command'], cwd=plan['cwd'],
                                           stdout=log, stderr=subprocess.STDOUT, check=False)
                output = Path(case['output'])
                result_path = output / 'result.json'
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                evaluation_path = output / 'evaluations.jsonl'
                evaluations = ([json.loads(line) for line in evaluation_path.read_text().splitlines()
                                if line.strip()] if evaluation_path.exists() else [])
                last = evaluations[-1] if evaluations else None
                status = 'complete' if child.returncode == 0 and result.get('status') == 'complete' else 'failed'
                if status == 'complete':
                    if (result['accepted_samples'] != case['expected_accepted_samples'] or last is None
                            or last['accepted_samples'] != case['expected_accepted_samples']
                            or last['full_test'] != case['expected_full_test']):
                        raise RuntimeError(f'Incomplete result reported as complete: {case["name"]}')
                record = {'case': case['name'], 'stage': case['stage'], 'status': status,
                          'returncode': child.returncode, 'completed_utc': now(),
                          'wall_seconds': time.monotonic() - started,
                          'output': str(output), 'log': case['log'],
                          'accepted_samples': result.get('accepted_samples'),
                          'attempts': result.get('attempts'), 'retries': result.get('retries'),
                          'error': result.get('error'),
                          'failure_context': result.get('failure_context'),
                          'failed_presentation': result.get('failed_presentation'),
                          'last_evaluation': last,
                          'result_sha256': sha256(result_path) if result_path.exists() else None}
                write_new(root / 'case_results' / f'{position + 1:02d}_{case["name"]}.json', record)
                results.append(record)
                event({'event': 'case_finished', 'position': position + 1, 'case': case['name'],
                       'stage': case['stage'], 'status': status, 'returncode': child.returncode,
                       'accuracy_percent': last['accuracy_percent'] if last else None})
            if not (root / 'pilot_summary.json').exists():
                write_new(root / 'pilot_summary.json', {'completed_utc': now(), 'cases': results.copy()})
            write_new(root / 'queue_result.json', {'status': 'finished', 'completed_utc': now(),
                      'completed_cases': sum(item['status'] == 'complete' for item in results),
                      'failed_cases': sum(item['status'] == 'failed' for item in results), 'cases': results})
            event({'event': 'queue_finished', 'case_count': len(results)})
        except BaseException as exc:
            write_new(root / 'queue_failure.json', {'utc': now(), 'error': repr(exc),
                      'completed_case_count': len(results)})
            event({'event': 'queue_failed', 'error': repr(exc)})
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    run_plan(parser.parse_args().plan)
