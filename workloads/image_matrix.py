#!/usr/bin/env python3
"""Run the four image datasets across the three dense baseline learning rules."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workloads.provenance import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fashion-path', type=Path, required=True)
    parser.add_argument('--cifar-path', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', choices=['fashion-mnist', 'cifar10'],
                        default=['fashion-mnist', 'cifar10'])
    parser.add_argument('--rules', nargs='+', type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument('--backend', choices=['cuda', 'single_threaded_cpu', 'nest'], default='cuda')
    parser.add_argument('--train-samples', type=int, default=1000)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--nest-prefix', type=Path)
    parser.add_argument('--module', type=Path)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    commands, results = [], []
    for dataset in args.datasets:
        for subset in ('full', 'top2'):
            for rule in args.rules:
                name = f'{dataset}_{subset}_{rule}trace'
                call = [sys.executable, str(ROOT / 'workloads/images.py'),
                    '--dataset', dataset, '--data-path', str(args.fashion_path if dataset == 'fashion-mnist' else args.cifar_path),
                    '--output', str(args.output / name), '--rule', str(rule),
                    '--backend', args.backend, '--train-samples', str(args.train_samples)]
                if subset == 'top2':
                    call += ['--top-classes', '2']
                if args.evaluate:
                    call += ['--evaluate']
                if args.checkpoint_root:
                    call += ['--checkpoint', str(args.checkpoint_root / name / 'checkpoint.npz')]
                if args.backend == 'nest':
                    if not args.nest_prefix or not args.module:
                        parser.error('NEST requires prefix and module')
                    call += ['--nest-prefix', str(args.nest_prefix), '--module', str(args.module),
                             '--threads', str(args.threads)]
                commands.append({'case': name, 'command': call})
    write_json(args.output / 'commands.json', commands)
    for entry in commands:
        print(f"START {entry['case']}", flush=True)
        with (args.output / (entry['case'] + '.log')).open('x') as stream:
            completed = subprocess.run(entry['command'], stdout=stream, stderr=subprocess.STDOUT)
        results.append({'case': entry['case'], 'returncode': completed.returncode})
        print(f"END {entry['case']} returncode={completed.returncode}", flush=True)
    write_json(args.output / 'completion.json', results)
    if any(row['returncode'] for row in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
