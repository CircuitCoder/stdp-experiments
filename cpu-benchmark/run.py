#!/usr/bin/env python3
"""Optimized GeNN CPU single-network latency, pinned to one logical CPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reimpl'))
sys.path.insert(0, str(ROOT / 'brunel'))
sys.path.insert(0, str(ROOT / 'genn-sweep'))
# baseline imports the older sweep as `run`; import it before this file can
# become a named module in a spawned worker.
import baseline
from ports.common import DT_MS, make_genn_default_model, weight_stats
from ports.genn_port import GeNNBrunel
from zd3.io import sha256_file
from energy import PackageEnergy


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_json(path, value)


def brunel(args, name):
    spec = make_genn_default_model(baseline.sweep.BRUNEL_CASES[name])
    diagnostic = args.mode == 'diagnostic'
    build = args.output / 'builds' / name
    reuse = args.reuse_build_root / name if args.reuse_build_root else None
    network = GeNNBrunel(spec=spec, seed=args.seed, state_seed=args.seed,
        backend='single_threaded_cpu', build_path=build, recording_steps=11000,
        precision='float', stdp_timing='arrival', stdp_tie_order='nest_causal_boundary',
        timing_enabled=False, reuse_build=reuse, record_spikes=False,
        collect_connectivity_stats=diagnostic)
    try:
        network.step(1000, synchronize=False)
        before = network.population_spike_counts()
        args.on_ready()
        started = time.perf_counter()
        bins = []
        if diagnostic:
            previous = before[0]
            for offset in range(0, 10000, 30):
                network.step(min(30, 10000 - offset), synchronize=False)
                current = network.population_spike_counts()[0]
                if offset + 30 <= 10000:
                    bins.append(int((current - previous).sum()))
                previous = current
        else:
            network.step(10000, synchronize=False)
        after = network.population_spike_counts()
        wall = time.perf_counter() - started
        args.on_finished()
        exc, inh = (after[i] - before[i] for i in (0, 1))
        result = {'wall_seconds': wall, 'simulation_steps': 10000,
            'us_per_step': wall * 100, 'total_spikes': int(exc.sum() + inh.sum()),
            'event_counters': {'excitatory_spikes': int(exc.sum()), 'inhibitory_spikes': int(inh.sum())},
            'exc_rate_hz': float(exc.mean()), 'inh_rate_hz': float(inh.mean())}
        if not 0.1 < result['exc_rate_hz'] < 100 or not 0.1 < result['inh_rate_hz'] < 100:
            raise RuntimeError(f'Brunel silence/runaway: {result}')
        if diagnostic:
            state = {}
            for pop_name, pop in [('E', network.exc), ('I', network.inh)]:
                state[pop_name] = {}
                for variable_name in ['V', 'Iex', 'Iin', 'dIex', 'dIin']:
                    variable = pop.vars[variable_name]
                    variable.pull_from_device()
                    values = np.asarray(variable.view)
                    if not np.all(np.isfinite(values)):
                        raise RuntimeError(f'Nonfinite {pop_name}/{variable_name}')
                    state[pop_name][variable_name] = [float(values.min()), float(values.max())]
            weights = network.sample_weights(100000)
            if not np.all(np.isfinite(weights)) or weights.min() < 0:
                raise RuntimeError('Invalid Brunel weights')
            if spec.rule.weight_max_pa and weights.max() > spec.rule.weight_max_pa * 1.00001:
                raise RuntimeError('Brunel weight exceeds bound')
            result['diagnostic'] = {'state_ranges': state,
                'weight_min': float(weights.min()), 'weight_max': float(weights.max()),
                'weight_mean': float(weights.mean()), 'weight_std': float(weights.std()),
                'connectivity': network.event_counters(before, after),
                'exc_spikes_per_neuron': exc.tolist(), 'guards_passed': True}
            histogram = np.asarray(bins)
            result['diagnostic']['population_fano_3ms'] = float(histogram.var() / histogram.mean())
        return result
    finally:
        network.close()


def worker(options, case, index, messages, start, release):
    args = argparse.Namespace(**options)
    try:
        if args.pin:
            cpus = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {cpus[index % len(cpus)]})
        args.backend = 'single_threaded_cpu'
        def ready():
            messages.put(('ready', index, None))
            if not start.wait(600):
                raise TimeoutError('Batch start timed out')
        def finished():
            messages.put(('timed', index, None))
            if not release.wait(600):
                raise TimeoutError('Batch completion timed out')
        args.on_ready, args.on_finished = ready, finished
        with baseline.sweep.redirect_process_output(args.output / 'logs' / f'worker-{index}.log'):
            if case in baseline.CASES:
                manifest = json.loads((args.inputs / 'manifest.json').read_text())
                result = baseline.mnist_run(args, case, manifest['cases'][case])
            else:
                result = brunel(args, case)
            result['cpu_affinity'] = sorted(os.sched_getaffinity(0))
            result['simulation_backend'] = args.backend
            write(args.output / f'worker-{index}.json', result)
        messages.put(('result', index, result))
    except BaseException:
        messages.put(('error', index, traceback.format_exc()))


def receive(messages, processes, kind, count):
    received = {}
    deadline = time.monotonic() + 900
    while len(received) < count:
        try:
            tag, index, value = messages.get(timeout=1)
        except queue.Empty:
            if time.monotonic() > deadline or any(p.exitcode not in (None, 0) for p in processes):
                raise RuntimeError(f'Worker exited or timed out waiting for {kind}')
            continue
        if tag == 'error':
            raise RuntimeError(f'Worker {index}:\n{value}')
        if tag != kind:
            raise RuntimeError(f'Expected {kind}, got {tag}')
        received[index] = value
    return [received[i] for i in range(count)]


def batch(args, case, output, count, diagnostic=False):
    if count != 1:
        raise ValueError('Latency measurements require exactly one network; replicas measure throughput')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    context = mp.get_context('spawn')
    messages, start, release = context.Queue(), context.Event(), context.Event()
    options = dict(vars(args), output=output, mode='diagnostic' if diagnostic else 'timing',
        reuse_build_root=None if diagnostic else args.output / 'diagnostic' / case / 'builds')
    processes = [context.Process(target=worker, args=(options, case, i, messages, start, release))
                 for i in range(count)]
    energy = PackageEnergy()
    try:
        for process in processes:
            process.start()
        receive(messages, processes, 'ready', count)
        load_before = os.getloadavg()
        energy.start()
        started = time.perf_counter()
        start.set()
        receive(messages, processes, 'timed', count)
        wall = time.perf_counter() - started
        measured_energy = energy.stop()
        release.set()
        results = receive(messages, processes, 'result', count)
        for process in processes:
            process.join(timeout=30)
        steps = results[0]['simulation_steps']
        result = {'case': case, 'metric': 'single_network_latency',
            'replicas': 1, 'simulation_threads_per_replica': 1,
            'batch_wall_seconds': wall, 'total_simulation_steps': steps,
            'us_per_step': results[0]['us_per_step'],
            'energy': measured_energy,
            'microjoules_per_step': measured_energy['joules'] * 1e6 / steps
                if measured_energy['joules'] is not None else None,
            'load_average_before': load_before, 'load_average_after': os.getloadavg(),
            'workers': results}
        write(output / 'batch.json', result)
        return result
    finally:
        release.set()
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'copilot/tmp/baseline_inputs_20260907_v2')
    parser.add_argument('--data-path', type=Path, default=ROOT / 'data/mnist')
    parser.add_argument('--cases', nargs='+', choices=baseline.ALL_CASES, default=baseline.ALL_CASES)
    parser.add_argument('--workers', type=int, choices=[1], default=1,
        help='Exactly one network; retained for compatibility with earlier commands')
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260724)
    parser.add_argument('--pin', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.workers, args.repetitions, args.samples, args.seed) <= 0:
        parser.error('workers, repetitions, samples and seed must be positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    import pygenn
    inputs = json.loads((args.inputs / 'manifest.json').read_text())
    for relative, expected in inputs['dataset_sha256'].items():
        if sha256_file(args.data_path / Path(relative).name) != expected:
            raise ValueError(f'Dataset hash mismatch: {relative}')
    source_paths = [Path(__file__), Path(__file__).with_name('energy.py'),
        Path(__file__).with_name('shell.nix'), ROOT / 'genn-sweep/run.py',
        ROOT / 'genn-sweep/baseline.py', ROOT / 'genn-sweep/baseline_cases.py',
        ROOT / 'reimpl/backends/genn_backend.py', ROOT / 'brunel/ports/genn_port.py',
        ROOT / 'brunel/ports/common.py', ROOT / 'reimpl/zd3/variants.py',
        ROOT / 'reimpl/zd3/io.py', ROOT / 'reimpl/zd3/constants.py']
    write(args.output / 'manifest.json', {
        'schema': 'genn-cpu-latency-v2', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'cwd': str(Path.cwd()), 'python': sys.version,
        'genn': pygenn.__version__, 'numpy': np.__version__, 'host': platform.uname()._asdict(),
        'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'allowed_cpus': sorted(os.sched_getaffinity(0)),
        'input_manifest_sha256': sha256_file(args.inputs / 'manifest.json'),
        'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        'environment': {k: os.environ.get(k) for k in ['CXX', 'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH',
            'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NIX_CFLAGS_COMPILE',
            'NIX_ENFORCE_NO_NATIVE']},
        'cpu': baseline.command_output(['lscpu']), 'compiler': baseline.command_output(['g++', '--version']),
        'git_revision': baseline.command_output(['git', 'rev-parse', 'HEAD']),
        'genn_revision': baseline.command_output(['git', '-C', str(ROOT / '3rdparty/genn'), 'rev-parse', 'HEAD']),
        'git_status': baseline.command_output(['git', 'status', '--short']),
        'processes_before': baseline.command_output(['ps', '-eo', 'pid,comm,nlwp,pcpu,pmem', '--sort=-pcpu']),
        'power': PackageEnergy().metadata(), 'precision': 'FP32; host normalization FP64',
        'protocol': 'one network at a time; single simulation thread; worker elapsed wall time divided by its executed steps; setup and parent process rendezvous excluded'})
    for p in source_paths:
        target = args.output / 'source' / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    summary = {}
    for case in args.cases:
        print(f'DIAGNOSTIC case={case}', flush=True)
        diagnostic = batch(args, case, args.output / 'diagnostic' / case, 1, diagnostic=True)
        samples = []
        for repetition in range(args.repetitions):
            result = batch(args, case, args.output / 'timing' / case / f'r{repetition+1}', args.workers)
            # Every repetition starts from identical state and must reproduce
            # the CPU diagnostic despite compilation reuse.
            control = diagnostic['workers'][0]
            for measured in result['workers']:
                for key in ['simulation_steps', 'total_spikes', 'event_counters']:
                    if measured[key] != control[key]:
                        raise RuntimeError(f'Diagnostic/timing mismatch: {case}/{key}')
            samples.append(result['us_per_step'])
            print(f'TIMING case={case} repetition={repetition+1} networks=1 threads=1 '
                  f'us_per_step={samples[-1]:.6f} wall_seconds={result["workers"][0]["wall_seconds"]:.3f}', flush=True)
        summary[case] = {'metric': 'single_network_latency',
            'median_us_per_step': float(np.median(samples)),
            'range': [min(samples), max(samples)], 'iqr': float(np.subtract(*np.percentile(samples, [75, 25]))),
            'replicas': args.workers, 'repetitions': args.repetitions, 'samples': samples,
            'diagnostic': diagnostic['workers'][0]}
        write(args.output / f'{case}.json', summary[case])
    write(args.output / 'summary.json', summary)
    print(f'COMPLETE output={args.output}', flush=True)


if __name__ == '__main__':
    main()
