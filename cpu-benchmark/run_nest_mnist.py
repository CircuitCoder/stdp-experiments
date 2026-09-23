#!/usr/bin/env python3
"""Single-network NEST latency for the frozen 21-case MNIST benchmark matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'reimpl'))
sys.path.insert(0, str(ROOT / 'genn-sweep'))
import baseline
from baseline_cases import mnist_cases
from zd3.constants import MODEL, ModelConstants
from zd3.io import load_checkpoint, load_mnist, sha256_file
from zd3.variants import validate_checkpoint_topology

CASES = mnist_cases()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def counts(pop):
    return np.asarray(pop.get('spike_count'), dtype=np.int64)


def learning_parameters(variant):
    rule = {'one-trace-power': 1, 'two-trace-power': 2, 'three-trace': 3}[variant.learning_rule]
    return {'rule': rule, 'pre_tau': MODEL.pre_tau_ms, 'post1_tau': MODEL.post1_tau_ms,
        'post2_tau': MODEL.post2_tau_ms, 'potentiation': variant.potentiation_rate,
        'depression': variant.depression_rate, 'weight_max': variant.weight_max,
        'pre_target': variant.pre_trace_target, 'pre_exponent': variant.pre_weight_exponent,
        'post_exponent': variant.post_weight_exponent}


class Network:
    def __init__(self, nest, args, variant, checkpoint, mask, *, constants: ModelConstants = MODEL,
                 plasticity=True, inhibition=None):
        self.constants = constants
        self.nest, self.variant, self.mask = nest, variant, mask
        self.pre_indices = [np.flatnonzero(mask[:, j]) for j in range(self.constants.n_exc)]
        self._count_baseline = np.zeros(self.constants.n_exc, dtype=np.int64)
        self.normalization_observation = {'attempts_above_wmax': 0,
            'attempts_above_102pct_wmax': 0, 'maximum_weight_over_wmax': 0.0}
        nest.ResetKernel()
        nest.SetKernelStatus({'resolution': self.constants.dt_ms, 'local_num_threads': args.threads,
            'rng_seed': args.seed, 'print_time': False})
        nest.Install(str(args.module.with_suffix('')))
        nest.verbosity = nest.VerbosityLevel.ERROR
        p = learning_parameters(variant) | {'plasticity': float(plasticity)}
        if not plasticity:
            p.update(theta_plus=0.0, theta_decay=1.0)
        self.inputs = nest.Create('cpu_mnist_neuron', self.constants.n_input, p | {'kind': 0})
        capacity = self.inputs[0].get('input_capacity')
        if self.constants.n_input > capacity:
            raise ValueError(f'NEST module input capacity {capacity} < {self.constants.n_input}')
        self.inputs.set({'input_index': np.arange(self.constants.n_input, dtype=np.int64)})
        self.exc = nest.Create('cpu_mnist_neuron', self.constants.n_exc, p | {'kind': 1})
        self.exc.set([{'theta': float(np.float32(checkpoint.theta_mv[j])),
            'pre_indices': pre.tolist(),
            'ff_weights': checkpoint.weights[pre, j].astype(np.float32).astype(float).tolist()}
            for j, pre in enumerate(self.pre_indices)])
        self.inh = nest.Create('cpu_mnist_neuron', self.constants.n_inh, {'kind': 2,
            'V': self.constants.inh_v_rest_mv - 40, 'tau_m': self.constants.inh_tau_m_ms,
            'v_rest': self.constants.inh_v_rest_mv, 'v_reset': self.constants.inh_v_reset_mv,
            'v_threshold': self.constants.inh_v_threshold_mv, 'e_inh': self.constants.inh_e_inh_mv,
            'refractory_steps': round(self.constants.inh_refractory_ms / self.constants.dt_ms), 'plasticity': 0.0})
        pre, post = np.nonzero(mask)
        sources = np.asarray(self.inputs, dtype=np.int64)[pre]
        targets = np.asarray(self.exc, dtype=np.int64)[post]
        nest.Connect(sources, targets, 'one_to_one', {'synapse_model': 'static_synapse',
            'weight': np.ones(len(pre)), 'delay': np.full(len(pre), self.constants.dt_ms),
            'receptor_type': (pre + 1).astype(np.int64)})
        nest.Connect(self.exc, self.inh, 'one_to_one', {'synapse_model': 'static_synapse_hpc',
            'weight': float(np.float32(self.constants.exc_to_inh_weight)), 'delay': self.constants.dt_ms})
        recurrent = np.full((self.constants.n_inh, self.constants.n_exc), -(self.constants.train_inhibition if inhibition is None else inhibition))
        np.fill_diagonal(recurrent, 0)
        nest.Connect(self.inh, self.exc, 'all_to_all', {'synapse_model': 'static_synapse_hpc',
            'weight': recurrent, 'delay': self.constants.dt_ms})
        self.num_ff_connections = len(nest.GetConnections(self.inputs, self.exc))
        if self.num_ff_connections != int(mask.sum()):
            raise RuntimeError('NEST structural connection count differs from frozen mask')
        if nest.min_delay != self.constants.dt_ms or nest.max_delay != self.constants.dt_ms:
            raise RuntimeError('All NEST MNIST connections must have one-step latency')
        nest.Prepare()

    def weights(self):
        result = np.zeros(self.mask.shape, dtype=np.float64)
        values = self.exc.get('ff_weights')
        for j, pre in enumerate(self.pre_indices):
            result[pre, j] = values[j]
        return result

    def normalize(self, diagnostic):
        # Eager FP64 sum/scale, followed by FP32 storage, on the owning NEST
        # neurons. No pending postsynaptic update is materialized early here.
        self.exc.set({'normalize_now': True})
        if diagnostic:
            weights = self.weights()
            ratio = float(weights.max() / self.variant.weight_max)
            self.normalization_observation['attempts_above_wmax'] += int(ratio > 1.0)
            self.normalization_observation['attempts_above_102pct_wmax'] += int(ratio > 1.02)
            self.normalization_observation['maximum_weight_over_wmax'] = max(
                ratio, self.normalization_observation['maximum_weight_over_wmax'])
            if not np.allclose(weights.sum(axis=0), 78, atol=1e-4, rtol=0):
                raise RuntimeError('NEST normalization did not produce column sums of 78')

    def set_image(self, image, intensity):
        self.inputs.set({'rate_hz': (image.astype(np.float64) / 8.0 * intensity).astype(np.float32)})

    def run_stimulus(self):
        self.nest.Run(self.constants.stimulus_ms)
        current = counts(self.exc)
        result = current - self._count_baseline
        self._count_baseline = current
        return result

    def run_rest(self):
        self.inputs.set({'rate_hz': 0.0})
        self.nest.Run(self.constants.rest_ms)

    def population_counts(self):
        return counts(self.inputs), counts(self.exc), counts(self.inh)

    def validate(self, stimulus_counts):
        if int(stimulus_counts.sum()) >= 5000:
            raise RuntimeError('NEST MNIST runaway stimulus')
        for pop in [self.exc, self.inh]:
            for field in ['V', 'ge', 'gi', 'theta']:
                values = np.asarray(pop.get(field))
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(f'Nonfinite NEST state: {field}')

    def close(self):
        self.nest.Cleanup()


def execute(nest, args, case, repetition, diagnostic):
    variant = CASES[case]
    inputs = json.loads((args.inputs / 'manifest.json').read_text())
    entry = inputs['cases'][case]
    path = args.inputs / entry['checkpoint']
    if sha256_file(path) != entry['checkpoint_sha256']:
        raise RuntimeError(f'Checkpoint hash mismatch: {case}')
    checkpoint = load_checkpoint(path)
    if checkpoint.manifest['variant'] != variant.as_dict():
        raise RuntimeError(f'Variant differs from frozen checkpoint: {case}')
    mask = validate_checkpoint_topology(checkpoint.weights, variant)
    data = load_mnist(args.data_path, 'train')
    construction_start = time.perf_counter()
    network = Network(nest, args, variant, checkpoint, mask)
    construction = time.perf_counter() - construction_start
    before_weights = network.weights()
    accepted = attempts = 0
    stimuli = []
    load_before = os.getloadavg()
    try:
        started = time.perf_counter()
        while accepted < args.samples:
            intensity = MODEL.initial_intensity
            while True:
                network.normalize(diagnostic)
                network.set_image(data.images[checkpoint.accepted_samples + accepted], intensity)
                spikes = network.run_stimulus()
                total = int(spikes.sum())
                if diagnostic:
                    network.validate(spikes)
                if total >= 5000:
                    raise RuntimeError(f'Runaway stimulus: {total} E spikes')
                retry = total < MODEL.minimum_exc_spikes
                network.run_rest()
                attempts += 1
                if retry:
                    intensity += 1
                    if intensity > 20:
                        raise RuntimeError('Retry intensity exceeded 20')
                else:
                    if diagnostic:
                        stimuli.append({'spikes': total, 'active_neurons': int(np.count_nonzero(spikes)), 'intensity': intensity})
                    accepted += 1
                    break
        input_counts, exc_counts, inh_counts = network.population_counts()
        total_spikes = int(input_counts.sum() + exc_counts.sum() + inh_counts.sum())
        wall = time.perf_counter() - started
        weights, theta = network.weights(), np.asarray(network.exc.get('theta'))
        events = {'input_spikes': int(input_counts.sum()), 'excitatory_spikes': int(exc_counts.sum()),
            'inhibitory_spikes': int(inh_counts.sum()), 'total_firing_count': total_spikes,
            'feedforward_pre_synapse_updates': int(input_counts @ mask.sum(axis=1)),
            'feedforward_post_synapse_updates': int(exc_counts @ mask.sum(axis=0))}
        executed = {'ff_pre_visits': int(np.sum(network.exc.get('pre_visits'))),
            'ff_post_visits': int(np.sum(network.exc.get('post_visits')))}
        network.validate(exc_counts * 0)
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(theta)) or weights.min() < 0:
            raise RuntimeError('Invalid NEST final weights/theta')
        if np.any(weights[~mask] != 0):
            raise RuntimeError('NEST sparsity guard failed')
        result = {'case': case, 'repetition': repetition, 'diagnostic_mode': diagnostic,
            'metric': 'single_network_latency', 'networks': 1, 'threads': nest.local_num_threads,
            'construction_wall_seconds': construction, 'wall_seconds': wall,
            'simulation_steps': attempts * MODEL.attempt_ticks, 'us_per_step': wall * 1e6 / (attempts * MODEL.attempt_ticks),
            'accepted_samples': accepted, 'attempts': attempts, 'total_spikes': total_spikes,
            'event_counters': events, 'executed_synaptic_visits': executed,
            'final_weights_sha256': hashlib.sha256(weights.astype(np.float32).tobytes()).hexdigest(),
            'final_theta_sha256': hashlib.sha256(theta.astype(np.float32).tobytes()).hexdigest(),
            'final_weight_bound_observation': {'maximum_weight_over_wmax': float(weights.max() / variant.weight_max),
                'weights_above_wmax': int(np.count_nonzero(weights > variant.weight_max))},
            'load_average_before': load_before, 'load_average_after': os.getloadavg()}
        if diagnostic:
            ranges = {name: {field: [float(np.min(pop.get(field))), float(np.max(pop.get(field)))]
                            for field in ['V', 'ge', 'gi', 'theta']}
                      for name, pop in [('E', network.exc), ('I', network.inh)]}
            result['diagnostic'] = {'state_ranges': ranges, 'weight_min': float(weights.min()),
                'weight_max': float(weights.max()), 'weight_mean': float(weights.mean()),
                'weight_change_rms': float(np.sqrt(np.mean((weights[mask] - before_weights[mask])**2))),
                'weight_change_max': float(np.max(np.abs(weights[mask] - before_weights[mask]))),
                'column_sum_min': float(weights.sum(axis=0).min()), 'column_sum_max': float(weights.sum(axis=0).max()),
                'theta_min_mv': float(theta.min()), 'theta_max_mv': float(theta.max()),
                'ff_connections': network.num_ff_connections,
                'ff_outdegree_range': [int(mask.sum(axis=1).min()), int(mask.sum(axis=1).max())],
                'ff_indegree_range': [int(mask.sum(axis=0).min()), int(mask.sum(axis=0).max())],
                'normalization_bound_observation': network.normalization_observation,
                'exc_spikes_per_neuron': exc_counts.tolist(), 'stimuli': stimuli, 'guards_passed': True}
        return result
    finally:
        network.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--module', type=Path, required=True)
    parser.add_argument('--nest-prefix', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'copilot/tmp/baseline_inputs_20260907_v2')
    parser.add_argument('--data-path', type=Path, default=ROOT / 'data/mnist')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260724)
    args = parser.parse_args()
    if min(args.threads, args.repetitions, args.samples, args.seed) <= 0:
        parser.error('Thread count, repetitions, samples and seed must be positive')
    args.module = args.module.resolve()
    args.output = args.output.resolve()
    candidates = list(args.nest_prefix.glob('lib*/python*/site-packages'))
    if len(candidates) != 1:
        raise RuntimeError('Cannot identify source-built PyNEST')
    sys.path.insert(0, str(candidates[0]))
    import nest
    if nest.num_processes != 1:
        raise RuntimeError('The shared-trace MNIST module requires one MPI process')
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = json.loads((args.inputs / 'manifest.json').read_text())
    for relative, expected in inputs['dataset_sha256'].items():
        if sha256_file(args.data_path / Path(relative).name) != expected:
            raise RuntimeError(f'Dataset hash mismatch: {relative}')
    source_paths = [Path(__file__), ROOT / 'cpu-benchmark/mnist_module/module.cpp',
        ROOT / 'cpu-benchmark/mnist_module/mechanics.h', ROOT / 'cpu-benchmark/mnist_module/CMakeLists.txt',
        ROOT / 'cpu-benchmark/shell.nix', ROOT / 'cpu-benchmark/test_nest_mnist.py',
        ROOT / 'genn-sweep/baseline.py', ROOT / 'genn-sweep/baseline_cases.py',
        ROOT / 'reimpl/zd3/constants.py', ROOT / 'reimpl/zd3/io.py', ROOT / 'reimpl/zd3/variants.py']
    manifest = {'schema': 'nest-mnist-latency-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'cwd': str(Path.cwd()), 'host': platform.uname()._asdict(),
        'nest_version': nest.__version__, 'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        'module_sha256': sha256_file(args.module), 'input_manifest_sha256': sha256_file(args.inputs / 'manifest.json'),
        'environment': {k: os.environ.get(k) for k in ['CXX', 'NIX_ENFORCE_NO_NATIVE', 'OPENBLAS_NUM_THREADS',
            'OMP_NUM_THREADS', 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'OMP_PROC_BIND', 'OMP_PLACES']},
        'cpu': baseline.command_output(['lscpu']), 'compiler': baseline.command_output(['g++', '--version']),
        'git_revision': baseline.command_output(['git', 'rev-parse', 'HEAD']),
        'git_status': baseline.command_output(['git', 'status', '--short']),
        'nest_revision': baseline.command_output(['git', '-C', str(ROOT / '3rdparty/nest-simulator'), 'rev-parse', 'HEAD']),
        'processes_before': baseline.command_output(['ps', '-eo', 'pid,comm,nlwp,pcpu', '--sort=-pcpu']),
        'precision': 'FP32 neuron/trace/plastic state; FP64 normalization sum and scaling; NEST recurrent event buffers FP64',
        'implementation': 'NEST threads and native spike routing; eager target-owned plastic arrays; two alternating shared input-trace buffers; one barrier-separated tick per update',
        'protocol': 'same frozen model definitions, weights/theta checkpoints and masks as GeNN; other runtime state reset; native NEST input RNG; one network, sequential repetitions; synchronization included; normalization runs inside target nodes',
        'weight_bound_policy': 'normalization and final cap overshoots are recorded, not fatal; no bound check in timing; configured STDP clipping is unchanged',
        'power': 'deferred; no energy or per-watt claim'}
    write(args.output / 'manifest.json', manifest)
    for p in source_paths:
        target = args.output / 'source' / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    summary = {}
    for case in args.cases:
        print(f'DIAGNOSTIC case={case} threads={args.threads}', flush=True)
        control = execute(nest, args, case, 0, True)
        write(args.output / case / 'diagnostic.json', control)
        results = []
        for repetition in range(1, args.repetitions + 1):
            result = execute(nest, args, case, repetition, False)
            for key in ['simulation_steps', 'total_spikes', 'event_counters', 'executed_synaptic_visits',
                        'final_weights_sha256', 'final_theta_sha256']:
                if result[key] != control[key]:
                    write(args.output / case / f'r{repetition}-mismatch.json', result)
                    raise RuntimeError(f'Diagnostic/timing mismatch: {case}/{key}')
            write(args.output / case / f'r{repetition}.json', result)
            results.append(result)
            print(f'TIMING case={case} threads={args.threads} r={repetition} '
                  f'us_per_step={result["us_per_step"]:.6f} steps={result["simulation_steps"]}', flush=True)
        samples = [r['us_per_step'] for r in results]
        summary[case] = {'metric': 'single_network_latency', 'networks': 1, 'threads': args.threads,
            'repetitions': args.repetitions, 'median_us_per_step': float(np.median(samples)),
            'range': [min(samples), max(samples)], 'iqr': float(np.subtract(*np.percentile(samples, [75, 25]))),
            'samples': samples, 'diagnostic': control}
        write(args.output / f'{case}.json', summary[case])
    write(args.output / 'summary.json', summary)
    print(f'COMPLETE output={args.output}', flush=True)


if __name__ == '__main__':
    main()
