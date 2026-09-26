#!/usr/bin/env python3
"""Fresh image learning pilots and checkpoint continuation on GeNN or NEST."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'reimpl'), str(ROOT / 'brunel'),
               str(ROOT / 'genn-sweep'), str(ROOT / 'cpu-benchmark')]
from baseline_cases import mnist_cases
from backends.genn_backend import GeNNNetwork
from zd3.constants import MODEL
from zd3.io import PortableCheckpoint, load_checkpoint, save_checkpoint, normalize_columns
from zd3.variants import connectivity_mask
from workloads.datasets import load_images, stratified_indices, score_activity
from workloads.provenance import manifest, write_json, sha256, command


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class NestImageNetwork:
    def __init__(self, args, constants, variant, checkpoint, mask, plasticity):
        if args.nest_prefix is None or args.module is None:
            raise ValueError('NEST requires --nest-prefix and --module')
        for path in args.nest_prefix.glob('lib*/python*/site-packages'):
            sys.path.insert(0, str(path))
        import nest
        from run_nest_mnist import Network
        self.network = Network(nest, args, variant, checkpoint, mask, constants=constants,
            plasticity=plasticity, inhibition=constants.train_inhibition if plasticity else constants.inference_inhibition)
        self.version = nest.__version__

    def normalize(self):
        self.network.normalize(False)

    def set_image(self, pixels, intensity):
        self.network.set_image(pixels, intensity)

    def run_stimulus(self):
        return self.network.run_stimulus()

    def run_rest(self):
        self.network.run_rest()

    def weights(self):
        return self.network.weights()

    def theta_mv(self):
        return np.asarray(self.network.exc.get('theta'), dtype=np.float64)

    def close(self):
        self.network.close()

    def validate_runtime(self, counts, limit):
        self.network.validate(counts)

    def event_counters(self):
        counts = self.network.population_counts()
        return {f'{name}_spikes': int(value.sum()) for name, value in
                zip(('input', 'excitatory', 'inhibitory'), counts)}

    def runtime_diagnostics(self):
        return {f'{name}_{field}_{op}': float(getattr(np, op)(pop.get(field)))
                for name, pop in [('e', self.network.exc), ('i', self.network.inh)]
                for field in ('V', 'ge', 'gi', 'theta') for op in ('min', 'max')}


def make_network(args, constants, variant, checkpoint, mask, plasticity, tag):
    if args.backend == 'nest':
        return NestImageNetwork(args, constants, variant, checkpoint, mask, plasticity)
    return GeNNNetwork(weights=checkpoint.weights, theta_mv=checkpoint.theta_mv,
        plasticity=plasticity,
        inhibition=constants.train_inhibition if plasticity else constants.inference_inhibition,
        seed=args.seed, backend=args.backend,
        build_path=(getattr(args, 'build_root', None) or args.output) / f'build_{tag}',
        variant=variant, structural_mask=mask, precision='float',
        parallelism='postsynaptic' if variant.topology == 'dense' else 'presynaptic',
        num_threads_per_spike=1 if variant.topology == 'dense' else 32,
        timing_enabled=False, reuse_build=None, constants=constants,
        integer_timestamps=getattr(args, 'integer_timestamps', False))


class RetryLimitExceeded(RuntimeError):
    """Retain the failed attempt without changing presentation/retry behavior."""

    def __init__(self, max_intensity, attempts, counts, wall_seconds):
        super().__init__(f'Retry intensity exceeded {max_intensity}')
        self.details = {'max_intensity': max_intensity, 'attempts': attempts,
                        'last_exc_spikes': int(counts.sum()),
                        'last_spike_counts': counts.tolist(),
                        'attempt_wall_seconds': wall_seconds}


def present(network, pixels, constants, *, training, max_intensity=20):
    intensity = constants.initial_intensity
    wall = 0.0
    attempts = 0
    while True:
        started = time.perf_counter()
        if training:
            if isinstance(network, GeNNNetwork):
                network.normalize(validate=False, check_weight_bound=False)
            else:
                network.normalize()
        network.set_image(pixels, intensity)
        counts = network.run_stimulus()
        network.run_rest()
        wall += time.perf_counter() - started
        attempts += 1
        if int(counts.sum()) >= 5000:
            raise RuntimeError(f'Runaway stimulus: {int(counts.sum())} excitatory spikes')
        if int(counts.sum()) >= constants.minimum_exc_spikes:
            return counts, attempts, intensity, wall
        intensity += constants.intensity_increment
        if intensity > max_intensity:
            raise RetryLimitExceeded(max_intensity, attempts, counts, wall)


def summary(network, activity, variant):
    weights, theta = network.weights(), network.theta_mv()
    state = network.runtime_diagnostics()
    for values in (weights, theta, np.asarray(list(state.values()))):
        if not np.all(np.isfinite(values)):
            raise RuntimeError('Non-finite final state')
    if weights.min() < 0 or theta.min() < 0 or theta.max() > 1000:
        raise RuntimeError('Invalid weight or threshold range')
    if not (-150 <= min(v for k, v in state.items() if 'v_min' in k.lower())):
        raise RuntimeError('Implausible membrane voltage')
    sums = weights.sum(axis=0)
    neurons = activity.sum(axis=0)
    return {'counters': network.event_counters(), 'state': state,
        'weights': {'min': float(weights.min()), 'max': float(weights.max()),
            'mean': float(weights.mean()), 'zero_fraction': float(np.mean(weights == 0)),
            'at_or_above_cap_fraction': float(np.mean(weights >= variant.weight_max)),
            'column_sum_min': float(sums.min()), 'column_sum_max': float(sums.max()),
            'sha256': digest(weights.astype(np.float32))},
        'theta': {'min': float(theta.min()), 'max': float(theta.max()), 'mean': float(theta.mean()),
            'sha256': digest(theta.astype(np.float32))},
        'active_neurons': int(np.count_nonzero(neurons)),
        'largest_neuron_spike_fraction': float(neurons.max() / max(1, neurons.sum()))}


def run(args):
    data = load_images(args.dataset, args.data_path, args.top_classes)
    constants = replace(MODEL, n_input=data.train_images.shape[1])
    tag = 'dense' if args.topology == 'dense' else f'{args.topology}_{int(args.density * 1000):04d}'
    variant = mnist_cases()[f'mnist_{args.rule}trace_{tag}']
    mask = connectivity_mask(variant, model=constants)
    marking = stratified_indices(data.train_labels, data.classes, args.assignment_per_class, args.seed)
    testing = stratified_indices(data.test_labels, data.classes, args.test_per_class, args.seed + 1)
    available = np.setdiff1d(np.arange(len(data.train_images)), marking)
    order = np.random.RandomState(args.seed + 2).permutation(available)
    contract = {'dataset': data.manifest, 'model': constants.as_dict(), 'variant': variant.as_dict(),
                'marking_index_sha256': digest(marking), 'training_order_sha256': digest(order),
                'mask_sha256': digest(mask)}
    if args.checkpoint:
        checkpoint = load_checkpoint(args.checkpoint, model=constants)
        saved = checkpoint.manifest.get('image_contract')
        # Dataset paths can differ across hosts; content hashes identify the data.
        expected = dict(contract, dataset={k: v for k, v in data.manifest.items() if k != 'root'})
        # JSON normalizes integer keys and tuples.
        import json
        if saved != json.loads(json.dumps(expected)):
            raise ValueError('Checkpoint dataset, selection, dimensions or learning rule mismatch')
    else:
        weights = np.random.RandomState(args.seed).uniform(.01, 1.0, mask.shape) * mask
        normalize_columns(weights)
        checkpoint = PortableCheckpoint(weights, np.full(400, constants.theta_initial_mv), 0, {})
    args.output.mkdir(parents=True, exist_ok=False)
    record = manifest(args) | contract | {
        'checkpoint_sha256': sha256(args.checkpoint) if args.checkpoint else None,
        'initial_weights_sha256': digest(checkpoint.weights),
        'initial_theta_sha256': digest(checkpoint.theta_mv),
        'protocol': 'STDP uses no labels; assignment from held-out training images; scoring on separate test images; inference uses fresh runtime and frozen weights/theta',
        'continuation': 'weights/theta only; other runtime state reset' if args.checkpoint else 'fresh seeded random initialization',
        'timing': 'normalization, transfers, stimulus, rest and retries; compilation, loading, diagnostics and evaluation excluded',
        'stopping_rules': {'max_exc_spikes_per_attempt': 5000, 'max_intensity': 20,
                           'nonfinite_state': 'abort', 'max_theta_mv': 1000},
        'assignment_count': len(marking), 'test_count': len(testing),
        'training_pool_count': len(order)}
    if args.backend == 'nest':
        record['module_sha256'] = sha256(args.module)
    else:
        import pygenn
        record['genn_version'] = pygenn.__version__
    if args.backend == 'cuda':
        record['gpu'] = command(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv'])
    write_json(args.output / 'manifest.json', record)
    np.savez(args.output / 'selection.npz', train_order=order, marking_indices=marking, test_indices=testing)
    network = None
    result = {'status': 'running'}
    try:
        start = time.perf_counter()
        network = make_network(args, constants, variant, checkpoint, mask, True, 'train')
        result['build_seconds'] = time.perf_counter() - start
        activity = np.zeros((args.train_samples, 400), dtype=np.int64)
        attempts, wall, intensities = 0, 0.0, []
        for sample in range(args.train_samples):
            index = order[(checkpoint.accepted_samples + sample) % len(order)]
            counts, tries, intensity, duration = present(network, data.train_images[index], constants, training=True)
            activity[sample] = counts
            attempts += tries
            wall += duration
            intensities.append(intensity)
            if sample == 0 or (sample + 1) % 50 == 0 or sample + 1 == args.train_samples:
                network.validate_runtime(counts, 5000)
                print(f'accepted={sample + 1} attempts={attempts} E_spikes={counts.sum()} intensity={intensity} wall={wall:.3f}', flush=True)
        result.update(summary(network, activity, variant))
        result.update({'accepted_samples': args.train_samples, 'attempts': attempts,
            'retries': attempts - args.train_samples, 'executed_ticks': attempts * constants.attempt_ticks,
            'wall_seconds': wall, 'us_per_step': wall * 1e6 / (attempts * constants.attempt_ticks),
            'max_intensity': max(intensities), 'mean_exc_spikes_per_image': float(activity.sum(axis=1).mean())})
        np.savez(args.output / 'training_activity.npz', counts=activity, intensities=intensities)
        trained = PortableCheckpoint(network.weights(), network.theta_mv(),
            checkpoint.accepted_samples + args.train_samples, {})
        saved_contract = dict(contract, dataset={k: v for k, v in data.manifest.items() if k != 'root'})
        save_checkpoint(args.output / 'checkpoint.npz', weights=trained.weights, theta_mv=trained.theta_mv,
            accepted_samples=trained.accepted_samples, model=constants,
            manifest={'image_contract': saved_contract, 'variant': variant.as_dict()})
        network.close()
        network = None
        if args.evaluate:
            network = make_network(args, constants, variant, trained, mask, False, 'inference')
            before_weights, before_theta = network.weights(), network.theta_mv()
            activities = []
            inference_attempts = 0
            for images, indices in [(data.train_images, marking), (data.test_images, testing)]:
                counts_list = []
                for index in indices:
                    counts, tries, _, _ = present(network, images[index], constants, training=False)
                    network.validate_runtime(counts, 5000)
                    counts_list.append(counts)
                    inference_attempts += tries
                activities.append(np.asarray(counts_list))
            np.testing.assert_array_equal(network.weights(), before_weights)
            np.testing.assert_array_equal(network.theta_mv(), before_theta)
            evaluation = score_activity(activities[0], data.train_labels[marking], activities[1],
                                        data.test_labels[testing], data.classes)
            evaluation['attempts'] = inference_attempts
            evaluation['frozen_weights_and_theta_verified'] = True
            result['evaluation'] = evaluation
            np.savez(args.output / 'evaluation_activity.npz', assignment=activities[0], test=activities[1],
                      assignment_labels=data.train_labels[marking], test_labels=data.test_labels[testing])
        result['status'] = 'complete'
        print(result, flush=True)
    except Exception as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(args.output / 'result.json', result)
        if network is not None:
            network.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['fashion-mnist', 'cifar10'], required=True)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--top-classes', type=int, choices=[2])
    parser.add_argument('--rule', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--topology', choices=['dense', 'bernoulli', 'fixed-fanout'], default='dense')
    parser.add_argument('--density', type=float, choices=[.5, .25, .125], default=.125)
    parser.add_argument('--backend', choices=['cuda', 'single_threaded_cpu', 'nest'], default='cuda')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--train-samples', type=int, default=1000)
    parser.add_argument('--assignment-per-class', type=int, default=20)
    parser.add_argument('--test-per-class', type=int, default=20)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--seed', type=int, default=20260724)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--nest-prefix', type=Path)
    parser.add_argument('--module', type=Path)
    args = parser.parse_args()
    if min(args.seed, args.train_samples, args.assignment_per_class, args.test_per_class, args.threads) <= 0:
        parser.error('seeds, sample counts and threads must be positive')
    if args.module:
        args.module = args.module.resolve()
    run(args)


if __name__ == '__main__':
    main()
