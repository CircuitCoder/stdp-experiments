#!/usr/bin/env python3
"""Uninterrupted image training with immutable, separate-process evaluations."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'reimpl'), str(ROOT / 'genn-sweep')]
from baseline_cases import mnist_cases
from zd3.constants import MODEL, ModelConstants
from zd3.io import PortableCheckpoint, load_checkpoint, save_checkpoint, normalize_columns
from zd3.variants import NetworkVariant, connectivity_mask
from workloads.datasets import load_images, stratified_indices, score_activity
from workloads.images import RetryLimitExceeded, digest, make_network, present, summary
from workloads.provenance import manifest, write_json, sha256, command


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()


def training_parameters(args, n_input):
    base = mnist_cases()['mnist_3trace_dense']
    variant = replace(base,
        depression_rate=base.depression_rate * args.learning_rate_scale,
        potentiation_rate=base.potentiation_rate * args.learning_rate_scale)
    constants = replace(MODEL, n_input=n_input, theta_plus_mv=args.theta_plus_mv,
        stimulus_ms=args.stimulus_ms, depression_rate=variant.depression_rate,
        potentiation_rate=variant.potentiation_rate)
    return constants, variant


def inference_parameters(record):
    # Legacy theta10 records had the same 350-ms training/inference duration.
    constants = ModelConstants(**record.get('inference_model', record['model']))
    return constants, NetworkVariant(**record['variant'])


def evaluate(args):
    """Fresh process: no training arrays, runtime state or RNG are shared."""
    record = json.loads((args.training / 'manifest.json').read_text())
    config = record['configuration']
    training_constants = ModelConstants(**record['model'])
    constants, variant = inference_parameters(record)
    data = load_images(config['dataset'], Path(config['data_path']), config['top_classes'])
    assert data.manifest['file_sha256'] == record['dataset']['file_sha256']
    selection = np.load(args.training / 'selection.npz')
    marking = selection['marking_indices']
    testing = selection['full_test_indices' if args.full_test else 'probe_indices']
    checkpoint_hash = sha256(args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint, model=training_constants)
    if (checkpoint.manifest['training_manifest_sha256'] != sha256(args.training / 'manifest.json')
            or checkpoint.manifest['model'] != record['model']
            or checkpoint.manifest['variant'] != record['variant']):
        raise ValueError('Checkpoint does not belong to this training configuration')
    args.output.mkdir(parents=True, exist_ok=False)
    network_args = SimpleNamespace(backend=config['backend'], seed=config['seed'],
        output=args.output, integer_timestamps=True,
        build_root=(Path(config['build_root']) / 'evaluations' / args.output.name
                    if config.get('build_root') else None))
    write_json(args.output / 'manifest.json', manifest(args) | {
        'training_manifest_sha256': sha256(args.training / 'manifest.json'),
        'checkpoint_sha256': checkpoint_hash, 'accepted_samples': checkpoint.accepted_samples,
        'assignment_count': len(marking), 'test_count': len(testing),
        'assignment_index_sha256': digest(marking), 'test_index_sha256': digest(testing),
        'plasticity': False, 'theta_adaptation': False, 'inhibition': constants.inference_inhibition,
        'model': constants.as_dict(), 'variant': variant.as_dict(),
        'build_root': str(network_args.build_root) if network_args.build_root else None,
        'protocol': 'fresh inference; raw checkpoint weights; held-out training assignment followed by disjoint official test images'})
    started = time.perf_counter()
    network = None
    context = {'phase': 'build'}
    try:
        network = make_network(network_args, constants, variant, checkpoint,
                               connectivity_mask(variant, model=constants), False, 'inference')
        before_weights, before_theta = network.weights(), network.theta_mv()
        activities, attempts = [], 0
        for split, images, indices in [('assignment', data.train_images, marking),
                                       ('test', data.test_images, testing)]:
            activity = np.zeros((len(indices), constants.n_exc), dtype=np.uint16)
            for pos, index in enumerate(indices):
                context = {'phase': split, 'position': pos + 1, 'dataset_index': int(index)}
                counts, tries, _, _ = present(network, images[index], constants, training=False,
                                               max_intensity=config.get('max_intensity', 20))
                activity[pos] = counts
                attempts += tries
                if (pos + 1) % 100 == 0 or pos + 1 == len(indices):
                    network.validate_runtime(counts, 5000)
                    print(json.dumps({'phase': split, 'images': pos + 1, 'attempts': attempts}), flush=True)
            activities.append(activity)
        np.testing.assert_array_equal(network.weights(), before_weights)
        np.testing.assert_array_equal(network.theta_mv(), before_theta)
        assert sha256(args.checkpoint) == checkpoint_hash
        result = score_activity(activities[0], data.train_labels[marking], activities[1],
                                data.test_labels[testing], data.classes)
        result.update(status='complete', accepted_samples=checkpoint.accepted_samples,
            attempts=attempts, assignment_count=len(marking), test_count=len(testing),
            full_test=args.full_test, frozen_weights_and_theta_verified=True,
            checkpoint_unchanged_verified=True, process_wall_seconds=time.perf_counter() - started,
            diagnostics=summary(network, activities[1], variant))
        np.savez_compressed(args.output / 'activity.npz', assignment=activities[0], test=activities[1],
            assignment_labels=data.train_labels[marking], test_labels=data.test_labels[testing])
        write_json(args.output / 'result.json', result)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        failure = {'error': repr(exc), 'context': context}
        if isinstance(exc, RetryLimitExceeded):
            failure['retry_details'] = exc.details
        write_json(args.output / 'failure.json', failure)
        raise
    finally:
        if network is not None:
            network.close()


def train(args):
    data = load_images(args.dataset, args.data_path, args.top_classes)
    constants, variant = training_parameters(args, data.train_images.shape[1])
    inference_constants = replace(constants, stimulus_ms=args.inference_stimulus_ms)
    mask = connectivity_mask(variant, model=constants)
    marking = stratified_indices(data.train_labels, data.classes, args.assignment_per_class, args.seed)
    probe = stratified_indices(data.test_labels, data.classes, args.probe_per_class, args.seed + 1)
    # Official Fashion-MNIST and CIFAR-10 test pools both contain 1000 images/class.
    full_test = stratified_indices(data.test_labels, data.classes, 1000, args.seed + 1)
    pool = np.setdiff1d(np.arange(len(data.train_images)), marking)
    order = np.random.RandomState(args.seed + 2).permutation(pool)
    assert not np.intersect1d(order, marking).size
    total = args.epochs * len(order)
    if args.max_samples is not None:
        total = min(total, args.max_samples)
    if total <= 0:
        raise ValueError('training pool/budget must be positive')
    checkpoints = sorted({min(1000, total), total,
        *range(args.checkpoint_interval, total + 1, args.checkpoint_interval),
        *range(len(order), total + 1, len(order))})
    weights = np.random.RandomState(args.seed).uniform(.01, 1., mask.shape)
    normalize_columns(weights, constants.normalization_target)
    initial = PortableCheckpoint(weights, np.full(constants.n_exc, constants.theta_initial_mv), 0, {})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'checkpoints').mkdir()
    (args.output / 'evaluations').mkdir()
    record = manifest(args) | {'dataset': data.manifest, 'model': constants.as_dict(),
        'inference_model': inference_constants.as_dict(),
        'variant': variant.as_dict(), 'training_pool_count': len(order), 'target_accepted_samples': total,
        'checkpoint_samples': checkpoints, 'initial_weights_sha256': digest(weights),
        'initial_theta_sha256': digest(initial.theta_mv), 'mask_sha256': digest(mask),
        'training_order_sha256': digest(order), 'marking_index_sha256': digest(marking),
        'probe_index_sha256': digest(probe), 'full_test_index_sha256': digest(full_test),
        'intervention': {'theta_plus_mv': constants.theta_plus_mv,
            'theta_increment_scale': constants.theta_plus_mv / MODEL.theta_plus_mv,
            'learning_rate_scale': args.learning_rate_scale,
            'training_stimulus_ms': constants.stimulus_ms,
            'inference_stimulus_ms': inference_constants.stimulus_ms},
        'timestamp_representation': 'uint32 neuron spike ticks; subtract integers before converting elapsed ticks to FP32 ms',
        'protocol': 'three passes through one seeded shuffled training pool; assignment images excluded throughout; no labels used for STDP; separate-process frozen inference',
        'checkpoint_scope': 'immutable weights/theta for inference; not an exact training-resume snapshot',
        'evaluation_isolation': 'training runtime remains loaded and paused while a fresh child process evaluates each checkpoint',
        'timing_scope': 'attempt wall time includes normalization, transfers, retries, stimulus and blank rest; excludes build, diagnostics and evaluation; concurrent GPU jobs prevent performance comparisons',
        'stopping_rules': {'max_exc_spikes_per_attempt': 5000, 'max_intensity': args.max_intensity,
                           'nonfinite_state': 'abort', 'max_theta_mv': 1000},
        'assignment_count': len(marking), 'probe_count': len(probe), 'full_test_count': len(full_test)}
    import pygenn
    record['genn_version'] = pygenn.__version__
    if args.backend == 'cuda':
        record['gpu'] = command(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv'])
    write_json(args.output / 'manifest.json', record)
    np.savez(args.output / 'selection.npz', train_order=order, marking_indices=marking,
             probe_indices=probe, full_test_indices=full_test)
    activity = np.lib.format.open_memmap(args.output / 'training_activity.npy', mode='w+',
                                        dtype=np.uint16, shape=(total, constants.n_exc))
    per_image = np.lib.format.open_memmap(args.output / 'training_attempts.npy', mode='w+',
                                        dtype=np.uint16, shape=(total,))
    accepted = attempts = 0
    attempt_wall = 0.
    intensity_histogram = {}
    started = time.perf_counter()
    network = None
    result = {'status': 'running'}
    context = {'phase': 'build'}
    try:
        network = make_network(args, constants, variant, initial, mask, True, 'train')
        result['build_seconds'] = time.perf_counter() - started
        for sample in range(total):
            index = int(order[sample % len(order)])
            context = {'phase': 'training', 'sample_number': sample + 1, 'dataset_index': index}
            counts, tries, intensity, wall = present(network, data.train_images[index],
                                                     constants, training=True, max_intensity=args.max_intensity)
            activity[sample] = counts
            per_image[sample] = tries
            accepted = sample + 1
            attempts += tries
            attempt_wall += wall
            intensity_histogram[str(intensity)] = intensity_histogram.get(str(intensity), 0) + 1
            context['phase'] = 'training_diagnostics'
            if accepted % 100 == 0 or accepted in checkpoints:
                network.validate_runtime(counts, 5000)
                diagnostic = summary(network, activity[max(0, accepted - 1000):accepted], variant)
                progress = {'accepted_samples': accepted, 'attempts': attempts, 'retries': attempts - accepted,
                    'epoch_equivalents': accepted / len(order), 'attempt_wall_seconds': attempt_wall,
                    'simulated_seconds': attempts * constants.attempt_ticks * constants.dt_ms / 1000,
                    'process_wall_seconds': time.perf_counter() - started,
                    'last_1000_mean_attempts': float(per_image[max(0, accepted - 1000):accepted].mean()),
                    'max_intensity': max(float(x) for x in intensity_histogram),
                    'diagnostics': diagnostic}
                append(args.output / 'progress.jsonl', progress)
                print(json.dumps(progress), flush=True)
            if accepted in checkpoints:
                context['phase'] = 'checkpoint'
                activity.flush()
                per_image.flush()
                checkpoint_path = args.output / 'checkpoints' / f'sample_{accepted:06d}.npz'
                before_weights, before_theta = network.weights(), network.theta_mv()
                save_checkpoint(checkpoint_path, weights=before_weights, theta_mv=before_theta,
                    accepted_samples=accepted, model=constants,
                    manifest={'training_manifest_sha256': sha256(args.output / 'manifest.json'),
                              'variant': variant.as_dict(), 'scope': record['checkpoint_scope']})
                evaluation_path = args.output / 'evaluations' / f'sample_{accepted:06d}'
                call = [sys.executable, str(Path(__file__).resolve()), 'evaluate',
                    '--training', str(args.output), '--checkpoint', str(checkpoint_path),
                    '--output', str(evaluation_path)]
                if accepted == total and args.max_samples is None:
                    call.append('--full-test')
                append(args.output / 'evaluation_commands.jsonl', {'accepted_samples': accepted, 'command': call})
                saved_tick = network.model.timestep
                context['phase'] = 'evaluation'
                with evaluation_path.with_suffix('.log').open('x') as log:
                    subprocess.run(call, stdout=log, stderr=subprocess.STDOUT, check=True)
                assert network.model.timestep == saved_tick
                np.testing.assert_array_equal(network.weights(), before_weights)
                np.testing.assert_array_equal(network.theta_mv(), before_theta)
                evaluation = json.loads((evaluation_path / 'result.json').read_text())
                append(args.output / 'evaluations.jsonl', evaluation)
                print(json.dumps({'checkpoint': str(checkpoint_path),
                    'accuracy_percent': evaluation['accuracy_percent'],
                    'training_state_preserved': True}), flush=True)
        result.update(status='complete', diagnostics=summary(network, activity[:accepted], variant))
    except Exception as exc:
        result.update(status='failed', error=repr(exc), failure_context=context)
        if isinstance(exc, RetryLimitExceeded):
            result['failed_presentation'] = exc.details
        raise
    finally:
        activity.flush()
        per_image.flush()
        result.update(accepted_samples=accepted, attempts=attempts, retries=attempts - accepted,
            attempt_wall_seconds=attempt_wall, process_wall_seconds=time.perf_counter() - started,
            intensity_histogram=intensity_histogram)
        write_json(args.output / 'result.json', result)
        if network is not None:
            network.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    training = sub.add_parser('train')
    training.add_argument('--dataset', choices=['fashion-mnist', 'cifar10'], required=True)
    training.add_argument('--data-path', type=Path, required=True)
    training.add_argument('--top-classes', type=int)
    training.add_argument('--output', type=Path, required=True)
    training.add_argument('--backend', choices=['cuda', 'single_threaded_cpu'], default='cuda')
    training.add_argument('--seed', type=int, default=20260724)
    training.add_argument('--epochs', type=int, default=3)
    training.add_argument('--assignment-per-class', type=int, default=200)
    training.add_argument('--probe-per-class', type=int, default=200)
    training.add_argument('--checkpoint-interval', type=int, default=5000)
    training.add_argument('--max-samples', type=int, help='pilot cap; final evaluation then uses the small probe')
    training.add_argument('--max-intensity', type=int, default=20,
                          help='retry stopping limit; does not change initial intensity or increments')
    training.add_argument('--theta-plus-mv', type=float, default=.5,
                          help='adaptive threshold increment per training E spike (default: historical theta10 run)')
    training.add_argument('--learning-rate-scale', type=float, default=1.,
                          help='multiply both three-trace STDP coefficients')
    training.add_argument('--stimulus-ms', type=float, default=350., help='training presentation duration')
    training.add_argument('--inference-stimulus-ms', type=float, default=350.,
                          help='fixed assignment/test presentation duration, independent of training')
    training.add_argument('--build-root', type=Path,
                          help='optional separate compilation directory; checkpoints and activity remain under output')
    evaluation = sub.add_parser('evaluate')
    evaluation.add_argument('--training', type=Path, required=True)
    evaluation.add_argument('--checkpoint', type=Path, required=True)
    evaluation.add_argument('--output', type=Path, required=True)
    evaluation.add_argument('--full-test', action='store_true')
    args = parser.parse_args()
    psutil.cpu_count = lambda logical=True: 4
    if args.action == 'train':
        args.integer_timestamps = True
        for name in ('epochs', 'assignment_per_class', 'probe_per_class', 'checkpoint_interval'):
            if getattr(args, name) < 1:
                parser.error(f'{name} must be positive')
        for name in ('theta_plus_mv', 'learning_rate_scale', 'stimulus_ms', 'inference_stimulus_ms'):
            value = getattr(args, name)
            if not np.isfinite(value) or value <= 0:
                parser.error(f'{name} must be finite and positive')
        for name in ('stimulus_ms', 'inference_stimulus_ms'):
            ticks = getattr(args, name) / MODEL.dt_ms
            if ticks < 1 or not np.isclose(ticks, round(ticks), rtol=0, atol=1e-9):
                parser.error(f'{name} must be a positive whole number of ticks')
        # Poisson inputs use Bernoulli draws per tick. Do not allow rates above
        # one spike per tick, even on the brightest possible pixel.
        if args.max_intensity < MODEL.initial_intensity or args.max_intensity * (255 / 8) * MODEL.dt_ms / 1000 >= 1:
            parser.error('max-intensity must allow the initial attempt and keep every pixel probability below one')
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
