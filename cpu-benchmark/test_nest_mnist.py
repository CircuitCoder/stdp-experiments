"""Independent recurrences for the NEST MNIST rules and timing boundaries."""
import os
from math import exp
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.fixture
def nest_api():
    prefix = os.environ.get('CPU_BENCH_NEST_PREFIX')
    module = os.environ.get('CPU_BENCH_MNIST_MODULE')
    if not prefix or not module:
        pytest.skip('Set source-built NEST prefix and MNIST extension')
    for path in Path(prefix).glob('lib*/python*/site-packages'):
        sys.path.insert(0, str(path))
    import nest
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': 0.5, 'local_num_threads': 4, 'rng_seed': 20260724})
    nest.Install(str(Path(module).resolve().with_suffix('')))
    nest.set_verbosity('M_ERROR')
    return nest


@pytest.mark.parametrize('rule', [1, 2, 3])
@pytest.mark.parametrize('threads', [1, 4])
def test_forced_events_match_pre_before_post_recurrence(nest_api, rule, threads):
    nest = nest_api
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': 0.5, 'local_num_threads': threads, 'rng_seed': 20260724})
    nest.Install(str(Path(os.environ['CPU_BENCH_MNIST_MODULE']).resolve().with_suffix('')))
    pre_steps, post_steps = {2, 6, 8, 9, 12}, {4, 6, 9, 10, 12}
    pre = nest.Create('cpu_mnist_neuron', 1, {'kind': 0, 'rule': rule, 'input_index': 0,
        'force_only': True, 'forced_steps': sorted(pre_steps)})
    post = nest.Create('cpu_mnist_neuron', 1, {'kind': 1, 'rule': rule,
        'force_only': True, 'forced_steps': sorted(post_steps), 'potentiation': 0.0005,
        'pre_indices': [0], 'ff_weights': [0.5]})
    nest.Connect(pre, post, syn_spec={'synapse_model': 'static_synapse', 'receptor_type': 1, 'delay': 0.5, 'weight': 1.0})
    weight, x, y = 0.5, 0.0, 0.0
    last_pre = last_post = None
    for step in range(1, 16):
        source_step = step - 1
        delivered = 0.0
        if source_step > 0:
            x *= exp(-0.5 / 20); y *= exp(-0.5 / 20)
            if source_step in pre_steps:
                delivered = weight
                x += 1
                last_pre = source_step
                if rule == 2:
                    weight = np.clip(weight - 0.0001 * y * weight**0.2, 0, 1)
                elif rule == 3:
                    old_y = 0.0 if last_post is None else exp(-(source_step - last_post) * 0.5 / 20)
                    weight = np.clip(weight - 0.0001 * old_y, 0, 1)
            if source_step in post_steps:
                if rule in (1, 2):
                    weight = np.clip(weight + 0.0005 * (x - 0.4) * (1 - weight)**0.2, 0, 1)
                else:
                    pre_trace = 0.0 if last_pre is None else exp(-(source_step - last_pre) * 0.5 / 20)
                    post2 = 0.0 if last_post is None else exp(-(source_step - last_post) * 0.5 / 40)
                    weight = np.clip(weight + 0.0005 * pre_trace * post2, 0, 1)
                y += 1
                last_post = source_step
        nest.Simulate(0.5)
        assert post.get('ff_weights')[0] == pytest.approx(weight, abs=2e-6)
        assert post.get('last_input_current') == pytest.approx(delivered, abs=2e-6)
    assert post.get('pre_visits') == len(pre_steps)
    assert post.get('post_visits') == len(post_steps)


def test_midpoint_and_refractory_freeze(nest_api):
    nest = nest_api
    neurons = nest.Create('cpu_mnist_neuron', 2, {'kind': 1, 'force_only': True,
        'V': -60.0, 'ge': 2.0, 'gi': 1.0, 'theta': 20.0})
    neurons[1].set({'refractory': 2})
    nest.Simulate(0.5)
    ge_mid, gi_mid = 2 * exp(-0.25), exp(-0.125)
    g = 1 + ge_mid + gi_mid
    v_inf = (-65 - 100 * gi_mid) / g
    expected = v_inf + (-60 - v_inf) * exp(-0.5 * g / 100)
    assert neurons[0].get('V') == pytest.approx(expected, abs=8e-6)
    assert neurons[0].get('ge') == pytest.approx(2 * exp(-0.5), abs=2e-6)
    assert neurons[1].get('V') == -60.0
    assert neurons[1].get('ge') == 2.0
    assert neurons[1].get('gi') == 1.0
    assert neurons[1].get('theta') == 20.0
    assert neurons[1].get('refractory') == 1


def test_recurrent_arrival_is_next_tick(nest_api):
    nest = nest_api
    pre = nest.Create('cpu_mnist_neuron', 1, {'kind': 2, 'force_only': True, 'forced_steps': [2]})
    post = nest.Create('cpu_mnist_neuron', 1, {'kind': 1,
        'force_only': True, 'refractory': 10})
    nest.Connect(pre, post, syn_spec={'synapse_model': 'static_synapse_hpc', 'delay': 0.5, 'weight': -4.0})
    nest.Simulate(1.0)
    assert post.get('gi') == 0.0
    nest.Simulate(0.5)
    assert post.get('gi') == 4.0


@pytest.mark.parametrize('rule', [1, 2, 3])
def test_normalization_between_emission_and_stdp_and_silent_input(nest_api, rule):
    nest = nest_api
    pre = nest.Create('cpu_mnist_neuron', 2, {'kind': 0, 'rule': rule, 'force_only': True})
    pre[0].set({'input_index': 0, 'forced_steps': [2]})
    pre[1].set({'input_index': 1})
    post = nest.Create('cpu_mnist_neuron', 1, {'kind': 1, 'rule': rule,
        'force_only': True, 'forced_steps': [2], 'potentiation': 0.0005,
        'pre_indices': [0, 1], 'ff_weights': [0.5, 0.5], 'weight_max': 100.0})
    for i in range(2):
        nest.Connect(pre[i], post, syn_spec={'synapse_model': 'static_synapse', 'receptor_type': i + 1, 'delay': 0.5, 'weight': 1.0})
    nest.Simulate(1.0)
    np.testing.assert_array_equal(post.get('ff_weights'), [0.5, 0.5])
    post.set({'normalize_now': True})
    np.testing.assert_array_equal(post.get('ff_weights'), [39.0, 39.0])
    nest.Simulate(0.5)
    assert post.get('last_input_current') == 39.0
    if rule in (1, 2):
        expected = [39 + 0.0005 * v * 61**0.2 for v in (0.6, -0.4)]
    else:
        expected = [39.0, 39.0]  # No preceding post for the slow trace.
    np.testing.assert_allclose(post.get('ff_weights'), expected, atol=3e-6, rtol=0)
    assert post.get('post_visits') == 2  # Even the never-firing input is updated.
    assert post.get('normalization_count') == 1


def test_shared_input_trace_cross_thread_broadcast(nest_api):
    nest = nest_api
    pre = nest.Create('cpu_mnist_neuron', 1, {'kind': 0, 'rule': 1, 'input_index': 0,
        'force_only': True, 'forced_steps': [1, 2, 3, 5]})
    post = nest.Create('cpu_mnist_neuron', 12, {'kind': 1, 'rule': 1,
        'force_only': True, 'forced_steps': [2, 5, 8], 'pre_indices': [0], 'ff_weights': [0.5]})
    nest.Connect(pre, post, syn_spec={'synapse_model': 'static_synapse', 'receptor_type': 1, 'delay': 0.5, 'weight': 1.0})
    nest.Simulate(5.0)
    weights = np.asarray(post.get('ff_weights')).reshape(-1)
    assert weights[0] != 0.5
    np.testing.assert_array_equal(weights, np.full(12, weights[0]))
    np.testing.assert_array_equal(post.get('pre_visits'), np.full(12, 4))
    np.testing.assert_array_equal(post.get('post_visits'), np.full(12, 3))


def test_normalization_may_exceed_stdp_weight_cap(nest_api):
    nest = nest_api
    post = nest.Create('cpu_mnist_neuron', 1, {'kind': 1, 'rule': 1,
        'force_only': True, 'pre_indices': [0, 1], 'ff_weights': [0.25, 0.75], 'weight_max': 1.0})
    post.set({'normalize_now': True})
    # A two-input column cannot sum to 78 while respecting a cap of one.
    # Benchmark normalization still enforces the sum; STDP clipping is separate.
    np.testing.assert_array_equal(post.get('ff_weights'), [19.5, 58.5])
    assert sum(post.get('ff_weights')) == 78.0
