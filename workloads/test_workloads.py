from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'reimpl'), str(ROOT / 'brunel')]
from workloads.datasets import most_frequent, read_cifar, score_activity, stratified_indices
from workloads.morrison import make_morrison
from ports.common import alpha_propagator
from zd3.constants import MODEL
from zd3.io import save_checkpoint, load_checkpoint
from zd3.variants import get_variant, connectivity_mask


def test_frequency_selection_filters_original_labels_and_breaks_ties():
    assert most_frequent(np.repeat(np.arange(10), 4), 2) == (0, 1)
    assert most_frequent(np.array([8, 8, 8, 3, 3, 1]), 2) == (3, 8)


def test_native_rgb_channel_layout(tmp_path):
    record = np.r_[np.uint8(7), np.repeat(np.array([11, 22, 33], dtype=np.uint8), 1024)]
    path = tmp_path / 'batch.bin'
    record.tofile(path)
    images, labels = read_cifar(path)
    assert images.shape == (1, 3072)
    np.testing.assert_array_equal(images[0].reshape(3, 32, 32)[:, 7, 9], [11, 22, 33])
    np.testing.assert_array_equal(labels, [7])


def test_stratified_holdout_and_retained_class_readout():
    labels = np.repeat([3, 8], 20)
    indices = stratified_indices(labels, (3, 8), 4, 17)
    assert len(np.unique(indices)) == 8
    assert np.sum(labels[indices] == 3) == 4
    scores = score_activity(np.array([[2, 0], [0, 2]]), np.array([3, 8]),
                            np.array([[0, 0], [0, 5]]), np.array([3, 8]), (3, 8))
    assert scores['accuracy_percent'] == 100
    assert scores['silent_test_samples'] == 1


def test_rgb_checkpoint_roundtrip_and_dimension_rejection(tmp_path):
    model = replace(MODEL, n_input=3072)
    path = tmp_path / 'rgb.npz'
    weights = np.full((3072, 400), 78 / 3072)
    save_checkpoint(path, weights=weights, theta_mv=np.full(400, 20),
                    accepted_samples=7, manifest={}, model=model)
    saved = load_checkpoint(path, model=model)
    np.testing.assert_array_equal(saved.weights, weights)
    assert saved.manifest['model']['n_input'] == 3072
    with pytest.raises(ValueError, match='feedforward shape'):
        load_checkpoint(path)
    variant = replace(get_variant('one-trace-dense'), topology='fixed-fanout', connection_rate=.125)
    mask = connectivity_mask(variant, model=model)
    np.testing.assert_array_equal(mask.sum(axis=1), np.full(3072, 50))


def test_morrison_size_delay_and_original_parameters():
    model = make_morrison()
    assert model.ne + model.ni == 112500
    assert model.plastic_synapses == 810000000
    assert model.recurrent_synapses == 1265625000
    assert model.delay_ms == 0
    assert model.recurrent_delivery_scale == 1
    assert model.external_rate_hz == 20880
    assert model.rule.depression_ratio == .1057
    assert model.tau_minus_ms == model.tau_plus_ms == 20
    assert model.as_dict()['tau_syn_ms'] == .33


def test_morrison_alpha_step_against_independent_matrix_exponential():
    # State order: voltage, alpha current, alpha derivative.
    a = np.array([[-1 / 10, 1 / 250, 0], [0, -1 / .33, 1], [0, 0, -1 / .33]])
    expected = expm(a * .1) @ np.array([5., 7., 11.])
    p = alpha_propagator(.33)
    actual = [p['p33'] * 5 + p['p32'] * 7 + p['p31'] * 11,
              p['p22'] * 7 + p['p21'] * 11, p['p11'] * 11]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_generated_morrison_zero_delay_and_simultaneous_pair(tmp_path):
    pytest.importorskip('pygenn')
    from ports.genn_port import GeNNBrunel
    model = replace(make_morrison(), ne=2, ni=2, ce=1, ci=1, external_rate_scale=0)
    network = GeNNBrunel(spec=model, seed=17, state_seed=17,
        backend='single_threaded_cpu', build_path=tmp_path / 'build',
        recording_steps=0, precision='float', stdp_timing='arrival',
        stdp_tie_order='nest_causal_boundary', timing_enabled=False,
        record_spikes=False, collect_connectivity_stats=False)
    try:
        network.exc.vars['V'].view[:] = 21
        network.inh.vars['V'].view[:] = 0
        network.step(1)
        np.testing.assert_array_equal(network.exc.vars['dIex'].view, [0, 0])
        network.step(1)
        # Both cells spiked simultaneously. Zero-lag pairs are excluded, and
        # each other's current arrives on the following 0.1 ms tick.
        np.testing.assert_allclose(network.ee.vars['g'].values, 45.61, rtol=1e-6)
        np.testing.assert_allclose(network.exc.vars['dIex'].view,
                                   np.e / .33 * 45.61, rtol=1e-6)
        from workloads.morrison import sample_weights
        sample = sample_weights(network, model, count=2)
        assert sample['mean'] == pytest.approx(45.61, abs=1e-5)
    finally:
        network.close()


@pytest.mark.parametrize('rule', [1, 2, 3])
def test_rgb_genn_inference_is_frozen_even_above_learning_cap(tmp_path, rule):
    pytest.importorskip('pygenn')
    sys.path.insert(0, str(ROOT / 'genn-sweep'))
    from baseline_cases import mnist_cases
    from backends.genn_backend import GeNNNetwork
    constants = replace(MODEL, n_input=3072, n_exc=2, n_inh=2)
    weights = np.full((3072, 2), 1.5)
    network = GeNNNetwork(weights=weights, theta_mv=np.full(2, 20.), plasticity=False,
        inhibition=17., seed=17, backend='single_threaded_cpu', build_path=tmp_path / 'build',
        variant=mnist_cases()[f'mnist_{rule}trace_dense'], structural_mask=np.ones_like(weights, dtype=bool),
        precision='float', parallelism='postsynaptic', num_threads_per_spike=1,
        timing_enabled=False, reuse_build=None, constants=constants)
    try:
        pixels = np.zeros(3072)
        pixels[-1] = 255
        network.set_image(pixels, 2000 * 8 / 255)
        network.exc.vars['V'].view[:] = 21
        network._step(25)
        assert network.event_counters()['input_spikes'] == 25
        assert network.event_counters()['excitatory_spikes'] > 0
        np.testing.assert_array_equal(network.weights(), weights)
        np.testing.assert_array_equal(network.theta_mv(), [20., 20.])
    finally:
        network.close()
