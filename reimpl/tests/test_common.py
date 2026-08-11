from math import exp
from pathlib import Path

import numpy as np
import pytest

from zd3.constants import MODEL
from zd3.dynamics import TraceState, event_time_weight_updates, midpoint_step
from zd3.evaluation import simple_demo_accuracy
from zd3.io import load_checkpoint, normalize_columns, save_checkpoint
from zd3.variants import (
    connectivity_mask,
    get_variant,
    prepare_initial_weights,
    validate_normalized_weight_bound,
)


def test_model_tick_counts_are_exact() -> None:
    assert MODEL.stimulus_ticks == 700
    assert MODEL.rest_ticks == 300
    assert MODEL.attempt_ticks == 1000


def test_midpoint_step_matches_independent_calculation() -> None:
    result = midpoint_step(
        v_mv=-61.0,
        ge=3.0,
        gi=2.0,
        theta_mv=20.0,
        tau_m_ms=MODEL.exc_tau_m_ms,
        v_rest_mv=MODEL.exc_v_rest_mv,
        e_exc_mv=MODEL.exc_e_exc_mv,
        e_inh_mv=MODEL.exc_e_inh_mv,
    )
    ge_mid = 3.0 * exp(-0.25 / 1.0)
    gi_mid = 2.0 * exp(-0.25 / 2.0)
    total = 1.0 + ge_mid + gi_mid
    v_inf = (-65.0 + ge_mid * 0.0 + gi_mid * -100.0) / total
    expected_v = v_inf + (-61.0 - v_inf) * exp(-0.5 * total / 100.0)
    assert result.v_mv == pytest.approx(expected_v, abs=1e-12)
    assert result.ge == pytest.approx(3.0 * exp(-0.5), abs=1e-12)
    assert result.gi == pytest.approx(2.0 * exp(-0.25), abs=1e-12)


def test_refractory_freezes_equation_state() -> None:
    result = midpoint_step(
        v_mv=-65.0,
        ge=4.0,
        gi=3.0,
        theta_mv=22.0,
        tau_m_ms=MODEL.exc_tau_m_ms,
        v_rest_mv=MODEL.exc_v_rest_mv,
        e_exc_mv=MODEL.exc_e_exc_mv,
        e_inh_mv=MODEL.exc_e_inh_mv,
        refractory=True,
    )
    assert result.v_mv == -65.0
    assert result.ge == 4.0
    assert result.gi == 3.0
    assert result.theta_mv == 22.0


def test_three_trace_rule_samples_post2_before_reset() -> None:
    state = TraceState(weight=0.5)
    state.on_pre(0.0)
    state.on_post(10.0)
    assert state.weight == pytest.approx(0.5)
    state.on_pre(20.0)
    depressed = 0.5 - MODEL.depression_rate * exp(-10.0 / MODEL.post1_tau_ms)
    assert state.weight == pytest.approx(depressed)
    state.on_post(30.0)
    expected = depressed + (
        MODEL.potentiation_rate
        * exp(-10.0 / MODEL.pre_tau_ms)
        * exp(-20.0 / MODEL.post2_tau_ms)
    )
    assert state.weight == pytest.approx(expected)


def test_same_tick_pre_before_post_uses_updated_pre_trace() -> None:
    state = TraceState(weight=0.5, post2=0.4, time_ms=1.0)
    state.on_pre(1.0)
    state.on_post(1.0)
    assert state.weight == pytest.approx(0.504)


def test_event_time_rule_matches_same_tick_pre_before_post() -> None:
    weight = event_time_weight_updates(
        weight=0.5,
        pre_time_ms=10.0,
        post_time_ms=10.0,
        previous_post_time_ms=10.0 - MODEL.post2_tau_ms * -np.log(0.4),
        apply_pre=True,
        apply_post=True,
    )
    expected_post1 = 0.4 ** (MODEL.post2_tau_ms / MODEL.post1_tau_ms)
    assert weight == pytest.approx(
        0.5 - MODEL.depression_rate * expected_post1
        + MODEL.potentiation_rate * 0.4
    )


def test_event_time_rule_first_post_has_no_potentiation() -> None:
    weight = event_time_weight_updates(
        weight=0.5,
        pre_time_ms=5.0,
        post_time_ms=10.0,
        apply_post=True,
    )
    assert weight == 0.5


def test_normalization_sets_every_column_sum() -> None:
    weights = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    normalize_columns(weights)
    np.testing.assert_allclose(weights.sum(axis=0), 78.0, rtol=1e-6)


def test_checkpoint_round_trip_and_fail_if_exists(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.npz"
    weights = np.full((MODEL.n_input, MODEL.n_exc), 0.1, dtype=np.float32)
    theta = np.full(MODEL.n_exc, 20.0)
    save_checkpoint(
        path,
        weights=weights,
        theta_mv=theta,
        accepted_samples=123,
        manifest={"backend": "test"},
    )
    checkpoint = load_checkpoint(path)
    np.testing.assert_array_equal(checkpoint.weights, weights)
    np.testing.assert_array_equal(checkpoint.theta_mv, theta)
    assert checkpoint.accepted_samples == 123
    assert checkpoint.manifest["backend"] == "test"
    with pytest.raises(FileExistsError):
        save_checkpoint(
            path,
            weights=weights,
            theta_mv=theta,
            accepted_samples=123,
            manifest={"backend": "test"},
        )


def test_simple_demo_evaluation() -> None:
    labels = np.tile(np.arange(10, dtype=np.uint8), 2)
    activity = np.zeros((20, 20), dtype=np.float64)
    for row, label in enumerate(labels):
        activity[row, 2 * label : 2 * label + 2] = 5.0
    score = simple_demo_accuracy(activity, labels)
    assert score["accuracy_percent"] == 100.0
    assert score["assigned_neurons"] == 20


def test_reference_bernoulli_mask_and_initialization() -> None:
    variant = get_variant("one-trace-bernoulli-0125")
    mask = connectivity_mask(variant)
    assert np.array_equal(mask, connectivity_mask(variant))
    assert mask.shape == (MODEL.n_input, MODEL.n_exc)
    assert mask.mean() == pytest.approx(0.125, abs=0.002)
    assert np.all(mask.sum(axis=0) > 0)

    parent = np.arange(1, MODEL.n_input * MODEL.n_exc + 1, dtype=np.float64)
    parent = parent.reshape(MODEL.n_input, MODEL.n_exc)
    weights, returned_mask = prepare_initial_weights(parent, variant)
    assert np.array_equal(mask, returned_mask)
    assert np.all(weights[~mask] == 0.0)
    assert np.allclose(weights.sum(axis=0), MODEL.normalization_target)


def test_sparse_normalization_weight_bound() -> None:
    variant = get_variant("one-trace-bernoulli-0125")
    weights = np.zeros((MODEL.n_input, MODEL.n_exc), dtype=np.float64)
    weights[0, 0] = variant.weight_max * 1.02
    validate_normalized_weight_bound(weights, variant)
    weights[0, 0] += 1.0e-6
    with pytest.raises(RuntimeError, match="normalization exceeded wmax tolerance"):
        validate_normalized_weight_bound(weights, variant)
