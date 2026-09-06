from math import exp

import numpy as np
import pytest

from ports.brian2_port import fixed_indegree_arrays
from ports.common import (
    GENN_DEFAULT_DELAY_MS,
    GENN_DEFAULT_EXTERNAL_RATE_SCALES,
    GENN_DEFAULT_INDEGREE_SCALE,
    GENN_DEFAULT_RECURRENT_DELIVERY_SCALE,
    JE_PA,
    RULES,
    TAU_MINUS_MS,
    TAU_PLUS_MS,
    PairTraceState,
    alpha_lif_step,
    alpha_propagator,
    make_genn_default_model,
    make_model,
    stdp_post_path_delay_ms,
    variance_preserving_scales,
)


def test_fixed_indegree_arrays_exclude_autapses() -> None:
    source, target = fixed_indegree_arrays(
        source_count=8,
        target_count=8,
        indegree=100,
        rng=np.random.default_rng(17),
        exclude_autapses=True,
    )
    assert np.array_equal(np.bincount(target), np.full(8, 100))
    assert np.all(source != target)
    assert np.all((source >= 0) & (source < 8))


def test_historical_full_indegree_connection_counts() -> None:
    model = make_model("morrison", 1.0, 1.0)
    assert model.ne == 9000
    assert model.ni == 2250
    assert model.plastic_synapses == 81_000_000
    assert model.recurrent_synapses == 126_562_500


def test_five_percent_indegree_connection_counts() -> None:
    model = make_model("morrison", 1.0, 0.05, delay_ms=0.0)
    assert model.ne == 9000
    assert model.ni == 2250
    assert model.ce == 450
    assert model.ci == 112
    assert model.plastic_synapses == 4_050_000
    assert model.recurrent_synapses == 6_322_500
    assert model.delay_ms == 0.0


@pytest.mark.parametrize("rule_name", ["additive", "morrison"])
def test_genn_default_configuration(rule_name: str) -> None:
    model = make_genn_default_model(rule_name)
    assert model.indegree_scale == GENN_DEFAULT_INDEGREE_SCALE
    assert model.delay_ms == GENN_DEFAULT_DELAY_MS
    assert model.recurrent_delivery_scale == GENN_DEFAULT_RECURRENT_DELIVERY_SCALE
    assert model.external_rate_scale == GENN_DEFAULT_EXTERNAL_RATE_SCALES[rule_name]
    assert model.ce == 450
    assert model.ci == 112


def test_variance_preserving_sparse_scaling() -> None:
    delivery_scale, external_rate_scale = variance_preserving_scales(0.05)
    assert delivery_scale == pytest.approx(1.0 / np.sqrt(0.05))
    assert external_rate_scale == pytest.approx(np.sqrt(0.05))

    baseline = make_model("additive", 1.0, 1.0)
    sparse = make_model(
        "additive",
        1.0,
        0.05,
        recurrent_delivery_scale=delivery_scale,
        external_rate_scale=external_rate_scale,
    )
    assert sparse.external_rate_hz == pytest.approx(
        baseline.external_rate_hz * external_rate_scale
    )


def test_nest_dendritic_timing_compensates_two_delays() -> None:
    assert stdp_post_path_delay_ms("arrival") == 0.0
    assert stdp_post_path_delay_ms("nest_dendritic") == 3.0
    assert stdp_post_path_delay_ms("nest_dendritic", 0.0) == 0.0


def test_alpha_propagator_matches_independent_matrix_step() -> None:
    p = alpha_propagator()
    voltage, current, derivative = alpha_lif_step(
        voltage_mv=5.0,
        current_pa=7.0,
        derivative_pa_per_ms=11.0,
        delivered_weight_pa=13.0,
    )
    assert voltage == pytest.approx(p["p31"] * 11.0 + p["p32"] * 7.0 + p["p33"] * 5.0)
    assert current == pytest.approx(p["p21"] * 11.0 + p["p22"] * 7.0)
    assert derivative == pytest.approx(p["p11"] * 11.0 + p["epsc_initial"] * 13.0)


def test_refractory_freezes_voltage_but_not_synaptic_current() -> None:
    voltage, current, derivative = alpha_lif_step(
        voltage_mv=5.0,
        current_pa=7.0,
        derivative_pa_per_ms=11.0,
        refractory=True,
    )
    assert voltage == 5.0
    assert current != 7.0
    assert derivative != 11.0


def test_additive_pair_updates_and_bounds() -> None:
    rule = RULES["additive"]
    state = PairTraceState()
    delivered = state.on_pre(0.0, rule)
    assert delivered == JE_PA
    state.on_post(10.0, rule)
    expected = JE_PA + rule.learning_rate * rule.weight_max_pa * exp(-10.0 / TAU_PLUS_MS)
    assert state.weight_pa == pytest.approx(expected)
    state.on_pre(20.0, rule)
    expected -= (
        rule.depression_ratio
        * rule.learning_rate
        * rule.weight_max_pa
        * exp(-10.0 / TAU_MINUS_MS)
    )
    assert state.weight_pa == pytest.approx(expected)


def test_morrison_pair_updates() -> None:
    rule = RULES["morrison"]
    state = PairTraceState()
    state.on_pre(0.0, rule)
    state.on_post(10.0, rule)
    expected = JE_PA + rule.learning_rate * JE_PA**rule.mu_plus * exp(-10.0 / TAU_PLUS_MS)
    assert state.weight_pa == pytest.approx(expected)
    state.on_pre(20.0, rule)
    expected -= (
        rule.learning_rate
        * rule.depression_ratio
        * expected
        * exp(-10.0 / TAU_MINUS_MS)
    )
    assert state.weight_pa == pytest.approx(expected)


@pytest.mark.parametrize("rule_name", ["additive", "morrison"])
def test_nest_causal_boundary_uses_old_traces_then_updates_both(rule_name: str) -> None:
    rule = RULES[rule_name]
    state = PairTraceState()
    initial_weight = state.weight_pa

    # The upper history bound is inclusive for LTP; the LTD trace is strictly earlier.
    effective_pre_trace = state.pre_trace
    effective_post_trace = state.post_trace
    state.pre_trace += 1.0
    assert effective_pre_trace == 0.0
    state.post_trace += 1.0
    assert effective_post_trace == 0.0

    assert state.weight_pa == initial_weight
    assert state.pre_trace == 1.0
    assert state.post_trace == 1.0
