from dataclasses import dataclass
from math import exp

from .constants import MODEL, ModelConstants


@dataclass(frozen=True)
class NeuronStep:
    v_mv: float
    ge: float
    gi: float
    theta_mv: float


def midpoint_step(
    *,
    v_mv: float,
    ge: float,
    gi: float,
    theta_mv: float,
    tau_m_ms: float,
    v_rest_mv: float,
    e_exc_mv: float,
    e_inh_mv: float,
    refractory: bool = False,
    plasticity: bool = True,
    constants: ModelConstants = MODEL,
) -> NeuronStep:
    """One reference midpoint step, excluding spike/reset processing.

    Brian 1's ``freeze=True`` suppresses all state equations during refractory,
    including conductance and theta decay. Synaptic arrivals are applied outside
    this function and can still accumulate while the neuron is refractory.
    """
    if refractory:
        return NeuronStep(v_mv, ge, gi, theta_mv)

    half_dt = 0.5 * constants.dt_ms
    ge_mid = ge * exp(-half_dt / constants.tau_ge_ms)
    gi_mid = gi * exp(-half_dt / constants.tau_gi_ms)
    g_mid = 1.0 + ge_mid + gi_mid
    v_inf = (
        v_rest_mv + ge_mid * e_exc_mv + gi_mid * e_inh_mv
    ) / g_mid
    v_next = v_inf + (v_mv - v_inf) * exp(
        -constants.dt_ms * g_mid / tau_m_ms
    )
    ge_next = ge * exp(-constants.dt_ms / constants.tau_ge_ms)
    gi_next = gi * exp(-constants.dt_ms / constants.tau_gi_ms)
    theta_next = theta_mv
    if plasticity:
        theta_next *= exp(-constants.dt_ms / constants.theta_tau_ms)
    return NeuronStep(v_next, ge_next, gi_next, theta_next)


@dataclass
class TraceState:
    weight: float
    pre: float = 0.0
    post1: float = 0.0
    post2: float = 0.0
    time_ms: float = 0.0

    def decay_to(self, time_ms: float, constants: ModelConstants = MODEL) -> None:
        if time_ms < self.time_ms:
            raise ValueError("trace events must be applied in nondecreasing time order")
        elapsed = time_ms - self.time_ms
        self.pre *= exp(-elapsed / constants.pre_tau_ms)
        self.post1 *= exp(-elapsed / constants.post1_tau_ms)
        self.post2 *= exp(-elapsed / constants.post2_tau_ms)
        self.time_ms = time_ms

    def on_pre(self, time_ms: float, constants: ModelConstants = MODEL) -> None:
        self.decay_to(time_ms, constants)
        self.pre = 1.0
        self.weight = max(
            constants.weight_min,
            min(constants.weight_max, self.weight - constants.depression_rate * self.post1),
        )

    def on_post(self, time_ms: float, constants: ModelConstants = MODEL) -> None:
        self.decay_to(time_ms, constants)
        post2_before = self.post2
        self.weight = max(
            constants.weight_min,
            min(
                constants.weight_max,
                self.weight + constants.potentiation_rate * self.pre * post2_before,
            ),
        )
        self.post1 = 1.0
        self.post2 = 1.0


def trace_from_last_spike(
    event_time_ms: float,
    last_spike_ms: float | None,
    tau_ms: float,
) -> float:
    """Return a nearest-neighbour trace sampled at an event time."""
    if last_spike_ms is None:
        return 0.0
    if last_spike_ms > event_time_ms:
        raise ValueError("last spike cannot be later than the sampled event")
    return exp(-(event_time_ms - last_spike_ms) / tau_ms)


def event_time_weight_updates(
    *,
    weight: float,
    pre_time_ms: float | None = None,
    post_time_ms: float | None = None,
    previous_pre_time_ms: float | None = None,
    previous_post_time_ms: float | None = None,
    apply_pre: bool = False,
    apply_post: bool = False,
    constants: ModelConstants = MODEL,
) -> float:
    """Apply event-time form of the rule used by the GeNN implementation.

    When pre and post fire at the same time, depression samples the preceding
    postsynaptic spike while potentiation sees the current presynaptic spike.
    """
    if apply_pre:
        if pre_time_ms is None:
            raise ValueError("a presynaptic update requires pre_time_ms")
        post_before_pre = post_time_ms
        if post_before_pre is not None and post_before_pre >= pre_time_ms:
            post_before_pre = previous_post_time_ms
        post1 = trace_from_last_spike(
            pre_time_ms, post_before_pre, constants.post1_tau_ms
        )
        weight -= constants.depression_rate * post1
        weight = min(constants.weight_max, max(constants.weight_min, weight))
    if apply_post:
        if post_time_ms is None:
            raise ValueError("a postsynaptic update requires post_time_ms")
        pre_before_post = pre_time_ms
        if pre_before_post is not None and pre_before_post > post_time_ms:
            pre_before_post = previous_pre_time_ms
        pre = trace_from_last_spike(
            post_time_ms, pre_before_post, constants.pre_tau_ms
        )
        post2_before = trace_from_last_spike(
            post_time_ms, previous_post_time_ms, constants.post2_tau_ms
        )
        weight += constants.potentiation_rate * pre * post2_before
        weight = min(constants.weight_max, max(constants.weight_min, weight))
    return weight
