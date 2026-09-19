"""Compare generated GeNN updates with an independent event recurrence."""
import os
from math import exp

import numpy as np
import pytest

from backends.genn_backend import create_two_trace_model


@pytest.mark.parametrize("plasticity,nu_pre,nu_post,initial", [
    (1.0, 0.0001, 0.0005, 0.5),
    (1.0, 3.0, 3.0, 0.99999),
    (0.0, 0.0001, 0.0005, 0.5),
])
def test_two_trace_event_order(tmp_path, plasticity, nu_pre, nu_post, initial):
    genn = pytest.importorskip("pygenn")
    backend = os.environ.get("GENN_TEST_BACKEND", "single_threaded_cpu")
    model = genn.GeNNModel("float", "two_trace_probe", backend=backend)
    model.dt = 0.5
    model.seed = 20260724
    scheduled = genn.create_neuron_model(
        "Scheduled", params=["mask"],
        vars=[("step", "unsigned int"), ("received", "scalar")],
        sim_code="step++; received = Isyn;",
        threshold_condition_code="((unsigned int)mask & (1u << step)) != 0",
    )
    pre_steps, post_steps = {2, 6, 8, 9, 12}, {4, 6, 9, 10, 12}
    pre = model.add_neuron_population("Pre", 1, scheduled,
        {"mask": sum(1 << s for s in pre_steps)}, {"step": 0, "received": 0.0})
    post = model.add_neuron_population("Post", 1, scheduled,
        {"mask": sum(1 << s for s in post_steps)}, {"step": 0, "received": 0.0})
    two_trace_model = create_two_trace_model(genn.create_weight_update_model)
    syn = model.add_synapse_population("Plastic", "DENSE", pre, post,
        genn.init_weight_update(two_trace_model,
            {"preTau": 20.0, "postTau": 20.0, "potentiationRate": nu_post,
             "depressionRate": nu_pre, "preTarget": 0.4, "preExponent": 0.2,
             "postExponent": 0.2, "weightMin": 0.0, "weightMax": 1.0,
             "plasticity": plasticity}, {"g": initial},
            pre_vars={"x": 0.0}, post_vars={"y": 0.0}),
        genn.init_postsynaptic("DeltaCurr"))
    (tmp_path / "build").mkdir()
    model.build(str(tmp_path / "build"))
    model.load()
    weight, x, y = initial, 0.0, 0.0
    try:
        for step in range(1, 16):
            # Synapses consume the preceding neuron update's spikes.
            source_step = step - 1
            delivered = 0.0
            if source_step > 0:
                x *= exp(-0.5 / 20.0)
                y *= exp(-0.5 / 20.0)
                if source_step in pre_steps:
                    delivered = weight
                    x += 1.0
                    weight = np.clip(weight - plasticity * nu_pre * y * weight**0.2, 0, 1)
                if source_step in post_steps:
                    weight = np.clip(weight + plasticity * nu_post * (x - 0.4)
                                     * (1 - weight)**0.2, 0, 1)
                    y += 1.0
            model.step_time()
            syn.vars["g"].pull_from_device()
            post.vars["received"].pull_from_device()
            assert float(syn.vars["g"].values[0]) == pytest.approx(weight, abs=2e-6)
            assert float(post.vars["received"].view[0]) == pytest.approx(delivered, abs=2e-6)
    finally:
        model.unload()
