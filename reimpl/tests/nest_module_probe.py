#!/usr/bin/env python3
"""Integration probes for a built ZD3 NEST extension.

Run with the source-built PyNEST on PYTHONPATH and the extension build
directory on LD_LIBRARY_PATH.
"""

from __future__ import annotations

from math import exp


def main() -> int:
    import nest

    # A spike stamped at 1.0 ms with one-resolution latency must first affect
    # the target's 1.0 -> 1.5 ms integration cycle. The logger records the
    # post-update value, so ge is zero at 1.0 ms and decayed from 2 at 1.5 ms.
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.5})
    nest.Install("zd3module")
    source = nest.Create("spike_generator", params={"spike_times": [1.0]})
    target = nest.Create(
        "zd3_midpoint_neuron",
        params={"plasticity": 0.0, "V_m": -65.0, "ge": 0.0, "tau_ge": 1.0},
    )
    meter = nest.Create(
        "multimeter", params={"interval": 0.5, "record_from": ["ge"]}
    )
    nest.Connect(meter, target)
    nest.Connect(
        source,
        target,
        syn_spec={"synapse_model": "static_synapse", "weight": 2.0, "delay": 0.5},
    )
    assert float(nest.GetConnections(source, target).delay) == 0.5
    nest.Simulate(2.0)
    recorded = dict(zip(meter.events["times"], meter.events["ge"], strict=True))
    assert recorded[1.0] == 0.0, recorded
    assert abs(recorded[1.5] - 2.0 * exp(-0.5)) < 1e-12, recorded

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.5})
    nest.Install("zd3module")
    neuron = nest.Create(
        "zd3_midpoint_neuron",
        params={
            "V_m": -61.0,
            "ge": 3.0,
            "gi": 2.0,
            "theta": 20.0,
            "tau_m": 100.0,
            "tau_ge": 1.0,
            "tau_gi": 2.0,
        },
    )
    nest.Simulate(0.5)
    ge_mid = 3.0 * exp(-0.25)
    gi_mid = 2.0 * exp(-0.125)
    g_mid = 1.0 + ge_mid + gi_mid
    v_inf = (-65.0 + gi_mid * -100.0) / g_mid
    expected_v = v_inf + (-61.0 - v_inf) * exp(-0.5 * g_mid / 100.0)
    assert abs(neuron.V_m - expected_v) < 1e-12
    assert abs(neuron.ge - 3.0 * exp(-0.5)) < 1e-12
    assert abs(neuron.gi - 2.0 * exp(-0.25)) < 1e-12

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.5})
    nest.Install("zd3module")
    pre = nest.Create("spike_generator", params={"spike_times": [20.0]})
    drive = nest.Create(
        "spike_generator", params={"spike_times": [1.0, 6.0, 11.0]}
    )
    post = nest.Create(
        "zd3_midpoint_neuron",
        params={
            "plasticity": 0.0,
            "t_ref": 1.0,
            "V_m": -65.0,
            "ff_raw_sum": 0.5,
        },
    )
    nest.Connect(
        pre,
        post,
        syn_spec={
            "synapse_model": "zd3_triplet_synapse",
            "weight": 0.5,
            "delay": 0.5,
        },
    )
    nest.Connect(
        drive,
        post,
        syn_spec={"synapse_model": "static_synapse", "weight": 100.0, "delay": 0.5},
    )
    plastic = nest.GetConnections(pre, post)
    nest.Simulate(25.0)
    assert post.spike_count >= 2
    assert 0.4999 < plastic.weight < 0.5

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.5})
    nest.Install("zd3module")
    pre = nest.Create("spike_generator", params={"spike_times": [1.0, 20.0]})
    drive = nest.Create("spike_generator", params={"spike_times": [6.0, 11.0]})
    post = nest.Create(
        "zd3_midpoint_neuron",
        params={
            "plasticity": 0.0,
            "t_ref": 1.0,
            "V_m": -65.0,
            "ff_raw_sum": 0.5,
        },
    )
    nest.Connect(
        pre,
        post,
        syn_spec={
            "synapse_model": "zd3_triplet_synapse",
            "weight": 0.5,
            "delay": 0.5,
        },
    )
    nest.Connect(
        drive,
        post,
        syn_spec={"synapse_model": "static_synapse", "weight": 50.0, "delay": 0.5},
    )
    plastic = nest.GetConnections(pre, post)
    nest.Simulate(25.0)
    assert post.spike_count >= 2, (post.spike_count, plastic.weight)
    assert plastic.weight > 0.504, plastic.weight
    print("NEST_MODULE_PROBE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
