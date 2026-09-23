"""Run generated-kernel timing checks in a process that may intentionally abort."""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import partial
import json
import math
from pathlib import Path
import resource
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "brunel")]
from ports import genn_port
from ports.common import DT_MS, RULES
from workloads.morrison import make_morrison


OFFSETS = (0, 2, 10001, 100001, 1000001, 1310695, 2000001, 5000001,
           20000001, (1 << 24) + 1, (1 << 25) + 3, (1 << 31) + 1,
           genn_port.STDP_TICK_MAX - 200)


def construct(backend, output, rule, delay):
    spec = replace(make_morrison(), ne=2, ni=2, ce=1, ci=1,
                   delay_ms=delay, external_rate_scale=0)
    if rule == "additive":
        spec = replace(spec, rule=RULES["additive"])
    build = output / f"{rule}_{delay}"
    net = genn_port.GeNNBrunel(
        spec=spec, seed=17, state_seed=17, backend=backend,
        build_path=build, reuse_build=build if build.exists() else None,
        recording_steps=0,
        precision="float", stdp_timing="nest_dendritic" if delay else "arrival",
        stdp_tie_order="nest_causal_boundary", timing_enabled=False,
        record_spikes=False, collect_connectivity_stats=False)
    return net, spec


def initialize(net, offset=0):
    net.model.timestep = offset
    net.ee.pull_connectivity_from_device()
    for pop in (net.exc, net.inh):
        pop.vars["V"].view[:] = 0
        pop.vars["V"].push_to_device()
        # Absolute-time jumps are test setup: synchronize the independent
        # integer neuron clocks while retaining an empty spike history.
        pop.vars["tick"].view[:] = offset
        pop.vars["tick"].push_to_device()


def force_spikes(net, ids):
    if ids:
        v = net.exc.vars["V"]
        v.pull_from_device()
        v.view[ids] = 21
        v.push_to_device()


def isolated_pair_expected(weight, rule, lag, old_pre):
    if lag > 0 or old_pre:
        dt = 2.0 if old_pre else DT_MS
        amplitude = rule.weight_max_pa if rule.name == "additive" else weight ** rule.mu_plus
        return weight + rule.learning_rate * amplitude * math.exp(-dt / 20)
    if lag < 0:
        amplitude = rule.weight_max_pa if rule.name == "additive" else weight
        return weight - rule.learning_rate * rule.depression_ratio * amplitude * math.exp(-DT_MS / 20)
    return weight


def pairs(args):
    rows = []
    by_schedule = {}
    for rule in ("morrison", "additive"):
        for delay in (0.0, 1.5):
            for offset in OFFSETS:
                for lag, old_pre in ((0, False), (-1, False), (1, False), (0, True)):
                    # GeNN's sparse runtime cannot safely be loaded twice on
                    # the same object. Reuse only the immutable compiled code.
                    net, spec = construct(args.backend, args.output, rule, delay)
                    initialize(net, offset)
                    net.ee.vars["g"].pull_from_device()
                    initial = float(net.ee.vars["g"].values[0])
                    post_step = 25 - round(delay / DT_MS) + lag
                    for step in range(80):
                        ids = []
                        if step == 25 or (old_pre and step == 5):
                            ids.append(0)
                        if step == post_step:
                            ids.append(1)
                        force_spikes(net, ids)
                        net.model.step_time()
                    net.ee.vars["g"].pull_from_device()
                    actual = float(net.ee.vars["g"].values[0])
                    expected = isolated_pair_expected(initial, spec.rule, lag, old_pre)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)
                    schedule = (rule, delay, lag, old_pre)
                    assert actual == by_schedule.setdefault(schedule, actual)
                    if lag == 0 and not old_pre:
                        assert actual == initial
                    pre_tick = offset + 26 + round(delay / DT_MS)
                    post_tick = pre_tick + lag
                    for field, expected_tick in (
                            ("lastPreUpdateTick", pre_tick),
                            ("lastPostUpdateTick", post_tick),
                            ("lastTraceTick", max(pre_tick, post_tick))):
                        var = net.ee.vars[field]
                        var.pull_from_device()
                        assert var.values.dtype == np.uint32
                        assert int(var.values[0]) == expected_tick
                    for pop in (net.exc, net.inh):
                        pop.vars["tick"].pull_from_device()
                        np.testing.assert_array_equal(pop.vars["tick"].view,
                                                      np.full(2, offset + 80, np.uint32))
                    net.exc.vars["spikeCount"].pull_from_device()
                    np.testing.assert_array_equal(net.exc.vars["spikeCount"].view,
                                                  [2 if old_pre else 1, 1])
                    rows.append(dict(rule=rule, delay_ms=delay, start_tick=offset,
                                     lag_ticks=lag, old_pre=old_pre,
                                     initial=initial, expected=expected, actual=actual))
                    net.close()
    return rows


def fail_guard(args):
    delay = 1.5 if args.case in ("clock-overflow-delayed", "arrival-overflow") else 0.0
    net, _ = construct(args.backend, args.output, "morrison", delay)
    initialize(net)
    if args.case.startswith("clock-overflow"):
        net.exc.vars["tick"].view[:] = net.max_spike_tick
        net.exc.vars["tick"].push_to_device()
        net.model.step_time()
    else:
        post_only = args.case.endswith("-post")
        force_spikes(net, [1] if post_only else [0])
        net.model.step_time()
        if args.case in ("missing-spike", "future-spike", "arrival-overflow"):
            var = net.exc.vars["lastSpikeTick"]
            var.pull_from_device()
            neuron = 1 if args.case == "future-spike" else 0
            value = {"missing-spike": 0, "future-spike": 2,
                     "arrival-overflow": genn_port.STDP_TICK_MAX}[args.case]
            var.view.reshape(-1, 2)[:, neuron] = value
            var.push_to_device()
        else:
            field = {"future-trace-pre": "lastTraceTick",
                     "future-post-pre": "lastPostUpdateTick",
                     "future-trace-post": "lastTraceTick",
                     "future-post-post": "lastPostUpdateTick",
                     "future-pre-post": "lastPreUpdateTick"}[args.case]
            var = net.ee.vars[field]
            var.pull_from_device()
            vals = var.values
            vals[0] = 2
            var.values = vals
            var.push_to_device()
        for _ in range(1 + round(delay / DT_MS)):
            net.model.step_time()
    # CUDA assertions are observed by the host at its next synchronization.
    net.exc.vars["spikeCount"].pull_from_device()
    raise RuntimeError("Timing guard failed to stop the invalid update")


def main():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="single_threaded_cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", default="pairs",
                        choices=("pairs", "missing-spike", "future-spike",
                                 "future-trace-pre", "future-post-pre",
                                 "future-trace-post", "future-post-post", "future-pre-post",
                                 "clock-overflow", "clock-overflow-delayed", "arrival-overflow"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    # Verify that hard errors survive optimized generated CPU/CUDA builds.
    api = genn_port._import_genn()
    api["GeNNModel"] = partial(api["GeNNModel"], optimize_code=True)
    genn_port._import_genn = lambda: api
    if args.case == "pairs":
        rows = pairs(args)
        (args.output / "result.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Validated {len(rows)} generated-kernel cases", flush=True)
    else:
        fail_guard(args)


if __name__ == "__main__":
    main()
