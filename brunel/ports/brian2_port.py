from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ports.common import (
    DELAY_MS,
    DT_MS,
    JE_PA,
    REFRACTORY_MS,
    TAU_MINUS_MS,
    TAU_PLUS_MS,
    STDP_TIMING_MODES,
    STDP_TIE_MODES,
    VM_MEAN_MV,
    VM_STD_MV,
    V_RESET_MV,
    V_THRESHOLD_MV,
    Model,
    alpha_propagator,
    base_manifest,
    create_output,
    make_model,
    spike_stats,
    stdp_post_path_delay_ms,
    weight_stats,
)


def _import_brian2() -> Any:
    try:
        import brian2 as b
    except ImportError as error:
        raise SystemExit("Brian2 is unavailable; use brunel/run_with_reimpl_env.sh") from error
    return b


def fixed_indegree_arrays(
    *,
    source_count: int,
    target_count: int,
    indegree: int,
    rng: np.random.Generator,
    exclude_autapses: bool,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.repeat(np.arange(target_count, dtype=np.int32), indegree)
    if exclude_autapses:
        if source_count != target_count or source_count < 2:
            raise ValueError("autapse exclusion requires equal populations of at least two neurons")
        source = rng.integers(0, source_count - 1, size=target.size, dtype=np.int32)
        source += source >= target
    else:
        source = rng.integers(0, source_count, size=target.size, dtype=np.int32)
    return source, target


def _connect_fixed_indegree(
    synapses: Any,
    *,
    source_count: int,
    target_count: int,
    indegree: int,
    rng: np.random.Generator,
    exclude_autapses: bool,
    target_chunk: int,
) -> None:
    for first in range(0, target_count, target_chunk):
        count = min(target_chunk, target_count - first)
        source, target = fixed_indegree_arrays(
            source_count=source_count,
            target_count=count,
            indegree=indegree,
            rng=rng,
            exclude_autapses=False,
        )
        target += first
        if exclude_autapses:
            source = rng.integers(0, source_count - 1, size=target.size, dtype=np.int32)
            source += source >= target
        synapses.connect(i=source, j=target)


class Brian2Brunel:
    def __init__(
        self,
        *,
        model: Model,
        seed: int,
        state_seed: int | None,
        n_record: int,
        codegen_target: str,
        connectivity_target_chunk: int,
        stdp_timing: str = "nest_dendritic",
        stdp_tie_order: str = "nest_causal_boundary",
    ) -> None:
        b = _import_brian2()
        self.b = b
        self.spec = model
        b.start_scope()
        b.defaultclock.dt = DT_MS * b.ms
        b.seed(seed)
        b.prefs.codegen.target = codegen_target
        rng = np.random.default_rng(seed)
        state_rng = rng if state_seed is None else np.random.default_rng(state_seed)
        p = alpha_propagator()
        equations = """
        dv/dt = (v_next - v) / integration_dt : 1 (unless refractory)
        v_next = p31 * dIex + p32 * Iex + p31 * dIin + p32 * Iin + p33 * v : 1
        dIex/dt = (Iex_next - Iex) / integration_dt : 1
        Iex_next = p21 * dIex + p22 * Iex : 1
        ddIex/dt = (dIex_next - dIex) / integration_dt : 1
        dIex_next = p11 * dIex : 1
        dIin/dt = (Iin_next - Iin) / integration_dt : 1
        Iin_next = p21 * dIin + p22 * Iin : 1
        ddIin/dt = (dIin_next - dIin) / integration_dt : 1
        dIin_next = p11 * dIin : 1
        integration_dt : second (constant)
        p11 : 1 (constant)
        p21 : 1 (constant)
        p22 : 1 (constant)
        p31 : 1 (constant)
        p32 : 1 (constant)
        p33 : 1 (constant)
        """
        namespace = {"v_reset": V_RESET_MV}
        self.exc = b.NeuronGroup(
            model.ne,
            equations,
            threshold=f"v >= {V_THRESHOLD_MV}",
            reset="v = v_reset",
            refractory=REFRACTORY_MS * b.ms,
            method="euler",
            namespace=namespace,
            name="exc",
        )
        self.inh = b.NeuronGroup(
            model.ni,
            equations,
            threshold=f"v >= {V_THRESHOLD_MV}",
            reset="v = v_reset",
            refractory=REFRACTORY_MS * b.ms,
            method="euler",
            namespace=namespace,
            name="inh",
        )
        for population in (self.exc, self.inh):
            population.integration_dt = DT_MS * b.ms
            population.p11 = p["p11"]
            population.p21 = p["p21"]
            population.p22 = p["p22"]
            population.p31 = p["p31"]
            population.p32 = p["p32"]
            population.p33 = p["p33"]
            population.Iex = 0.0
            population.Iin = 0.0
            population.dIex = 0.0
            population.dIin = 0.0
        self.exc.v = state_rng.normal(VM_MEAN_MV, VM_STD_MV, model.ne)
        self.inh.v = state_rng.normal(VM_MEAN_MV, VM_STD_MV, model.ni)
        if state_seed is not None:
            # Preserve the legacy connectivity RNG position while changing only V_m.
            rng.normal(VM_MEAN_MV, VM_STD_MV, model.ne)
            rng.normal(VM_MEAN_MV, VM_STD_MV, model.ni)

        external_source_rate = model.external_rate_hz / model.ce
        external_weight = p["epsc_initial"] * JE_PA
        self.external_to_exc = b.PoissonInput(
            self.exc,
            "dIex",
            N=model.ce,
            rate=external_source_rate * b.Hz,
            weight=external_weight,
        )
        self.external_to_inh = b.PoissonInput(
            self.inh,
            "dIex",
            N=model.ce,
            rate=external_source_rate * b.Hz,
            weight=external_weight,
        )

        synapse_model = """
        w : 1
        dpre_trace/dt = -pre_trace / tau_plus : 1 (event-driven)
        dpost_trace/dt = -post_trace / tau_minus : 1 (event-driven)
        last_pre_time : second
        last_post_time : second
        """
        common_namespace = {
            "epsc_initial": p["epsc_initial"],
            "tau_plus": TAU_PLUS_MS * b.ms,
            "tau_minus": TAU_MINUS_MS * b.ms,
            "learning_rate": model.rule.learning_rate,
            "depression_ratio": model.rule.depression_ratio,
            "mu_plus": model.rule.mu_plus,
        }
        if model.rule.name == "additive":
            assert model.rule.weight_max_pa is not None
            effective_post = (
                "post_trace - int(last_post_time == t)"
                if stdp_tie_order == "nest_causal_boundary"
                else "post_trace"
            )
            on_pre = f"""
            w = clip(w - depression_ratio * learning_rate * weight_max * ({effective_post}), 0, weight_max)
            dIex_post += epsc_initial * w
            pre_trace += 1
            """
            effective_pre = (
                "pre_trace - int(last_pre_time == t)"
                if stdp_tie_order == "nest_exclude_zero"
                else "pre_trace"
            )
            on_post = f"""
            w = clip(w + learning_rate * weight_max * ({effective_pre}), 0, weight_max)
            post_trace += 1
            """
            common_namespace["weight_max"] = model.rule.weight_max_pa
        else:
            effective_post = (
                "post_trace - int(last_post_time == t)"
                if stdp_tie_order == "nest_causal_boundary"
                else "post_trace"
            )
            on_pre = f"""
            w = clip(w - learning_rate * depression_ratio * w * ({effective_post}), 0, inf)
            dIex_post += epsc_initial * w
            pre_trace += 1
            """
            effective_pre = (
                "pre_trace - int(last_pre_time == t)"
                if stdp_tie_order == "nest_exclude_zero"
                else "pre_trace"
            )
            on_post = f"""
            w += learning_rate * (w ** mu_plus) * ({effective_pre})
            post_trace += 1
            """
        on_pre += "\nlast_pre_time = t"
        on_post += "\nlast_post_time = t"
        self.ee = b.Synapses(
            self.exc,
            self.exc,
            model=synapse_model,
            on_pre=on_pre,
            on_post=on_post,
            delay=DELAY_MS * b.ms,
            namespace=common_namespace,
            name="ee",
        )
        _connect_fixed_indegree(
            self.ee,
            source_count=model.ne,
            target_count=model.ne,
            indegree=model.ce,
            rng=rng,
            exclude_autapses=True,
            target_chunk=connectivity_target_chunk,
        )
        self.ee.w = JE_PA
        self.ee.last_pre_time = -1.0 * b.second
        self.ee.last_post_time = -1.0 * b.second
        self.ee.post.delay = stdp_post_path_delay_ms(stdp_timing) * b.ms
        if stdp_tie_order in ("nest_post_first", "nest_causal_boundary"):
            self.ee.post.order = self.ee.pre.order - 1
        elif stdp_tie_order == "nest_exclude_zero":
            self.ee.pre.order = -1
            self.ee.post.order = 1

        static_ex_code = "dIex_post += epsc_initial * static_weight"
        static_in_code = "dIin_post += epsc_initial * static_weight"
        static_ex_namespace = {"epsc_initial": p["epsc_initial"], "static_weight": JE_PA}
        static_in_namespace = {
            "epsc_initial": p["epsc_initial"],
            "static_weight": -model.rule.inhibitory_weight_ratio * JE_PA,
        }
        self.ie = b.Synapses(
            self.inh,
            self.exc,
            on_pre=static_in_code,
            delay=DELAY_MS * b.ms,
            namespace=static_in_namespace,
            name="ie",
        )
        _connect_fixed_indegree(
            self.ie,
            source_count=model.ni,
            target_count=model.ne,
            indegree=model.ci,
            rng=rng,
            exclude_autapses=False,
            target_chunk=connectivity_target_chunk,
        )
        self.ei = b.Synapses(
            self.exc,
            self.inh,
            on_pre=static_ex_code,
            delay=DELAY_MS * b.ms,
            namespace=static_ex_namespace,
            name="ei",
        )
        _connect_fixed_indegree(
            self.ei,
            source_count=model.ne,
            target_count=model.ni,
            indegree=model.ce,
            rng=rng,
            exclude_autapses=False,
            target_chunk=connectivity_target_chunk,
        )
        self.ii = b.Synapses(
            self.inh,
            self.inh,
            on_pre=static_in_code,
            delay=DELAY_MS * b.ms,
            namespace=static_in_namespace,
            name="ii",
        )
        _connect_fixed_indegree(
            self.ii,
            source_count=model.ni,
            target_count=model.ni,
            indegree=model.ci,
            rng=rng,
            exclude_autapses=True,
            target_chunk=connectivity_target_chunk,
        )
        self.n_record = min(n_record, model.ne)
        self.spikes = b.SpikeMonitor(self.exc[: self.n_record], name="exc_spikes")
        self.network = b.Network(
            self.exc,
            self.inh,
            self.external_to_exc,
            self.external_to_inh,
            self.ee,
            self.ie,
            self.ei,
            self.ii,
            self.spikes,
        )
        self.network.run(0 * b.ms, namespace={})

    def sample_weights(self, requested: int) -> np.ndarray:
        count = len(self.ee)
        sample_count = min(requested, count)
        indices = np.linspace(0, count - 1, sample_count, dtype=np.int64)
        return np.asarray(self.ee.w[indices], dtype=np.float64)


def run(args: argparse.Namespace) -> int:
    b = _import_brian2()
    model = make_model(args.rule, args.network_scale, args.indegree_scale)
    manifest = base_manifest("brian2-runtime", model)
    manifest.update(
        {
            "brian2_version": getattr(b, "__version__", "unknown"),
            "codegen_target": args.codegen_target,
            "connectivity": "fixed indegree with replacement; recurrent autapses excluded",
            "seed": args.seed,
            "state_seed": args.state_seed if args.state_seed is not None else args.seed,
            "presim_ms": args.presim_ms,
            "sim_ms": args.sim_ms,
            "chunk_ms": args.chunk_ms,
            "abort_rate_hz": args.abort_rate_hz,
            "stdp_timing": args.stdp_timing,
            "stdp_tie_order": args.stdp_tie_order,
            "stdp_post_path_delay_ms": stdp_post_path_delay_ms(args.stdp_timing),
        }
    )
    create_output(args.output, manifest)
    program_started = time.perf_counter()
    construction_started = time.perf_counter()
    network = Brian2Brunel(
        model=model,
        seed=args.seed,
        state_seed=args.state_seed,
        n_record=args.record_neurons,
        codegen_target=args.codegen_target,
        connectivity_target_chunk=args.connectivity_target_chunk,
        stdp_timing=args.stdp_timing,
        stdp_tie_order=args.stdp_tie_order,
    )
    construction_wall = time.perf_counter() - construction_started
    initial_weights = network.sample_weights(args.weight_sample_size)
    initial_stats = weight_stats(initial_weights, model.rule)
    print(
        f"CONFIG backend=brian2-runtime rule={args.rule} dt_ms={DT_MS} "
        f"delay_ms={DELAY_MS} ne={model.ne} ni={model.ni} ce={model.ce} ci={model.ci} "
        f"recurrent_synapses={model.recurrent_synapses} tau_plus_ms={TAU_PLUS_MS} "
        f"tau_minus_ms={TAU_MINUS_MS}",
        flush=True,
    )
    presim_started = time.perf_counter()
    network.network.run(args.presim_ms * b.ms, namespace={})
    presim_wall = time.perf_counter() - presim_started
    baseline_events = network.spikes.num_spikes
    previous_events = baseline_events
    periodic = []
    elapsed_ms = 0.0
    simulation_wall = 0.0
    termination_reason = "requested_duration_completed"
    while elapsed_ms < args.sim_ms - 1e-12:
        duration_ms = min(args.chunk_ms, args.sim_ms - elapsed_ms)
        started = time.perf_counter()
        network.network.run(duration_ms * b.ms, namespace={})
        call_wall = time.perf_counter() - started
        simulation_wall += call_wall
        elapsed_ms += duration_ms
        events = network.spikes.num_spikes
        chunk_events = events - previous_events
        previous_events = events
        sampled = network.sample_weights(args.weight_sample_size)
        stats = weight_stats(sampled, model.rule)
        row = {
            "elapsed_ms": elapsed_ms,
            "chunk_ms": duration_ms,
            "chunk_wall_seconds": call_wall,
            "chunk_seconds_per_step": call_wall / round(duration_ms / DT_MS),
            "chunk_rate_hz": chunk_events / (network.n_record * duration_ms) * 1000.0,
            "cumulative_rate_hz": (events - baseline_events)
            / (network.n_record * elapsed_ms)
            * 1000.0,
            "weight_mean": stats["mean"],
            "weight_std": stats["std"],
            "weight_boundary_fraction": stats["boundary_fraction"],
        }
        periodic.append(row)
        print("PERIODIC " + json.dumps(row, sort_keys=True), flush=True)
        if args.abort_rate_hz is not None and row["chunk_rate_hz"] >= args.abort_rate_hz:
            termination_reason = (
                f"chunk_rate_hz={row['chunk_rate_hz']} reached abort_rate_hz={args.abort_rate_hz}"
            )
            print("EARLY_STOP " + termination_reason, flush=True)
            break
    final_weights = network.sample_weights(args.weight_sample_size)
    final_stats = weight_stats(final_weights, model.rule)
    times = np.asarray(network.spikes.t / b.ms, dtype=np.float64) - args.presim_ms
    senders = np.asarray(network.spikes.i, dtype=np.int64)
    keep = (times >= 0.0) & (times <= elapsed_ms + DT_MS * 0.5)
    final_spikes = spike_stats(times[keep], senders[keep], elapsed_ms, network.n_record)
    simulation_steps = round(elapsed_ms / DT_MS)
    presim_steps = round(args.presim_ms / DT_MS)
    timing = {
        "construction_wall_seconds": construction_wall,
        "presimulation_wall_seconds": presim_wall,
        "simulation_wall_seconds": simulation_wall,
        "whole_program_wall_seconds": time.perf_counter() - program_started,
        "presimulation_steps": presim_steps,
        "simulation_steps": simulation_steps,
        "requested_simulation_steps": round(args.sim_ms / DT_MS),
        "actual_simulation_ms": elapsed_ms,
        "requested_simulation_ms": args.sim_ms,
        "simulation_seconds_per_step": simulation_wall / simulation_steps,
        "combined_simulate_call_seconds_per_step": (presim_wall + simulation_wall)
        / (presim_steps + simulation_steps),
    }
    results = {
        "manifest": manifest,
        "timing": timing,
        "initial_weight_stats": initial_stats,
        "final_weight_stats": final_stats,
        "spike_stats": final_spikes,
        "periodic": periodic,
        "termination": {
            "completed_requested_duration": termination_reason
            == "requested_duration_completed",
            "reason": termination_reason,
        },
    }
    with (args.output / "results.json").open("x", encoding="ascii") as stream:
        json.dump(results, stream, indent=2, sort_keys=True)
        stream.write("\n")
    np.savez_compressed(args.output / "weight_samples.npz", initial=initial_weights, final=final_weights)
    print(
        "RESULT "
        + json.dumps(
            {"timing": timing, "spike_stats": final_spikes, "final_weight_stats": final_stats},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brian2 port of the measured Brunel STDP models")
    parser.add_argument("--rule", choices=("additive", "morrison"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--state-seed", type=int)
    parser.add_argument("--network-scale", type=float, default=1.0)
    parser.add_argument("--indegree-scale", type=float, default=1.0)
    parser.add_argument("--presim-ms", type=float, default=100.0)
    parser.add_argument("--sim-ms", type=float, default=1000.0)
    parser.add_argument("--chunk-ms", type=float, default=100.0)
    parser.add_argument("--abort-rate-hz", type=float, default=100.0)
    parser.add_argument("--record-neurons", type=int, default=1000)
    parser.add_argument("--weight-sample-size", type=int, default=100000)
    parser.add_argument("--connectivity-target-chunk", type=int, default=32)
    parser.add_argument("--codegen-target", choices=("cython", "numpy"), default="cython")
    parser.add_argument("--stdp-timing", choices=STDP_TIMING_MODES, default="nest_dendritic")
    parser.add_argument(
        "--stdp-tie-order", choices=STDP_TIE_MODES, default="nest_causal_boundary"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in ("network_scale", "indegree_scale", "sim_ms", "chunk_ms"):
        if getattr(args, name) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.presim_ms < 0.0:
        raise SystemExit("--presim-ms must be non-negative")
    for name in ("presim_ms", "sim_ms", "chunk_ms"):
        steps = getattr(args, name) / DT_MS
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise SystemExit(f"--{name.replace('_', '-')} must be an integer number of timesteps")
    return run(args)
