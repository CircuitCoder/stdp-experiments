from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from ports.common import (
    DELAY_MS,
    DT_MS,
    JE_PA,
    TAU_MINUS_MS,
    TAU_PLUS_MS,
    VM_MEAN_MV,
    VM_STD_MV,
    Model,
    base_manifest,
    create_output,
    make_model,
    spike_stats,
    weight_stats,
)


def _import_nestgpu() -> Any:
    try:
        import nestgpu
    except ImportError as error:
        raise SystemExit(
            "NEST GPU is unavailable. Build the vendored 3rdparty/nest-gpu tree "
            "after exposing CUDA, then put its pythonlib and shared library on the search path."
        ) from error
    return nestgpu


def _flatten_spikes(spike_lists: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    senders: list[int] = []
    for sender, neuron_times in enumerate(spike_lists):
        times.extend(neuron_times)
        senders.extend([sender] * len(neuron_times))
    return np.asarray(times, dtype=np.float64), np.asarray(senders, dtype=np.int64)


class NESTGPUBrunel:
    def __init__(
        self,
        *,
        ngpu: Any,
        spec: Model,
        seed: int,
        n_record: int,
        max_recorded_spikes_per_neuron: int,
        weight_sample_size: int,
    ) -> None:
        self.ngpu = ngpu
        self.spec = spec
        ngpu.SetKernelStatus(
            {
                "verbosity_level": 0,
                "rnd_seed": seed,
                "time_resolution": DT_MS,
                "min_allowed_delay": DELAY_MS,
                "spike_buffer_algo": 0,
            }
        )
        neuron_params = {
            "E_L": 0.0,
            "C_m": 250.0,
            "tau_m": 10.0,
            "t_ref": 0.5,
            "Theta_rel": 20.0,
            "V_reset_rel": 0.0,
            "tau_syn_ex": 0.32582722403722841,
            "tau_syn_in": 0.32582722403722841,
            "V_m_rel": 0.0,
        }
        self.exc = ngpu.Create("iaf_psc_alpha", spec.ne, 1, neuron_params)
        self.inh = ngpu.Create("iaf_psc_alpha", spec.ni, 1, neuron_params)
        random_vm = {
            "distribution": "normal",
            "mu": VM_MEAN_MV,
            "sigma": VM_STD_MV,
        }
        ngpu.SetStatus(self.exc, {"V_m_rel": random_vm})
        ngpu.SetStatus(self.inh, {"V_m_rel": random_vm})
        # NEST GPU only permits recorded spike times for a complete node group.
        # Keep the full group as the recorder target and restrict the returned
        # spike lists to the requested measurement population in spikes().
        self.recorded = self.exc
        self.n_record = n_record
        ngpu.ActivateRecSpikeTimes(self.recorded, max_recorded_spikes_per_neuron)

        stimulus = ngpu.Create("poisson_generator", 1, 1, {"rate": spec.external_rate_hz})
        static_ex = {"weight": JE_PA, "delay": DELAY_MS, "receptor": 0}
        static_in = {
            "weight": -spec.rule.inhibitory_weight_ratio * JE_PA,
            "delay": DELAY_MS,
            "receptor": 0,
        }
        ngpu.Connect(stimulus, self.exc, {"rule": "all_to_all"}, static_ex)
        ngpu.Connect(stimulus, self.inh, {"rule": "all_to_all"}, static_ex)

        if spec.rule.name == "additive":
            assert spec.rule.weight_max_pa is not None
            plastic_group = ngpu.CreateSynGroup(
                "stdp",
                {
                    "tau_plus": TAU_PLUS_MS,
                    "tau_minus": TAU_MINUS_MS,
                    "lambda": spec.rule.learning_rate,
                    "alpha": spec.rule.depression_ratio,
                    "mu_plus": spec.rule.mu_plus,
                    "mu_minus": spec.rule.mu_minus,
                    "Wmax": spec.rule.weight_max_pa,
                },
            )
        else:
            plastic_group = ngpu.CreateSynGroup(
                "stdp_pl",
                {
                    "tau_plus": TAU_PLUS_MS,
                    "tau_minus": TAU_MINUS_MS,
                    "lambda": spec.rule.learning_rate,
                    "alpha": spec.rule.depression_ratio,
                    "mu": spec.rule.mu_plus,
                },
            )
        plastic = {
            "weight": JE_PA,
            "delay": DELAY_MS,
            "receptor": 0,
            "synapse_group": plastic_group,
        }
        ngpu.Connect(
            self.exc,
            self.exc,
            {"rule": "fixed_indegree", "indegree": spec.ce},
            plastic,
        )
        ngpu.Connect(
            self.inh,
            self.exc,
            {"rule": "fixed_indegree", "indegree": spec.ci},
            static_in,
        )
        ngpu.Connect(
            self.exc,
            self.inh,
            {"rule": "fixed_indegree", "indegree": spec.ce},
            static_ex,
        )
        ngpu.Connect(
            self.inh,
            self.inh,
            {"rule": "fixed_indegree", "indegree": spec.ci},
            static_in,
        )
        started = time.perf_counter()
        ngpu.Calibrate()
        self.calibration_wall_seconds = time.perf_counter() - started
        side = max(1, min(spec.ne, math.ceil(math.sqrt(weight_sample_size))))
        sampled_exc = self.exc[0:side]
        self.sample_connections = ngpu.GetConnections(sampled_exc, sampled_exc)
        if len(self.sample_connections) > weight_sample_size:
            self.sample_connections = self.sample_connections[:weight_sample_size]
        if len(self.sample_connections) == 0:
            raise RuntimeError("NEST GPU returned an empty E-to-E weight sample")

    def simulate(self, duration_ms: float) -> float:
        started = time.perf_counter()
        self.ngpu.Simulate(duration_ms)
        return time.perf_counter() - started

    def spikes(self) -> tuple[np.ndarray, np.ndarray]:
        spike_lists = self.ngpu.GetRecSpikeTimes(self.recorded)
        return _flatten_spikes(spike_lists[: self.n_record])

    def weights(self) -> np.ndarray:
        return np.asarray(
            self.ngpu.GetStatus(self.sample_connections, "weight"), dtype=np.float64
        )


def run(args: argparse.Namespace) -> int:
    spec = make_model(args.rule, args.network_scale, args.indegree_scale)
    manifest = base_manifest("nest-gpu", spec)
    manifest.update(
        {
            "seed": args.seed,
            "presim_ms": args.presim_ms,
            "sim_ms": args.sim_ms,
            "chunk_ms": args.chunk_ms,
            "abort_rate_hz": args.abort_rate_hz,
            "plasticity_timing": (
                "nearest signed pre/post interval; NEST GPU connection storage does not "
                "provide the all-to-all accumulated traces used by NEST"
            ),
            "connectivity": (
                "fixed indegree with replacement; this NEST GPU API cannot disable "
                "recurrent autapses"
            ),
            "external_input": "one Poisson generator with an independent stream per target",
            "morrison_extension": "3rdparty/nest-gpu/src/stdp_pl.{h,cu}",
        }
    )
    create_output(args.output, manifest)
    if args.dry_run:
        with (args.output / "validation.json").open("x", encoding="ascii") as stream:
            json.dump({"manifest": manifest, "status": "configuration_only"}, stream, indent=2)
            stream.write("\n")
        print("VALIDATED " + json.dumps(manifest, sort_keys=True), flush=True)
        return 0

    ngpu = _import_nestgpu()
    program_started = time.perf_counter()
    n_record = min(args.record_neurons, spec.ne)
    max_recorded = max(
        10,
        math.ceil((args.presim_ms + args.sim_ms) / 1000.0 * args.record_capacity_rate_hz),
    )
    construction_started = time.perf_counter()
    network = NESTGPUBrunel(
        ngpu=ngpu,
        spec=spec,
        seed=args.seed,
        n_record=n_record,
        max_recorded_spikes_per_neuron=max_recorded,
        weight_sample_size=args.weight_sample_size,
    )
    construction_wall = time.perf_counter() - construction_started
    initial_weights = network.weights()
    initial_stats = weight_stats(initial_weights, spec.rule)
    print(
        f"CONFIG backend=nest-gpu rule={args.rule} dt_ms={DT_MS} delay_ms={DELAY_MS} "
        f"ne={spec.ne} ni={spec.ni} ce={spec.ce} ci={spec.ci} "
        f"recurrent_synapses={spec.recurrent_synapses} tau_plus_ms={TAU_PLUS_MS} "
        f"tau_minus_ms={TAU_MINUS_MS}",
        flush=True,
    )
    presim_wall = network.simulate(args.presim_ms) if args.presim_ms else 0.0
    if args.presim_ms:
        # GetRecSpikeTimes drains NEST GPU's recording buffer.
        network.spikes()
    elapsed_ms = 0.0
    simulation_wall = 0.0
    periodic = []
    recorded_times: list[np.ndarray] = []
    recorded_ids: list[np.ndarray] = []
    cumulative_events = 0
    termination_reason = "requested_duration_completed"
    while elapsed_ms < args.sim_ms - 1e-12:
        duration_ms = min(args.chunk_ms, args.sim_ms - elapsed_ms)
        call_wall = network.simulate(duration_ms)
        simulation_wall += call_wall
        elapsed_ms += duration_ms
        times, ids = network.spikes()
        start = args.presim_ms + elapsed_ms - duration_ms
        stop = args.presim_ms + elapsed_ms
        keep = (times > start) & (times <= stop + DT_MS * 0.5)
        recorded_times.append(times[keep] - args.presim_ms)
        recorded_ids.append(ids[keep])
        chunk_events = int(np.count_nonzero(keep))
        cumulative_events += chunk_events
        sampled = network.weights()
        stats = weight_stats(sampled, spec.rule)
        steps = round(duration_ms / DT_MS)
        row = {
            "elapsed_ms": elapsed_ms,
            "chunk_ms": duration_ms,
            "chunk_wall_seconds": call_wall,
            "chunk_seconds_per_step": call_wall / steps,
            "chunk_rate_hz": chunk_events / (n_record * duration_ms) * 1000.0,
            "cumulative_rate_hz": cumulative_events / (n_record * elapsed_ms) * 1000.0,
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

    final_weights = network.weights()
    final_stats = weight_stats(final_weights, spec.rule)
    times = np.concatenate(recorded_times) if recorded_times else np.empty(0)
    ids = np.concatenate(recorded_ids) if recorded_ids else np.empty(0, dtype=np.int64)
    final_spikes = spike_stats(times, ids, elapsed_ms, n_record)
    simulation_steps = round(elapsed_ms / DT_MS)
    presim_steps = round(args.presim_ms / DT_MS)
    timing = {
        "construction_wall_seconds": construction_wall,
        "calibration_wall_seconds": network.calibration_wall_seconds,
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
    np.savez_compressed(
        args.output / "weight_samples.npz", initial=initial_weights, final=final_weights
    )
    print(
        "RESULT "
        + json.dumps(
            {"timing": timing, "spike_stats": final_spikes, "final_weight_stats": final_stats},
            sort_keys=True,
        ),
        flush=True,
    )
    ngpu.MpiFinalize()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NEST-GPU port of the measured Brunel STDP models")
    parser.add_argument("--rule", choices=("additive", "morrison"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--network-scale", type=float, default=1.0)
    parser.add_argument("--indegree-scale", type=float, default=1.0)
    parser.add_argument("--presim-ms", type=float, default=100.0)
    parser.add_argument("--sim-ms", type=float, default=1000.0)
    parser.add_argument("--chunk-ms", type=float, default=100.0)
    parser.add_argument("--abort-rate-hz", type=float, default=100.0)
    parser.add_argument("--record-neurons", type=int, default=1000)
    parser.add_argument("--record-capacity-rate-hz", type=float, default=250.0)
    parser.add_argument("--weight-sample-size", type=int, default=100000)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "network_scale",
        "indegree_scale",
        "sim_ms",
        "chunk_ms",
        "record_capacity_rate_hz",
    ):
        if getattr(args, name) <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.presim_ms < 0.0:
        raise SystemExit("--presim-ms must be non-negative")
    for name in ("presim_ms", "sim_ms", "chunk_ms"):
        steps = getattr(args, name) / DT_MS
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise SystemExit(f"--{name.replace('_', '-')} must be an integer number of timesteps")
    return run(args)
