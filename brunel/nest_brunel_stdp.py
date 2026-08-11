#!/usr/bin/env python3
"""Parameterized NEST Brunel network for static, additive, and Morrison runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import lambertw


TAU_SYN_MS = 0.32582722403722841
BASE_NE = 9000
BASE_NI = 2250
BASE_CE = 9000
BASE_CI = 2250


@dataclass(frozen=True)
class Workload:
    rule: str
    threads: int
    seed: int
    dt_ms: float
    delay_ms: float
    presim_ms: float
    sim_ms: float
    chunk_ms: float
    network_scale: float
    indegree_scale: float
    inhibitory_weight_ratio: float
    external_drive_eta: float
    abort_rate_hz: float | None
    ne: int
    ni: int
    ce: int
    ci: int
    n_record: int
    weight_sample_size: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule", choices=("static", "additive", "morrison"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=_positive_int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--dt-ms", type=_positive_float, default=0.1)
    parser.add_argument("--delay-ms", type=_positive_float, default=1.5)
    parser.add_argument("--presim-ms", type=float, default=50.0)
    parser.add_argument("--sim-ms", type=_positive_float, default=250.0)
    parser.add_argument("--chunk-ms", type=_positive_float, default=50.0)
    parser.add_argument("--network-scale", type=_positive_float, default=1.0)
    parser.add_argument("--indegree-scale", type=_positive_float, default=1.0)
    parser.add_argument("--inhibitory-weight-ratio", type=_positive_float, default=5.0)
    parser.add_argument("--external-drive-eta", type=_positive_float, default=1.685)
    parser.add_argument("--abort-rate-hz", type=_positive_float)
    parser.add_argument("--record-neurons", type=_positive_int, default=1000)
    parser.add_argument("--weight-sample-size", type=_positive_int, default=100000)
    parser.add_argument("--additive-lambda", type=_positive_float, default=0.01)
    parser.add_argument("--additive-alpha", type=_positive_float, default=1.05)
    parser.add_argument("--morrison-lambda", type=_positive_float, default=0.1)
    parser.add_argument("--morrison-alpha", type=_positive_float, default=0.0513)
    parser.add_argument("--morrison-mu", type=float, default=0.4)
    args = parser.parse_args()
    if args.presim_ms < 0.0:
        parser.error("--presim-ms must be non-negative")
    if not 0.0 <= args.morrison_mu <= 1.0:
        parser.error("--morrison-mu must be between zero and one")
    for name in ("presim_ms", "sim_ms", "chunk_ms", "delay_ms"):
        steps = getattr(args, name) / args.dt_ms
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            parser.error(f"--{name.replace('_', '-')} must be an integer number of timesteps")
    return args


def conversion_factor(tau_m_ms: float, tau_syn_ms: float, capacitance_pf: float) -> float:
    ratio = tau_m_ms / tau_syn_ms
    inverse_delta = 1.0 / tau_syn_ms - 1.0 / tau_m_ms
    branch_value = lambertw(-np.exp(-1.0 / ratio) / ratio, k=-1).real
    rise_ms = (-branch_value - 1.0 / ratio) / inverse_delta
    peak = (
        np.e
        / (tau_syn_ms * capacitance_pf * inverse_delta)
        * (
            (np.exp(-rise_ms / tau_m_ms) - np.exp(-rise_ms / tau_syn_ms))
            / inverse_delta
            - rise_ms * np.exp(-rise_ms / tau_syn_ms)
        )
    )
    return 1.0 / peak


def sample_weights(connections: Any, requested: int) -> np.ndarray:
    count = len(connections)
    if count <= requested:
        selected = connections
    else:
        stride = max(1, count // requested)
        selected = connections[::stride]
    return np.asarray(selected.weight, dtype=np.float64)


def histogram_mode_count(histogram: np.ndarray) -> int:
    if histogram.size < 3 or not np.any(histogram):
        return 0
    smoothed = np.convolve(histogram.astype(np.float64), np.ones(3) / 3.0, mode="same")
    cutoff = 0.05 * float(smoothed.max())
    return int(
        np.count_nonzero(
            (smoothed[1:-1] > smoothed[:-2])
            & (smoothed[1:-1] >= smoothed[2:])
            & (smoothed[1:-1] >= cutoff)
        )
    )


def weight_stats(weights: np.ndarray, *, rule: str, wmax: float | None) -> dict[str, Any]:
    if weights.size == 0:
        raise RuntimeError("weight sample is empty")
    quantiles = np.quantile(weights, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    mean = float(np.mean(weights))
    std = float(np.std(weights))
    centered = weights - mean
    skewness_threshold = max(abs(mean), 1.0) * 1e-12
    skewness = float(np.mean(centered**3) / std**3) if std > skewness_threshold else 0.0
    if wmax is not None:
        edges = np.linspace(0.0, wmax, 41)
        low_fraction = float(np.mean(weights <= 0.1 * wmax))
        high_fraction = float(np.mean(weights >= 0.9 * wmax))
    else:
        upper = max(float(np.quantile(weights, 0.999)) * 1.05, mean * 2.0, 1.0)
        edges = np.linspace(0.0, upper, 41)
        low_fraction = float(np.mean(weights <= 0.1 * mean)) if mean > 0.0 else 1.0
        high_fraction = float(np.mean(weights >= 1.9 * mean)) if mean > 0.0 else 0.0
    histogram, edges = np.histogram(weights, bins=edges)
    return {
        "rule": rule,
        "sample_size": int(weights.size),
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "minimum": float(quantiles[0]),
        "p01": float(quantiles[1]),
        "p10": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "maximum": float(quantiles[6]),
        "low_boundary_fraction": low_fraction,
        "high_boundary_fraction": high_fraction,
        "boundary_fraction": low_fraction + high_fraction,
        "histogram_mode_count": histogram_mode_count(histogram),
        "histogram_counts": histogram.tolist(),
        "histogram_edges": edges.tolist(),
    }


def spike_stats(times: np.ndarray, senders: np.ndarray, duration_ms: float, n_record: int) -> dict[str, float]:
    rate_hz = float(times.size / (n_record * duration_ms) * 1000.0)
    if times.size == 0:
        return {"rate_hz": 0.0, "population_fano_3ms": 0.0, "mean_cv_isi": 0.0}
    bin_width_ms = 3.0
    bins = max(1, int(math.ceil(duration_ms / bin_width_ms)))
    counts, _ = np.histogram(times, bins=bins, range=(0.0, duration_ms))
    fano = float(np.var(counts) / np.mean(counts)) if np.mean(counts) > 0.0 else 0.0
    cvs: list[float] = []
    for sender in np.unique(senders):
        intervals = np.diff(times[senders == sender])
        if intervals.size >= 2 and np.mean(intervals) > 0.0:
            cvs.append(float(np.std(intervals) / np.mean(intervals)))
    return {
        "rate_hz": rate_hz,
        "population_fano_3ms": fano,
        "mean_cv_isi": float(np.mean(cvs)) if cvs else 0.0,
    }


def git_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository / "3rdparty" / "nest-simulator"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    program_started = time.perf_counter()

    import nest

    ne = max(2, round(BASE_NE * args.network_scale))
    ni = max(2, round(BASE_NI * args.network_scale))
    ce = max(1, round(BASE_CE * args.indegree_scale))
    ci = max(1, round(BASE_CI * args.indegree_scale))
    n_record = min(args.record_neurons, ne)
    workload = Workload(
        rule=args.rule,
        threads=args.threads,
        seed=args.seed,
        dt_ms=args.dt_ms,
        delay_ms=args.delay_ms,
        presim_ms=args.presim_ms,
        sim_ms=args.sim_ms,
        chunk_ms=args.chunk_ms,
        network_scale=args.network_scale,
        indegree_scale=args.indegree_scale,
        inhibitory_weight_ratio=args.inhibitory_weight_ratio,
        external_drive_eta=args.external_drive_eta,
        abort_rate_hz=args.abort_rate_hz,
        ne=ne,
        ni=ni,
        ce=ce,
        ci=ci,
        n_record=n_record,
        weight_sample_size=args.weight_sample_size,
    )

    nest.ResetKernel()
    nest.SetKernelStatus(
        {
            "resolution": args.dt_ms,
            "local_num_threads": args.threads,
            "rng_seed": (args.seed % (2**32 - 2)) + 1,
            "print_time": False,
        }
    )
    nest.set_verbosity("M_WARNING")

    model_params = {
        "E_L": 0.0,
        "C_m": 250.0,
        "tau_m": 10.0,
        "t_ref": 0.5,
        "V_th": 20.0,
        "V_reset": 0.0,
        "tau_syn_ex": TAU_SYN_MS,
        "tau_syn_in": TAU_SYN_MS,
        "tau_minus": 30.0,
        "V_m": 5.7,
    }
    construction_started = time.perf_counter()
    excitatory = nest.Create("iaf_psc_alpha", ne, model_params)
    inhibitory = nest.Create("iaf_psc_alpha", ni, model_params)
    excitatory.V_m = nest.random.normal(5.7, 7.2)
    inhibitory.V_m = nest.random.normal(5.7, 7.2)

    je_pa = conversion_factor(10.0, TAU_SYN_MS, 250.0) * 0.14
    nu_threshold = 20.0 / (ce * 10.0 / 250.0 * je_pa * np.e * TAU_SYN_MS)
    external_rate_hz = nu_threshold * args.external_drive_eta * ce * 1000.0
    stimulus = nest.Create("poisson_generator", params={"rate": external_rate_hz})
    recorder = nest.Create("spike_recorder", params={"record_to": "memory"})

    static_ex = {"synapse_model": "static_synapse_hpc", "weight": je_pa, "delay": args.delay_ms}
    static_in = {
        "synapse_model": "static_synapse_hpc",
        "weight": -args.inhibitory_weight_ratio * je_pa,
        "delay": args.delay_ms,
    }
    if args.rule == "morrison":
        plastic_model = "stdp_pl_synapse_hom_hpc"
        plastic_parameters = {
            "weight": je_pa,
            "delay": args.delay_ms,
            "alpha": args.morrison_alpha,
            "lambda": args.morrison_lambda,
            "mu": args.morrison_mu,
        }
        plastic_metadata = {
            "enabled": True,
            "model": plastic_model,
            "alpha": args.morrison_alpha,
            "lambda": args.morrison_lambda,
            "mu": args.morrison_mu,
            "Wmax": None,
        }
        wmax = None
        ee_synapse_spec: str | dict[str, float | str] = plastic_model
    elif args.rule == "additive":
        plastic_model = "stdp_synapse_hom_hpc"
        wmax = 2.0 * je_pa
        plastic_parameters = {
            "weight": je_pa,
            "delay": args.delay_ms,
            "alpha": args.additive_alpha,
            "lambda": args.additive_lambda,
            "mu_plus": 0.0,
            "mu_minus": 0.0,
            "Wmax": wmax,
        }
        plastic_metadata = {
            "enabled": True,
            "model": plastic_model,
            "alpha": args.additive_alpha,
            "lambda": args.additive_lambda,
            "mu_plus": 0.0,
            "mu_minus": 0.0,
            "Wmax": wmax,
        }
        ee_synapse_spec = plastic_model
    else:
        plastic_model = "static_synapse_hpc"
        wmax = None
        plastic_parameters = None
        plastic_metadata = {
            "enabled": False,
            "model": plastic_model,
            "weight": je_pa,
        }
        ee_synapse_spec = {
            "synapse_model": plastic_model,
            "weight": je_pa,
            "delay": args.delay_ms,
        }

    if plastic_parameters is not None:
        nest.SetDefaults(plastic_model, plastic_parameters)
    nest.Connect(stimulus, excitatory, "all_to_all", static_ex)
    nest.Connect(stimulus, inhibitory, "all_to_all", static_ex)
    nest.Connect(
        excitatory,
        excitatory,
        {"rule": "fixed_indegree", "indegree": ce, "allow_autapses": False, "allow_multapses": True},
        ee_synapse_spec,
    )
    nest.Connect(
        inhibitory,
        excitatory,
        {"rule": "fixed_indegree", "indegree": ci, "allow_autapses": False, "allow_multapses": True},
        static_in,
    )
    nest.Connect(
        excitatory,
        inhibitory,
        {"rule": "fixed_indegree", "indegree": ce, "allow_autapses": False, "allow_multapses": True},
        static_ex,
    )
    nest.Connect(
        inhibitory,
        inhibitory,
        {"rule": "fixed_indegree", "indegree": ci, "allow_autapses": False, "allow_multapses": True},
        static_in,
    )
    nest.Connect(excitatory[:n_record], recorder, "all_to_all", "static_synapse_hpc")
    ee_connections = nest.GetConnections(excitatory, excitatory, synapse_model=plastic_model)
    construction_wall = time.perf_counter() - construction_started
    initial_weights = sample_weights(ee_connections, args.weight_sample_size)
    initial_weight_stats = weight_stats(initial_weights, rule=args.rule, wmax=wmax)

    print(
        "CONFIG "
        f"rule={args.rule} nest={nest.__version__} threads={args.threads} seed={args.seed} "
        f"dt_ms={args.dt_ms} delay_ms={args.delay_ms} ne={ne} ni={ni} ce={ce} ci={ci} "
        f"g={args.inhibitory_weight_ratio} eta={args.external_drive_eta} "
        f"presim_ms={args.presim_ms} sim_ms={args.sim_ms} chunk_ms={args.chunk_ms} "
        f"ee_synapses={len(ee_connections)} "
        f"plastic_synapses={len(ee_connections) if args.rule != 'static' else 0}",
        flush=True,
    )
    print("PLASTICITY " + json.dumps(plastic_metadata, sort_keys=True), flush=True)

    periodic: list[dict[str, Any]] = []
    termination_reason = "requested_duration_completed"
    presim_wall = 0.0
    simulation_call_wall = 0.0
    dynamic_started = time.perf_counter()
    with nest.RunManager():
        if args.presim_ms > 0.0:
            started = time.perf_counter()
            nest.Run(args.presim_ms)
            presim_wall = time.perf_counter() - started
        presim_events = int(recorder.n_events)
        elapsed_ms = 0.0
        previous_events = presim_events
        while elapsed_ms < args.sim_ms - 1e-12:
            duration_ms = min(args.chunk_ms, args.sim_ms - elapsed_ms)
            started = time.perf_counter()
            nest.Run(duration_ms)
            call_wall = time.perf_counter() - started
            simulation_call_wall += call_wall
            elapsed_ms += duration_ms
            events = int(recorder.n_events)
            chunk_events = events - previous_events
            previous_events = events
            sampled = sample_weights(ee_connections, args.weight_sample_size)
            stats = weight_stats(sampled, rule=args.rule, wmax=wmax)
            row = {
                "elapsed_ms": elapsed_ms,
                "chunk_ms": duration_ms,
                "chunk_wall_seconds": call_wall,
                "chunk_seconds_per_step": call_wall / round(duration_ms / args.dt_ms),
                "chunk_rate_hz": chunk_events / (n_record * duration_ms) * 1000.0,
                "cumulative_rate_hz": (events - presim_events) / (n_record * elapsed_ms) * 1000.0,
                "weight_mean": stats["mean"],
                "weight_std": stats["std"],
                "weight_boundary_fraction": stats["boundary_fraction"],
            }
            periodic.append(row)
            print("PERIODIC " + json.dumps(row, sort_keys=True), flush=True)
            if args.abort_rate_hz is not None and row["chunk_rate_hz"] >= args.abort_rate_hz:
                termination_reason = (
                    f"chunk_rate_hz={row['chunk_rate_hz']} reached "
                    f"abort_rate_hz={args.abort_rate_hz}"
                )
                print("EARLY_STOP " + termination_reason, flush=True)
                break
    dynamic_phase_wall = time.perf_counter() - dynamic_started

    final_weights = sample_weights(ee_connections, args.weight_sample_size)
    final_weight_stats = weight_stats(final_weights, rule=args.rule, wmax=wmax)
    events = recorder.events
    event_times = np.asarray(events["times"], dtype=np.float64) - args.presim_ms
    event_senders = np.asarray(events["senders"], dtype=np.int64)
    keep = (event_times >= 0.0) & (event_times <= elapsed_ms + args.dt_ms * 0.5)
    final_spike_stats = spike_stats(event_times[keep], event_senders[keep], elapsed_ms, n_record)

    presim_steps = round(args.presim_ms / args.dt_ms)
    requested_simulation_steps = round(args.sim_ms / args.dt_ms)
    simulation_steps = round(elapsed_ms / args.dt_ms)
    total_dynamic_steps = presim_steps + simulation_steps
    total_simulate_call_wall = presim_wall + simulation_call_wall
    whole_program_wall = time.perf_counter() - program_started
    kernel_status = nest.GetKernelStatus()
    timing = {
        "construction_wall_seconds": construction_wall,
        "presimulation_wall_seconds": presim_wall,
        "simulation_wall_seconds": simulation_call_wall,
        "combined_simulate_call_wall_seconds": total_simulate_call_wall,
        "dynamic_phase_wall_seconds": dynamic_phase_wall,
        "whole_program_wall_seconds": whole_program_wall,
        "presimulation_steps": presim_steps,
        "simulation_steps": simulation_steps,
        "requested_simulation_steps": requested_simulation_steps,
        "actual_simulation_ms": elapsed_ms,
        "requested_simulation_ms": args.sim_ms,
        "total_dynamic_steps": total_dynamic_steps,
        "simulation_seconds_per_step": simulation_call_wall / simulation_steps,
        "combined_simulate_call_seconds_per_step": total_simulate_call_wall / total_dynamic_steps,
        "dynamic_phase_seconds_per_step": dynamic_phase_wall / total_dynamic_steps,
        "nest_time_simulate_seconds": float(kernel_status["time_simulate"]),
        "nest_time_construction_create_seconds": float(kernel_status["time_construction_create"]),
        "nest_time_construction_connect_seconds": float(kernel_status["time_construction_connect"]),
    }
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "nest_version": nest.__version__,
        "nest_revision": git_revision(repository),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "workload": asdict(workload),
        "plasticity": plastic_metadata,
        "je_pa": je_pa,
        "inhibitory_weight_pa": -args.inhibitory_weight_ratio * je_pa,
        "inhibitory_to_excitatory_weight_ratio": -args.inhibitory_weight_ratio,
        "external_drive_eta": args.external_drive_eta,
        "external_rate_hz": external_rate_hz,
        "recurrent_synapses_expected": (ne + ni) * (ce + ci),
        "ee_synapses": len(ee_connections),
        "plastic_synapses": len(ee_connections) if args.rule != "static" else 0,
    }
    results = {
        "manifest": manifest,
        "timing": timing,
        "initial_weight_stats": initial_weight_stats,
        "final_weight_stats": final_weight_stats,
        "spike_stats": final_spike_stats,
        "periodic": periodic,
        "termination": {
            "completed_requested_duration": termination_reason == "requested_duration_completed",
            "reason": termination_reason,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output / "weight_samples.npz",
        initial=initial_weights,
        final=final_weights,
    )
    print("RESULT " + json.dumps({"timing": timing, "spike_stats": final_spike_stats, "final_weight_stats": final_weight_stats}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
