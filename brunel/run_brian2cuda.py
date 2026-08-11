#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from ports.brian2_port import Brian2Brunel
from ports.common import (
    DT_MS,
    JE_PA,
    base_manifest,
    create_output,
    make_model,
    spike_stats,
    stdp_post_path_delay_ms,
    STDP_TIE_MODES,
    weight_stats,
)


def _configure_cuda(b, args: argparse.Namespace) -> None:
    import brian2cuda  # noqa: F401

    b.prefs.devices.cpp_standalone.extra_make_args_unix = [f"-j{args.compile_jobs}"]
    cuda = b.prefs.devices.cuda_standalone.cuda_backend
    cuda.detect_cuda = False
    cuda.detect_gpus = False
    cuda.cuda_path = args.cuda_path
    cuda.cuda_runtime_version = args.cuda_runtime_version
    cuda.gpu_id = args.gpu_id
    cuda.compute_capability = args.compute_capability
    b.prefs.codegen.cpp.headers += ["<chrono>", "<fstream>"]
    b.set_device("cuda_standalone", build_on_run=False)


def _install_run_timers(b) -> None:
    device = b.get_device()
    device.insert_code(
        "before_start",
        """
        std::chrono::high_resolution_clock::time_point _stdp_run_start, _stdp_run_stop;
        std::ofstream _stdp_timing_file;
        _stdp_timing_file.open("results/network_run_us.txt");
        """,
    )
    device.insert_code(
        "before_network_run",
        """
        cudaDeviceSynchronize();
        _stdp_run_start = std::chrono::high_resolution_clock::now();
        """,
    )
    device.insert_code(
        "after_network_run",
        """
        cudaDeviceSynchronize();
        _stdp_run_stop = std::chrono::high_resolution_clock::now();
        _stdp_timing_file
            << std::chrono::duration_cast<std::chrono::microseconds>(
                   _stdp_run_stop - _stdp_run_start).count()
            << std::endl;
        """,
    )
    device.insert_code("after_end", "_stdp_timing_file.close();")


def run(args: argparse.Namespace) -> int:
    import brian2 as b
    import brian2cuda

    _configure_cuda(b, args)
    model = make_model(args.rule, args.network_scale, args.indegree_scale)
    manifest = base_manifest("brian2cuda-runtime", model)
    manifest.update(
        {
            "brian2_version": getattr(b, "__version__", "unknown"),
            "brian2cuda_version": getattr(brian2cuda, "__version__", "unknown"),
            "seed": args.seed,
            "state_seed": args.state_seed if args.state_seed is not None else args.seed,
            "presim_ms": args.presim_ms,
            "sim_ms": args.sim_ms,
            "chunk_ms": args.chunk_ms,
            "cuda_path": args.cuda_path,
            "cuda_runtime_version": args.cuda_runtime_version,
            "gpu_id": args.gpu_id,
            "compute_capability": args.compute_capability,
            "compile_jobs": args.compile_jobs,
            "connectivity": "fixed indegree with replacement; recurrent autapses excluded",
            "timing_scope": "CUDA-synchronized wall time around queued Network.run calls",
            "periodic_weight_scope": (
                "unavailable during one-shot standalone execution; initial and final samples only"
            ),
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
        codegen_target="cython",
        connectivity_target_chunk=args.connectivity_target_chunk,
        stdp_timing=args.stdp_timing,
        stdp_tie_order=args.stdp_tie_order,
    )
    construction_wall = time.perf_counter() - construction_started
    initial_weights = np.full(
        min(args.weight_sample_size, model.plastic_synapses), JE_PA, dtype=np.float64
    )
    initial_stats = weight_stats(initial_weights, model.rule)
    _install_run_timers(b)

    print(
        f"CONFIG backend=brian2cuda-runtime rule={args.rule} dt_ms={DT_MS} "
        f"ne={model.ne} ni={model.ni} ce={model.ce} ci={model.ci} "
        f"recurrent_synapses={model.recurrent_synapses} gpu_id={args.gpu_id} "
        f"compute_capability={args.compute_capability}",
        flush=True,
    )
    if args.presim_ms:
        network.network.run(args.presim_ms * b.ms, namespace={})
    chunk_durations = []
    elapsed_ms = 0.0
    while elapsed_ms < args.sim_ms - 1e-12:
        duration_ms = min(args.chunk_ms, args.sim_ms - elapsed_ms)
        chunk_durations.append(duration_ms)
        elapsed_ms += duration_ms
    network.network.run(args.sim_ms * b.ms, namespace={})

    code_directory = args.output / "cuda_standalone"
    build_started = time.perf_counter()
    b.get_device().build(
        directory=str(code_directory),
        compile=True,
        run=True,
        clean=True,
        with_output=True,
    )
    build_and_run_wall = time.perf_counter() - build_started
    device = b.get_device()
    compile_wall = device.timers["compile"]["make"]
    binary_wall = device.timers["run_binary"]

    timing_path = code_directory / "results" / "network_run_us.txt"
    run_walls = [float(value) / 1.0e6 for value in timing_path.read_text().split()]
    expected_runs = 1 + int(args.presim_ms > 0.0)
    if len(run_walls) != expected_runs:
        raise RuntimeError(f"expected {expected_runs} network timings, found {len(run_walls)}")
    if args.presim_ms:
        presim_wall = run_walls[0]
        simulation_wall = run_walls[1]
    else:
        presim_wall = 0.0
        simulation_wall = run_walls[0]

    times = np.asarray(network.spikes.t / b.ms, dtype=np.float64) - args.presim_ms
    senders = np.asarray(network.spikes.i, dtype=np.int64)
    keep = (times >= 0.0) & (times <= elapsed_ms + DT_MS * 0.5)
    times = times[keep]
    senders = senders[keep]
    periodic = []
    chunk_start = 0.0
    cumulative_events = 0
    for duration_ms in chunk_durations:
        chunk_stop = chunk_start + duration_ms
        chunk_events = int(
            np.count_nonzero((times > chunk_start) & (times <= chunk_stop + DT_MS * 0.5))
        )
        cumulative_events += chunk_events
        row = {
            "elapsed_ms": chunk_stop,
            "chunk_ms": duration_ms,
            "chunk_wall_seconds": None,
            "chunk_seconds_per_step": None,
            "chunk_rate_hz": chunk_events / (network.n_record * duration_ms) * 1000.0,
            "cumulative_rate_hz": cumulative_events
            / (network.n_record * chunk_stop)
            * 1000.0,
            "weight_stats": "initial-and-final-only",
        }
        periodic.append(row)
        print("PERIODIC " + json.dumps(row, sort_keys=True), flush=True)
        chunk_start = chunk_stop

    final_weights = network.sample_weights(args.weight_sample_size)
    final_stats = weight_stats(final_weights, model.rule)
    final_spikes = spike_stats(times, senders, elapsed_ms, network.n_record)
    simulation_steps = round(elapsed_ms / DT_MS)
    presim_steps = round(args.presim_ms / DT_MS)
    timing = {
        "construction_wall_seconds": construction_wall,
        "compile_wall_seconds": compile_wall,
        "binary_wall_seconds": binary_wall,
        "build_and_run_wall_seconds": build_and_run_wall,
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
            "completed_requested_duration": True,
            "reason": "requested_duration_completed",
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
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brian2CUDA Brunel STDP runtime")
    parser.add_argument("--rule", choices=("additive", "morrison"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--state-seed", type=int)
    parser.add_argument("--network-scale", type=float, default=1.0)
    parser.add_argument("--indegree-scale", type=float, default=1.0)
    parser.add_argument("--presim-ms", type=float, default=100.0)
    parser.add_argument("--sim-ms", type=float, default=1000.0)
    parser.add_argument("--chunk-ms", type=float, default=100.0)
    parser.add_argument("--record-neurons", type=int, default=1000)
    parser.add_argument("--weight-sample-size", type=int, default=100000)
    parser.add_argument("--connectivity-target-chunk", type=int, default=32)
    parser.add_argument("--cuda-path", default=os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    parser.add_argument("--cuda-runtime-version", type=float, default=12.8)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--compute-capability", type=float, default=8.6)
    parser.add_argument(
        "--compile-jobs",
        type=int,
        default=1,
        help="maximum parallel CUDA compilation jobs (default: 1 to limit host memory)",
    )
    parser.add_argument(
        "--stdp-timing", choices=("arrival", "nest_dendritic"), default="nest_dendritic"
    )
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
    if args.compile_jobs <= 0:
        raise SystemExit("--compile-jobs must be positive")
    for name in ("presim_ms", "sim_ms", "chunk_ms"):
        steps = getattr(args, name) / DT_MS
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise SystemExit(f"--{name.replace('_', '-')} must be an integer number of timesteps")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
