#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from backends.brian2_backend import Brian2Model
from zd3.constants import MODEL
from zd3.io import load_mnist, load_reference_triplets, sha256_file
from zd3.variants import VARIANTS, get_variant, prepare_initial_weights


def main() -> int:
    root = Path(__file__).resolve().parent
    repository = root.parent
    parser = argparse.ArgumentParser(
        description="Generate or run one Brian2CUDA ZD3 presentation attempt."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="triplet-dense")
    parser.add_argument(
        "--initial-weights",
        type=Path,
        default=None,
    )
    parser.add_argument("--data-path", type=Path, default=repository / "data" / "mnist")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--cuda-path", default=os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    parser.add_argument("--cuda-runtime-version", type=float, default=13.0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--compute-capability", type=float, default=8.6)
    parser.add_argument(
        "--compile-jobs",
        type=int,
        default=1,
        help="maximum parallel CUDA compilation jobs (default: 1 to limit host memory)",
    )
    args = parser.parse_args()
    if args.compile_jobs <= 0:
        raise SystemExit("--compile-jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)

    import brian2 as b
    import brian2cuda  # noqa: F401

    b.prefs.devices.cpp_standalone.extra_make_args_unix = [f"-j{args.compile_jobs}"]
    cuda = b.prefs.devices.cuda_standalone.cuda_backend
    cuda.detect_cuda = False
    cuda.detect_gpus = False
    cuda.cuda_path = args.cuda_path
    cuda.gpu_id = args.gpu_id
    cuda.compute_capability = args.compute_capability
    cuda.cuda_runtime_version = args.cuda_runtime_version
    cuda.extra_compile_args_nvcc = [
        *cuda.extra_compile_args_nvcc,
        f"-I{args.cuda_path}/include",
    ]
    b.prefs.codegen.cpp.extra_link_args += [f"-I{args.cuda_path}/include"]
    b.prefs.codegen.cpp.headers += ["<chrono>", "<fstream>"]
    b.set_device("cuda_standalone", build_on_run=False)
    variant = get_variant(args.variant)
    if args.initial_weights is None:
        args.initial_weights = (
            repository
            / "ref"
            / "zero_delay_one_trace_v1"
            / "full_nupost0005_30000"
            / "random"
            / "XeAe.npy"
            if variant.learning_rule == "one-trace-power"
            else repository
            / "ref"
            / "zero_delay_midpoint_v1"
            / "inhib150_full"
            / "random"
            / "XeAe.npy"
        )
    weights, structural_mask = prepare_initial_weights(
        load_reference_triplets(
            args.initial_weights, MODEL.n_input, MODEL.n_exc, dtype=np.float64
        ),
        variant,
    )
    data = load_mnist(args.data_path, "train")
    model = Brian2Model(
        weights=weights,
        theta_mv=np.full(MODEL.n_exc, MODEL.theta_initial_mv),
        plasticity=True,
        inhibition=MODEL.train_inhibition,
        seed=args.seed,
        codegen_target="cython",
        standalone_codegen=True,
        variant=variant,
        structural_mask=structural_mask,
    )
    recorded_input_spikes = b.SpikeMonitor(model.inputs, name="recorded_input_spikes")
    recorded_exc_spikes = b.SpikeMonitor(model.exc, name="recorded_exc_spikes")
    recorded_inh_spikes = b.SpikeMonitor(model.inh, name="recorded_inh_spikes")
    model.network.add(
        recorded_input_spikes, recorded_exc_spikes, recorded_inh_spikes
    )
    cuda_device = b.get_device()
    if args.run:
        cuda_device.insert_code(
            "before_start",
            """
            std::chrono::high_resolution_clock::time_point _zd3_start, _zd3_stop;
            std::ofstream _zd3_timing;
            _zd3_timing.open("results/network_run_us.txt");
            """,
        )
        cuda_device.insert_code(
            "before_network_run",
            "cudaDeviceSynchronize(); _zd3_start = std::chrono::high_resolution_clock::now();",
        )
        cuda_device.insert_code(
            "after_network_run",
            """
            cudaDeviceSynchronize();
            _zd3_stop = std::chrono::high_resolution_clock::now();
            _zd3_timing << std::chrono::duration_cast<std::chrono::microseconds>(
                _zd3_stop - _zd3_start).count() << std::endl;
            """,
        )
        cuda_device.insert_code("after_end", "_zd3_timing.close();")
    model.set_image(data.images[0], MODEL.initial_intensity)
    model.network.run(MODEL.stimulus_ms * b.ms, namespace={})
    model.set_zero_input()
    model.network.run(MODEL.rest_ms * b.ms, namespace={})
    code_directory = args.output / "cuda_standalone"
    # Brian2CUDA asks for nvcc while rendering its makefile even when
    # compile=False. Source generation itself does not need a toolkit.
    if args.run:
        build_started = time.perf_counter()
        cuda_device.build(
            directory=str(code_directory), compile=True, run=True, clean=True
        )
        build_and_run_wall = time.perf_counter() - build_started
    else:
        cuda_device.generate_makefile = lambda *unused_args, **unused_kwargs: None
        cuda_device.build(
            directory=str(code_directory), compile=False, run=False, clean=True
        )
    manifest = {
        "backend": "brian2cuda-codegen",
        "brian2_version": getattr(b, "__version__", "unknown"),
        "brian2cuda_version": getattr(brian2cuda, "__version__", "unknown"),
        "initial_weights": str(args.initial_weights),
        "initial_weights_sha256": sha256_file(args.initial_weights),
        "model": MODEL.as_dict(),
        "variant": variant.as_dict(),
        "structural_synapses": int(structural_mask.sum()),
        "actual_connection_rate": float(structural_mask.mean()),
        "scope": (
            "one-attempt CUDA runtime validation; no adaptive retry or accuracy claim"
            if args.run
            else "one-attempt source-generation validation; not a training benchmark"
        ),
        "cuda_compile": args.run,
        "cuda_run": args.run,
        "compute_capability": args.compute_capability,
        "cuda_runtime_version": args.cuda_runtime_version,
        "cuda_path": args.cuda_path,
        "compile_jobs": args.compile_jobs,
    }
    with (args.output / "manifest.json").open("x", encoding="ascii") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    if args.run:
        run_walls = [
            float(value) / 1.0e6
            for value in (code_directory / "results" / "network_run_us.txt").read_text().split()
        ]
        if len(run_walls) != 2:
            raise RuntimeError(f"expected two network timings, found {len(run_walls)}")
        spike_times_ms = np.asarray(recorded_exc_spikes.t / b.ms, dtype=np.float64)
        stimulus_spikes = int(np.count_nonzero(spike_times_ms < MODEL.stimulus_ms))
        input_spikes = int(len(recorded_input_spikes.i))
        excitatory_spikes = int(len(recorded_exc_spikes.i))
        inhibitory_spikes = int(len(recorded_inh_spikes.i))
        outdegree = structural_mask.sum(axis=1, dtype=np.int64)
        indegree = structural_mask.sum(axis=0, dtype=np.int64)
        input_ids = np.asarray(recorded_input_spikes.i, dtype=np.int64)
        exc_ids = np.asarray(recorded_exc_spikes.i, dtype=np.int64)
        pre_updates = int(outdegree[input_ids].sum())
        post_updates = int(indegree[exc_ids].sum())
        simulation_wall = sum(run_walls)
        result = {
            "stimulus_spikes": stimulus_spikes,
            "accepted_by_minimum_spike_rule": stimulus_spikes >= MODEL.minimum_exc_spikes,
            "stimulus_wall_seconds": run_walls[0],
            "rest_wall_seconds": run_walls[1],
            "simulation_wall_seconds": simulation_wall,
            "simulated_ticks": MODEL.attempt_ticks,
            "seconds_per_timestep_cycle": simulation_wall / MODEL.attempt_ticks,
            "input_spikes": input_spikes,
            "excitatory_spikes": excitatory_spikes,
            "inhibitory_spikes": inhibitory_spikes,
            "total_firing_count": input_spikes + excitatory_spikes + inhibitory_spikes,
            "feedforward_pre_synapse_updates": pre_updates,
            "feedforward_post_synapse_updates": post_updates,
            "average_firing_count_per_timestep": (
                input_spikes + excitatory_spikes + inhibitory_spikes
            )
            / MODEL.attempt_ticks,
            "average_pre_synapse_updates_per_timestep": pre_updates
            / MODEL.attempt_ticks,
            "average_post_synapse_updates_per_timestep": post_updates
            / MODEL.attempt_ticks,
            "seconds_per_pre_synapse_update": (
                simulation_wall / pre_updates if pre_updates else None
            ),
            "seconds_per_post_synapse_update": (
                simulation_wall / post_updates if post_updates else None
            ),
            "compile_wall_seconds": cuda_device.timers["compile"]["make"],
            "binary_wall_seconds": cuda_device.timers["run_binary"],
            "build_and_run_wall_seconds": build_and_run_wall,
            "weight_mean": float(np.mean(model.weights())),
            "weight_min": float(np.min(model.weights())),
            "weight_max": float(np.max(model.weights())),
            "theta_mean_mv": float(np.mean(model.theta_mv())),
            "variant": variant.name,
            "structural_synapses": int(structural_mask.sum()),
        }
        with (args.output / "results.json").open("x", encoding="ascii") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print("DIAGNOSTICS backend=brian2cuda-runtime " + " ".join(f"{k}={v}" for k, v in result.items()), flush=True)
        print(f"RUN_COMPLETE backend=brian2cuda-runtime attempts=1 accepted={int(result['accepted_by_minimum_spike_rule'])}", flush=True)
    else:
        print(
            f"CODEGEN_COMPLETE backend=brian2cuda path={code_directory} "
            "compile=false run=false",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
