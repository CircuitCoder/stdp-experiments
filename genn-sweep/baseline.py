#!/usr/bin/env python3
"""Prepare immutable inputs and run the 23-case cross-GPU baseline."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "reimpl"))
sys.path.insert(0, str(ROOT / "brunel"))

import run as sweep
from baseline_cases import mnist_cases, project_weights
from backends.genn_backend import GeNNNetwork
from zd3.constants import MODEL
from zd3.io import load_checkpoint, load_mnist, save_checkpoint, sha256_file
from zd3.variants import connectivity_mask, validate_checkpoint_topology

CASES = mnist_cases()
ALL_CASES = tuple(CASES) + tuple(sweep.BRUNEL_CASES)


def write_json(path, value):
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def command_output(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    return {"returncode": result.returncode, "output": result.stdout.strip()}


def provenance():
    import pygenn
    sources = [Path(__file__), ROOT / "genn-sweep/baseline_cases.py",
               ROOT / "genn-sweep/run.py", ROOT / "reimpl/backends/genn_backend.py",
               ROOT / "reimpl/zd3/variants.py", ROOT / "reimpl/zd3/constants.py",
               ROOT / "reimpl/zd3/io.py", ROOT / "brunel/ports/genn_port.py",
               ROOT / "brunel/ports/common.py"]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
        "host": platform.uname()._asdict(), "python": sys.version,
        "numpy": np.__version__, "genn": pygenn.__version__,
        "source_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sources},
        "environment": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "CUDA_PATH", "CUDAHOSTCXX", "NVCC_PREPEND_FLAGS",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PATH", "LD_LIBRARY_PATH")},
        "gpu": command_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,"
            "compute_cap,memory.total,power.limit,clocks.sm,clocks.mem,mig.mode.current",
            "--format=csv,noheader"]),
        "gpu_processes": command_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                         "--format=csv,noheader"]),
        "cpu": command_output(["lscpu"]),
        "nvcc": command_output([str(Path(os.environ["CUDA_PATH"]) / "bin/nvcc"), "--version"]),
    }


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    dense_one = ROOT / "genn-sweep/checkpoints/mnist_one_trace_dense_010000.npz"
    dense_three = ROOT / "genn-sweep/checkpoints/mnist_triplet_dense_010000.npz"
    sparse_one = ROOT / "genn-sweep/checkpoints/mnist_one_trace_sparse_0125_010000.npz"
    dense_two = ROOT / "genn-sweep/checkpoints/mnist_two_trace_dense_010000.npz"
    one, three = load_checkpoint(dense_one), load_checkpoint(dense_three)
    two = load_checkpoint(dense_two)
    inputs = {str(p.relative_to(ROOT)): sha256_file(p)
              for p in (dense_one, dense_two, dense_three, sparse_one)}
    manifest = {"schema": "genn-baseline-inputs-v1", "parent_sha256": inputs,
                "source_sha256": {str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
                                  "genn-sweep/baseline_cases.py": sha256_file(ROOT / "genn-sweep/baseline_cases.py")},
                "cases": {}}
    for name, variant in CASES.items():
        trace = int(name.split("_")[1][0])
        mask = connectivity_mask(variant)
        if trace == 2:
            weights, theta = two.weights.copy(), two.theta_mv.copy()
            parent = [str(dense_two.relative_to(ROOT))]
        else:
            checkpoint = one if trace == 1 else three
            weights, theta = checkpoint.weights.copy(), checkpoint.theta_mv.copy()
            parent = [str((dense_one if trace == 1 else dense_three).relative_to(ROOT))]
        kind = "trained-dense-checkpoint"
        if name == "mnist_1trace_bernoulli_0125":
            checkpoint = load_checkpoint(sparse_one)
            weights, theta = checkpoint.weights.copy(), checkpoint.theta_mv.copy()
            parent, kind = [str(sparse_one.relative_to(ROOT))], "trained-sparse-checkpoint"
        elif variant.topology != "dense":
            weights = project_weights(weights, mask, variant.weight_max)
            kind = "derived-dense-checkpoint-mask-bounded-column-normalization"
        validate_checkpoint_topology(weights, variant)
        path = args.output / f"{name}.npz"
        metadata = {"variant": variant.as_dict(), "starting_state": kind, "parents": parent,
                    "parent_sha256": {p: inputs[p] for p in parent},
                    "runtime_state": "reset; branched continuation", "theta_unit": "mV"}
        save_checkpoint(path, weights=weights, theta_mv=theta, accepted_samples=10000,
                        manifest=metadata)
        manifest["cases"][name] = {**metadata, "checkpoint": path.name,
            "checkpoint_sha256": sha256_file(path), "structural_synapses": int(mask.sum()),
            "weights_sha256_float64_le_c_order": hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest(),
            "theta_sha256_float64_le": hashlib.sha256(theta.astype("<f8").tobytes()).hexdigest(),
            "mask_sha256_packbits_big_c_order": hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()}
    manifest["dataset_sha256"] = {f"data/mnist/{name}": sha256_file(ROOT / "data/mnist" / name)
        for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")}
    write_json(args.output / "manifest.json", manifest)
    print(f"Prepared {len(CASES)} MNIST checkpoints in {args.output}", flush=True)


def mnist_run(args, name, entry):
    variant = CASES[name]
    path = args.inputs / entry["checkpoint"]
    if sha256_file(path) != entry["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    checkpoint = load_checkpoint(path)
    if checkpoint.manifest["variant"] != variant.as_dict():
        raise ValueError(f"workload definition changed: {name}")
    mask = validate_checkpoint_topology(checkpoint.weights, variant)
    sparse = variant.topology != "dense"
    build = args.output / "builds" / name
    reuse = args.reuse_build_root / name if args.reuse_build_root else None
    network = GeNNNetwork(weights=checkpoint.weights.copy(), theta_mv=checkpoint.theta_mv.copy(),
        plasticity=True, inhibition=MODEL.train_inhibition, seed=args.seed,
        backend=getattr(args, "backend", "cuda"), build_path=build, variant=variant, structural_mask=mask,
        precision="float", parallelism="presynaptic" if sparse else "postsynaptic",
        num_threads_per_spike=32 if sparse else 1, timing_enabled=False, reuse_build=reuse)
    accepted, attempts = 0, 0
    snapshots, stimuli = [], []
    diagnostic = args.mode == "diagnostic"
    data = load_mnist(args.data_path, "train")
    try:
        if getattr(args, "on_ready", None):
            args.on_ready()
        started = time.perf_counter()
        while accepted < args.samples:
            intensity = MODEL.initial_intensity
            while True:
                network.normalize(validate=diagnostic)
                network.set_image(data.images[checkpoint.accepted_samples + accepted], intensity)
                counts = network.run_stimulus()
                total = int(counts.sum())
                if diagnostic:
                    network.validate_runtime(counts, 5000)
                    if (accepted + 1) % 25 == 0:
                        snapshots.append({"accepted_before_attempt": accepted,
                                          **network.runtime_diagnostics()})
                if total >= 5000:
                    raise RuntimeError(f"runaway activity: {total} E spikes")
                retry = total < MODEL.minimum_exc_spikes
                network.run_rest(synchronize=False)
                attempts += 1
                if retry:
                    intensity += 1.0
                    if intensity > 20.0:
                        raise RuntimeError("retry intensity exceeded 20")
                else:
                    if diagnostic:
                        stimuli.append({"spikes": total, "active_neurons": int(np.count_nonzero(counts)),
                                        "intensity": intensity})
                    accepted += 1
                    break
        spike_count = network.total_spike_count()
        wall = time.perf_counter() - started
        if getattr(args, "on_finished", None):
            args.on_finished()
        counters = network.event_counters()
        result = {"wall_seconds": wall, "simulation_steps": attempts * 1000,
            "us_per_step": wall * 1e6 / (attempts * 1000), "accepted_samples": accepted,
            "attempts": attempts, "total_spikes": spike_count, "event_counters": counters}
        if diagnostic:
            weights, theta = network.weights(), network.theta_mv()
            if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(theta)):
                raise RuntimeError("non-finite final weights or thresholds")
            if weights.min() < 0 or weights.max() > variant.weight_max * 1.02:
                raise RuntimeError("final weight bound exceeded")
            exc_counts = network._pulled(network.exc, "spikeCount").astype(np.int64)
            result["diagnostic"] = {"stimuli": stimuli, "stimulus_snapshots": snapshots,
                "final_rest_snapshot": network.runtime_diagnostics(),
                "exc_spikes_per_neuron": exc_counts.tolist(),
                "active_exc_neurons_over_run": int(np.count_nonzero(exc_counts)),
                "maximum_exc_spike_share": float(exc_counts.max() / exc_counts.sum()),
                "weight_min": float(weights.min()), "weight_max": float(weights.max()),
                "weight_mean": float(weights.mean()),
                "upper_bound_fraction_present": float(np.mean(weights[mask] >= variant.weight_max)),
                "column_sum_min": float(weights.sum(axis=0).min()),
                "column_sum_max": float(weights.sum(axis=0).max()),
                "column_sum_mean": float(weights.sum(axis=0).mean()),
                "theta_min_mv": float(theta.min()), "theta_max_mv": float(theta.max()),
                "theta_mean_mv": float(theta.mean()), "runtime_guards_passed": True}
        return result
    finally:
        network.close()


def brunel_run(args, name):
    if args.mode == "diagnostic":
        output = args.output / name
        rule = sweep.BRUNEL_CASES[name]
        command = [sys.executable, str(ROOT / "brunel/run_genn.py"),
            "--backend", "cuda", "--precision", "float", "--rule", rule,
            "--seed", str(args.seed), "--state-seed", str(args.seed),
            "--indegree-scale", "0.05", "--delay-ms", "0",
            "--recurrent-delivery-scale", "4.47213595499958",
            "--external-rate-scale", "0.47" if rule == "additive" else "0.32",
            "--presim-ms", "100", "--sim-ms", "1000", "--chunk-ms", "100",
            "--record-neurons", "9000", "--weight-sample-size", "100000",
            "--no-genn-timing", "--no-profile-accounting", "--output", str(output)]
        subprocess.run(command, check=True)
        result = json.loads((output / "results.json").read_text())
        if not result["termination"]["completed_requested_duration"]:
            raise RuntimeError(f"Brunel diagnostic did not complete: {name}")
        return result
    config = sweep.build_parser().parse_args(["--work-dir", str(args.output),
        "--precision", "float", "--brunel-seed", str(args.seed)])
    config.reuse_build_root = args.reuse_build_root
    result = sweep.run_brunel(config, name, "float")
    return {**asdict(result), "total_spikes": result.spike_count,
            "us_per_step": result.wall_seconds * 1e6 / result.simulation_steps}


def run(args):
    current_provenance = provenance()
    input_hash = sha256_file(args.inputs / "manifest.json")
    if args.reuse_build_root:
        origin = json.loads((args.reuse_build_root.parent / "manifest.json").read_text())
        if (origin["mode"] != "timing" or origin["seed"] != args.seed
                or origin["input_manifest_sha256"] != input_hash
                or origin["source_sha256"] != current_provenance["source_sha256"]):
            raise ValueError("reusable build does not match the source, input bundle and seed")
        for key in ("CUDA_PATH", "CUDAHOSTCXX", "NVCC_PREPEND_FLAGS", "CUDA_VISIBLE_DEVICES"):
            if origin["environment"][key] != current_provenance["environment"][key]:
                raise ValueError(f"reusable build environment changed: {key}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "logs").mkdir()
    (args.output / "builds").mkdir()
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    for relative, expected in manifest["dataset_sha256"].items():
        if sha256_file(args.data_path / Path(relative).name) != expected:
            raise ValueError(f"dataset hash mismatch: {relative}")
    write_json(args.output / "manifest.json", {**current_provenance,
        "input_manifest_sha256": input_hash,
        "seed": args.seed, "mode": args.mode, "samples": args.samples,
        "cases": args.cases, "reuse_build_root": str(args.reuse_build_root) if args.reuse_build_root else None})
    for name in args.cases:
        with sweep.redirect_process_output(args.output / "logs" / f"{name}.log"):
            result = mnist_run(args, name, manifest["cases"][name]) if name in CASES else brunel_run(args, name)
        write_json(args.output / f"{name}.json", result)
        value = result.get("us_per_step")
        print(f"COMPLETE case={name} mode={args.mode} us_per_step={value}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--inputs", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--mode", choices=("diagnostic", "timing"), required=True)
    run_parser.add_argument("--cases", nargs="+", choices=ALL_CASES, default=ALL_CASES)
    run_parser.add_argument("--seed", type=int, default=20260724)
    run_parser.add_argument("--samples", type=int, default=100)
    run_parser.add_argument("--data-path", type=Path, default=ROOT / "data/mnist")
    run_parser.add_argument("--reuse-build-root", type=Path)
    args = parser.parse_args()
    if args.action == "run" and (args.seed <= 0 or not 0 < args.samples <= 50000):
        parser.error("seed must be positive; samples must be in [1, 50000]")
    (prepare if args.action == "prepare" else run)(args)


if __name__ == "__main__":
    main()
