#!/usr/bin/env python3
"""Measure dense MNIST training with exactly five nvidia-smi power samples.

No checkpoints are written. Uses the baseline's model, inputs and scheduling;
the measured interval follows a 10-second training warmup and lasts ~25 seconds.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "genn-sweep"))
import baseline
import numpy as np
from zd3.variants import validate_normalized_weight_bound

CASES = tuple(f"mnist_{n}trace_dense" for n in (1, 2, 3))
INTERVAL_SECONDS = 5.0
SAMPLE_COUNT = 5
WARMUP_SECONDS = 10.0


class PowerSampler(threading.Thread):
    def __init__(self, gpu_uuid, start):
        super().__init__(daemon=True)
        self.gpu_uuid, self.start_time = gpu_uuid, start
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.rows = []
        self.error = None
        self.command = ["nvidia-smi", "-i", gpu_uuid,
                        "--query-gpu=timestamp,uuid,power.draw,utilization.gpu,clocks.sm,temperature.gpu",
                        "--format=csv,noheader,nounits"]

    def run(self):
        try:
            for index in range(1, SAMPLE_COUNT + 1):
                due = self.start_time + index * INTERVAL_SECONDS
                if self.cancel.wait(max(0.0, due - time.monotonic())):
                    return
                before = time.monotonic()
                result = subprocess.run(self.command, text=True, capture_output=True,
                                        check=True, timeout=4)
                after = time.monotonic()
                fields = list(csv.reader(result.stdout.strip().splitlines()))
                if len(fields) != 1 or len(fields[0]) != 6:
                    raise ValueError(f"unexpected telemetry: {result.stdout}")
                timestamp, uuid, watts, utilization, clock, temperature = [s.strip() for s in fields[0]]
                power = float(watts)
                if uuid != self.gpu_uuid or not math.isfinite(power) or power <= 0:
                    raise ValueError(f"invalid power telemetry: {result.stdout}")
                self.rows.append({"index": index, "scheduled_seconds": index * INTERVAL_SECONDS,
                                  "query_start_seconds": before - self.start_time,
                                  "query_end_seconds": after - self.start_time,
                                  "sensor_timestamp": timestamp, "gpu_uuid": uuid,
                                  "power_w": power, "gpu_utilization_percent": utilization,
                                  "sm_clock_mhz": clock, "temperature_c": temperature,
                                  "raw_csv": result.stdout.strip()})
        except BaseException as exc:
            self.error = repr(exc)
        finally:
            self.done.set()


def run_case(args, case, entry):
    variant = baseline.CASES[case]
    path = args.inputs / entry["checkpoint"]
    if baseline.sha256_file(path) != entry["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {path}")
    checkpoint = baseline.load_checkpoint(path)
    if checkpoint.manifest["variant"] != variant.as_dict():
        raise ValueError(f"workload definition changed: {case}")
    mask = baseline.validate_checkpoint_topology(checkpoint.weights, variant)
    data = baseline.load_mnist(args.data_path, "train")
    network = baseline.GeNNNetwork(
        weights=checkpoint.weights.copy(), theta_mv=checkpoint.theta_mv.copy(),
        plasticity=True, inhibition=baseline.MODEL.train_inhibition, seed=20260724,
        backend="cuda", build_path=args.output / "builds" / case, variant=variant,
        structural_mask=mask, precision="float", parallelism="postsynaptic",
        num_threads_per_spike=1, timing_enabled=False,
        reuse_build=args.reuse_build_root / case if args.reuse_build_root else None)
    accepted = attempts = 0
    intensity = baseline.MODEL.initial_intensity
    sampler = None
    warmup_start = time.monotonic()
    first_100 = None
    try:
        while True:
            # This is the baseline attempt schedule; neither phase writes files.
            network.normalize(validate=False)
            sample = checkpoint.accepted_samples + accepted
            if sample >= len(data.images):
                raise RuntimeError("exhausted the sequential training split")
            network.set_image(data.images[sample], intensity)
            counts = network.run_stimulus()
            total = int(counts.sum())
            if total >= 5000:
                raise RuntimeError(f"runaway activity: {total} E spikes")
            retry = total < baseline.MODEL.minimum_exc_spikes
            network.run_rest(synchronize=False)
            attempts += 1
            if retry:
                intensity += 1.0
                if intensity > 20.0:
                    raise RuntimeError("retry intensity exceeded 20")
            else:
                accepted += 1
                intensity = baseline.MODEL.initial_intensity
            if first_100 is None and accepted == 100:
                first_100 = {"attempts": attempts, "event_counters": network.event_counters()}
                expected = args.reference["first_100"][case]
                if first_100 != expected:
                    raise RuntimeError(f"first 100 images differ from recorded baseline: {first_100}")
            if sampler is None:
                # End warmup at an accepted-image boundary with all rest delivered.
                if accepted >= 100 and not retry and time.monotonic() - warmup_start >= WARMUP_SECONDS:
                    network.validate_runtime(counts, 5000)
                    start_counters = network.event_counters()
                    start_attempts, start_accepted = attempts, accepted
                    measured_start = time.monotonic()
                    warmup_elapsed = measured_start - warmup_start
                    sampler = PowerSampler(args.gpu_uuid, measured_start)
                    sampler.start()
                    print(f"MEASURE_START case={case} warmup_images={accepted}", flush=True)
            elif sampler.done.is_set():
                network.total_spike_count()  # Complete the final asynchronous rest.
                measured_end = time.monotonic()
                if sampler.error:
                    raise RuntimeError(sampler.error)
                if len(sampler.rows) != SAMPLE_COUNT:
                    raise RuntimeError("power sample count must be exactly five")
                break
        elapsed = measured_end - measured_start
        measured_steps = (attempts - start_attempts) * baseline.MODEL.attempt_ticks
        network.validate_runtime(counts, 5000)
        final_counters = network.event_counters()
        weights, theta = network.weights(), network.theta_mv()
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(theta)):
            raise RuntimeError("non-finite weights or thresholds")
        if weights.min() < 0:
            raise RuntimeError("negative weights")
        # Column normalization is not clipped to the STDP event cap. Apply the
        # model's declared normalization policy (dense: no cap tolerance), and
        # retain observed extrema instead of imposing a new 2% model constraint.
        validate_normalized_weight_bound(weights, variant)
        powers = [row["power_w"] for row in sampler.rows]
        result = {
            "case": case, "gpu": args.gpu, "gpu_uuid": args.gpu_uuid,
            "checkpoint_writing": False, "checkpoint_sha256": entry["checkpoint_sha256"],
            "warmup_seconds": warmup_elapsed, "warmup_accepted_samples": start_accepted,
            "warmup_attempts": start_attempts, "first_100_baseline_match": True,
            "first_100": first_100, "wall_seconds": elapsed,
            "simulation_steps": measured_steps, "us_per_step": elapsed * 1e6 / measured_steps,
            "accepted_samples": accepted - start_accepted, "attempts": attempts - start_attempts,
            "accepted_images_before_run": checkpoint.accepted_samples,
            "measured_image_start_index": checkpoint.accepted_samples + start_accepted,
            "next_image_index": checkpoint.accepted_samples + accepted,
            "ended_during_retry": retry,
            "event_counters": {k: final_counters[k] - start_counters[k] for k in final_counters},
            "sampler_command": sampler.command, "power_samples": sampler.rows,
            "mean_power_w": statistics.mean(powers), "min_power_w": min(powers),
            "max_power_w": max(powers), "sample_sd_power_w": statistics.stdev(powers),
            "energy_uJ_per_step": statistics.mean(powers) * elapsed * 1e6 / measured_steps,
            "diagnostic": {"final_runtime": network.runtime_diagnostics(),
                           "weight_min": float(weights.min()), "weight_max": float(weights.max()),
                           "normalization_weight_max_tolerance": variant.normalization_weight_max_tolerance,
                           "weight_max_relative_to_stdp_cap": float(weights.max() / variant.weight_max),
                           "column_sum_min": float(weights.sum(axis=0).min()),
                           "column_sum_max": float(weights.sum(axis=0).max()),
                           "theta_min_mv": float(theta.min()), "theta_max_mv": float(theta.max()),
                           "runtime_guards_passed": True},
        }
        if list(args.output.rglob("*.npz")) or list(args.output.rglob("*.npy")):
            raise RuntimeError("unexpected checkpoint/array output")
        return result
    finally:
        if sampler is not None:
            sampler.cancel.set()
            sampler.join(timeout=5)
            baseline.write_json(args.output / f"{case}_power_samples.json", sampler.rows)
        network.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, choices=("RTX3090", "A100", "A800", "H800"))
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=ROOT / "data/mnist")
    parser.add_argument("--reuse-build-root", type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    args.reference = json.loads(args.reference.read_text())
    input_manifest = json.loads((args.inputs / "manifest.json").read_text())
    source = baseline.provenance()
    if source["source_sha256"] != args.reference["source_sha256"]:
        raise ValueError("simulation sources differ from the recorded baseline")
    if baseline.sha256_file(args.inputs / "manifest.json") != args.reference["input_manifest_sha256"]:
        raise ValueError("input manifest changed")
    for relative, expected in input_manifest["dataset_sha256"].items():
        if baseline.sha256_file(args.data_path / Path(relative).name) != expected:
            raise ValueError(f"dataset hash mismatch: {relative}")
    if args.reuse_build_root:
        origin = json.loads((args.reuse_build_root.parent / "manifest.json").read_text())
        if origin["source_sha256"] != source["source_sha256"] or origin["seed"] != 20260724:
            raise ValueError("reused build source/seed mismatch")
        for key in ("CUDA_PATH", "CUDAHOSTCXX", "NVCC_PREPEND_FLAGS", "CUDA_VISIBLE_DEVICES"):
            if origin["environment"][key] != source["environment"][key]:
                raise ValueError(f"reused build environment changed: {key}")
    processes = source["gpu_processes"]
    if processes["returncode"] != 0:
        raise RuntimeError("cannot check GPU process occupancy")
    if args.gpu_uuid in processes["output"]:
        raise RuntimeError("selected GPU already has a compute process")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "builds").mkdir()
    (args.output / "logs").mkdir()
    baseline.write_json(args.output / "manifest.json", {
        **source, "schema": "mnist-gpu-power-five-samples-v1", "gpu_label": args.gpu,
        "gpu_uuid": args.gpu_uuid, "script_sha256": baseline.sha256_file(Path(__file__)),
        "input_manifest_sha256": args.reference["input_manifest_sha256"],
        "cases": args.cases, "seed": 20260724, "checkpoint_writing": False,
        "hardware_reference_power_w": 33, "power_sample_count_per_case": SAMPLE_COUNT,
        "power_sample_interval_seconds": INTERVAL_SECONDS, "warmup_seconds": WARMUP_SECONDS,
        "stopping_rule": "Finish an attempt after the fifth power query completes; synchronize final rest.",
        "measurement_scope": "Training normalization, transfers, retry decisions, presentation and rest; "
                             "build/load/warmup/final diagnostics excluded. Running board power; no idle subtraction.",
        "baseline_timing_comparability": "Fresh 25-second continuation after 10-second warmup; "
                                        "different image range from historical 100-image timings.",
        "reuse_build_root": str(args.reuse_build_root) if args.reuse_build_root else None,
    })
    for case in args.cases:
        print(f"CASE_START {case}", flush=True)
        with baseline.sweep.redirect_process_output(args.output / "logs" / f"{case}.log"):
            result = run_case(args, case, input_manifest["cases"][case])
        baseline.write_json(args.output / f"{case}.json", result)
        print(f"COMPLETE {case} watts={result['mean_power_w']:.3f} "
              f"us_per_step={result['us_per_step']:.6f}", flush=True)
    baseline.write_json(args.output / "complete.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
                                                       "cases": args.cases, "checkpoint_writing": False})


if __name__ == "__main__":
    main()
