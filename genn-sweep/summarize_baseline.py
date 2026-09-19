#!/usr/bin/env python3
"""Validate and aggregate completed baseline runs without discarding outliers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reimpl"))
from baseline_cases import mnist_cases


def load(path):
    return json.loads(path.read_text())


def summarize(inputs, machines):
    input_manifest = load(inputs / "manifest.json")
    input_hash = hashlib.sha256((inputs / "manifest.json").read_bytes()).hexdigest()
    cases = tuple(mnist_cases()) + ("brunel_additive", "brunel_morrison")
    report = {"schema": "genn-gpu-baseline-v1", "precision": "float",
              "input_manifest": input_manifest, "input_manifest_sha256": input_hash,
              "quantiles": "numpy.quantile(method=linear)", "machines": {}}
    source_hashes = None
    for name, diagnostic_path, native_prefix in machines:
        diagnostic = Path(diagnostic_path)
        repetitions = [Path(f"{native_prefix}{i}") for i in range(1, 6)]
        manifests = [load(path / "manifest.json") for path in repetitions]
        for manifest in manifests:
            if (manifest["mode"] != "timing" or manifest["seed"] != 20260724
                    or manifest["input_manifest_sha256"] != input_hash
                    or tuple(manifest["cases"]) != cases):
                raise ValueError(f"incompatible timing protocol: {name}")
            if source_hashes is None:
                source_hashes = manifest["source_sha256"]
            if manifest["source_sha256"] != source_hashes:
                raise ValueError(f"simulation source differs: {name}")
        machine = {"timing_manifests": manifests,
                   "diagnostic_manifest": load(diagnostic / "manifest.json"), "cases": {}}
        for case in cases:
            rows = [load(path / f"{case}.json") for path in repetitions]
            diagnostics = load(diagnostic / f"{case}.json")
            counters = diagnostics["event_counters"]
            expected = counters.get("total_firing_count", counters.get("total_spikes"))
            for row in rows:
                if row["total_spikes"] != expected:
                    raise ValueError(f"timing/diagnostic spike mismatch: {name}/{case}")
                measured = row["wall_seconds"] * 1e6 / row["simulation_steps"]
                if not np.isclose(row["us_per_step"], measured, rtol=1e-12):
                    raise ValueError(f"inconsistent time denominator: {name}/{case}")
            values = np.array([row["us_per_step"] for row in rows])
            machine["cases"][case] = {
                "median_us_per_step": float(np.median(values)),
                "min_us_per_step": float(values.min()), "max_us_per_step": float(values.max()),
                "iqr_us_per_step": float(np.quantile(values, .75) - np.quantile(values, .25)),
                "raw_repetitions": rows, "diagnostic": diagnostics,
            }
        report["machines"][name] = machine
    report["source_sha256"] = source_hashes
    report["cross_machine_dynamics"] = {}
    for case in cases:
        counters = [m["cases"][case]["diagnostic"]["event_counters"]
                    for m in report["machines"].values()]
        fields = ("excitatory_spikes", "inhibitory_spikes")
        if case.startswith("mnist"):
            fields += ("input_spikes",)
        report["cross_machine_dynamics"][case] = {
            "identical_population_totals": all(len({c[field] for c in counters}) == 1 for field in fields),
            "population_totals": {name: {field: machine["cases"][case]["diagnostic"]
                ["event_counters"][field] for field in fields}
                for name, machine in report["machines"].items()},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--machine", nargs=3, action="append", required=True,
                        metavar=("NAME", "DIAGNOSTIC_DIRECTORY", "NATIVE_DIRECTORY_PREFIX"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.inputs, args.machine)
    with args.output.open("x", encoding="ascii") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(f"Verified {23 * 5 * len(args.machine)} timings across {len(args.machine)} GPUs: {args.output}")


if __name__ == "__main__":
    main()
