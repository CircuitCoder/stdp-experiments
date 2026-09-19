#!/usr/bin/env python3
"""Summarize five-sample GPU power measurements and render the 33 W comparison."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics as stats
import sys

import compare


HARDWARE_POWER_W = 33.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=compare.ROOT / "copilot/tmp")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "power")
    parser.add_argument("--run", nargs=2, action="append", default=[], metavar=("GPU", "DIRECTORY"))
    parser.add_argument("--scale", choices=("linear", "log"), default="linear")
    args = parser.parse_args()
    run_overrides = {gpu: Path(path) for gpu, path in args.run}
    hardware_path = compare.ROOT / "mnist_dense_results.json"
    baseline_path = compare.ROOT / "genn-sweep/baseline-20260907-matrix.json"
    hardware, historical, baseline = compare.load_inputs(hardware_path, baseline_path)
    raw, normalized, source_manifests = {}, {}, {}
    for gpu in compare.GPUS:
        run = run_overrides.get(gpu, args.runs / f"power_dense_20260910_v1_{gpu.lower()}")
        manifest = json.loads((run / "manifest.json").read_text())
        complete = json.loads((run / "complete.json").read_text())
        if manifest["checkpoint_writing"] or complete["checkpoint_writing"]:
            raise ValueError(f"checkpoint writing was enabled for {gpu}")
        if manifest["source_sha256"] != baseline["source_sha256"]:
            raise ValueError(f"simulation source mismatch for {gpu}")
        if manifest["input_manifest_sha256"] != baseline["input_manifest_sha256"]:
            raise ValueError(f"input manifest mismatch for {gpu}")
        source_manifests[gpu] = manifest
        raw[gpu], normalized[gpu] = {}, []
        for trace, rule in enumerate(compare.RULES, 1):
            row = json.loads((run / f"mnist_{trace}trace_dense.json").read_text())
            samples = row["power_samples"]
            if (len(samples) != 5 or row["checkpoint_writing"]
                    or not row["first_100_baseline_match"] or not row["diagnostic"]["runtime_guards_passed"]):
                raise ValueError(f"invalid measurement: {gpu}/{rule}")
            times = [s["query_start_seconds"] for s in samples]
            if any(abs(t - 5 * i) > .5 for i, t in enumerate(times, 1)):
                raise ValueError(f"power query schedule drifted: {gpu}/{rule}")
            if any(s["query_end_seconds"] > row["wall_seconds"] for s in samples):
                raise ValueError("a power query fell outside the timed workload")
            if row["simulation_steps"] != row["attempts"] * 1000:
                raise ValueError("invalid timestep count")
            elapsed_us = row["wall_seconds"] * 1e6 / row["simulation_steps"]
            if not math.isclose(elapsed_us, row["us_per_step"], rel_tol=1e-12):
                raise ValueError("invalid elapsed time denominator")
            powers = [s["power_w"] for s in samples]
            if any(not math.isfinite(p) or p <= 0 for p in powers):
                raise ValueError("invalid measured power")
            if not math.isclose(stats.mean(powers), row["mean_power_w"], rel_tol=1e-12):
                raise ValueError("invalid average power")
            raw[gpu][rule] = row
            normalized[gpu].extend({"class": f"{rule} dense", "presenting": None,
                                    "cycles": row["us_per_step"] * p / HARDWARE_POWER_W * 1000}
                                   for p in powers)
    hw, report = compare.summarize(hardware, normalized)
    for rule in compare.GROUPS:
        group = report["groups"][rule]
        group["hardware"]["power_w"] = HARDWARE_POWER_W
        group["hardware"]["energy_uJ_per_step"] = group["hardware"]["mean_us"] * HARDWARE_POWER_W
        for gpu in compare.GPUS:
            item = group["gpus"][gpu]
            item["power_samples"] = item.pop("repetitions")
            item["equivalent_mean_us_at_33w"] = item.pop("mean_us")
            item["mean_sample_energy_advantage"] = item.pop("mean_sample_speedup")
            item["ratio_of_mean_energies"] = item.pop("ratio_of_mean_durations")
            if rule != "Combined":
                item["equivalent_us_at_33w_from_each_power_sample"] = item.pop("raw_repetition_means_us")
            selected = list(compare.RULES) if rule == "Combined" else [rule]
            rows = [raw[gpu][r] for r in selected]
            counts = [len(hw[r]) for r in selected]
            weighted = lambda xs: sum(n * v for n, v in zip(counts, xs)) / sum(counts)
            item["measured_mean_us_per_step"] = weighted([r["us_per_step"] for r in rows])
            item["equal_rule_mean_power_w"] = weighted([r["mean_power_w"] for r in rows])
            item["estimated_mean_energy_uJ_per_step"] = weighted([r["mean_power_w"] * r["us_per_step"] for r in rows])
            # Keep historical timing + new power as an explicitly separate estimate.
            old_times = [stats.mean(v["cycles"] / 1000 for v in historical[gpu]
                                   if v["class"] == r + " dense") for r in selected]
            old_normalized = [t * row["mean_power_w"] / HARDWARE_POWER_W for t, row in zip(old_times, rows)]
            item["historical_timing_projection_us_at_33w"] = weighted(old_normalized)
            item["historical_timing_projection_mean_sample_advantage"] = weighted([
                stats.mean(t / h for h in hw[r]) for r, t in zip(selected, old_normalized)])
        group["all_gpus"] = {k: stats.mean(item[k] for item in group["gpus"].values()) for k in (
            "equivalent_mean_us_at_33w", "mean_sample_energy_advantage", "ratio_of_mean_energies",
            "estimated_mean_energy_uJ_per_step", "historical_timing_projection_mean_sample_advantage")}
    report["schema"] = "dense-mnist-power-normalized-v1"
    report["hardware_power_w"] = HARDWARE_POWER_W
    report["power_measurement"] = "Five board-power snapshots, five seconds apart, during each GPU/rule's single timed training run."
    report["interpretation"] = "Energy and performance-per-watt estimates, not measured operation at a 33 W GPU cap."
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data").mkdir(exist_ok=True)
    for gpu in compare.GPUS:
        compare.write_json(args.output / "data" / f"{gpu.lower()}_33w.json", normalized[gpu])
        compare.write_json(args.output / "data" / f"{gpu.lower()}_measurements.json", raw[gpu])
    compare.write_json(args.output / "summary.json", report)
    with (args.output / "power_samples.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gpu", "rule", "sample", "query_start_seconds", "power_w", "gpu_utilization_percent"])
        for gpu in compare.GPUS:
            for rule in compare.RULES:
                for sample in raw[gpu][rule]["power_samples"]:
                    writer.writerow([gpu, rule, sample["index"], sample["query_start_seconds"],
                                     sample["power_w"], sample["gpu_utilization_percent"]])
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        fields = ["measured_mean_us_per_step", "equal_rule_mean_power_w", "estimated_mean_energy_uJ_per_step",
                  "equivalent_mean_us_at_33w", "mean_sample_energy_advantage", "ratio_of_mean_energies"]
        writer = csv.writer(stream)
        writer.writerow(["gpu", "group", *fields])
        for gpu in compare.GPUS:
            for group in compare.GROUPS:
                item = report["groups"][group]["gpus"][gpu]
                writer.writerow([gpu, group, *(item[k] for k in fields)])
    _, chart = compare.summarize(hardware, normalized)
    absolute_gpu_us = {group: {gpu: report["groups"][group]["gpus"][gpu]["measured_mean_us_per_step"]
                               for gpu in compare.GPUS} for group in compare.GROUPS}
    suffix = "; log scale" if args.scale == "log" else ""
    compare.plot(hw, chart, args.output, args.scale, basename="mnist_dense_33w",
                 absolute_gpu_us=absolute_gpu_us, top_adjusted_gpu="RTX3090", labels={
        "title": "Dense MNIST: absolute and 33 W-adjusted performance",
        "subtitle": "400 excitatory neurons  ·  ActiveN: 1 GHz, 33 W  ·  GPU: measured running board power",
        "ylabel": f"Time per simulation step (µs{suffix})",
        "note1": "Dashed GPU bars = absolute duration × mean running power / 33 W. Five power readings per rule, 5 seconds apart.",
        "note2": "Multipliers = absolute GPU mean / ActiveN mean. 33 W adjustment estimates performance per watt; no physical power cap.",
    })
    write_report(report, raw, args.output)
    compare.write_json(args.output / "manifest.json", {
        "command": [sys.executable, *sys.argv], "hardware_sha256": compare.sha256(hardware_path),
        "gpu_baseline_sha256": compare.sha256(baseline_path), "measurement_manifests": source_manifests,
        "report_script_sha256": compare.sha256(Path(__file__)),
        "plot_script_sha256": compare.sha256(Path(compare.__file__)),
        "software": {"python": sys.version, "numpy": compare.np.__version__,
                     "matplotlib": compare.matplotlib.__version__},
        "plot": {"y_scale": args.scale, "hardware_label": "ActiveN",
                 "center_bar": "25th to 75th percentile", "diamond": "arithmetic mean",
                 "solid_gpu_bars": "measured_mean_us_per_step from the selected power measurement runs",
                 "dashed_gpu_bars": "equivalent_mean_us_at_33w from the same runs",
                 "absolute_gpu_multiplier": "measured_mean_us_per_step / ActiveN mean_us for the same group",
                 "top_strip": {"gpu": "RTX3090", "series": "33 W adjusted", "included_in_axis_limits": False,
                               "placement": "fixed row above the axes; labels retain actual microseconds"}},
        "selected_runs": {gpu: str(run_overrides.get(gpu, args.runs / f"power_dense_20260910_v1_{gpu.lower()}"))
                          for gpu in compare.GPUS},
        "output_sha256": {str(p.relative_to(args.output)): compare.sha256(p) for p in args.output.rglob("*")
                          if p.is_file() and p.name not in ("manifest.json", "README.md")},
        "gpu_json_cycles_field": "Equivalent nanoseconds at 33 W, not physical GPU cycles or observed GPU latency.",
    })
    print(json.dumps(report["groups"]["Combined"]["all_gpus"], indent=2))


def write_report(report, raw, output):
    lines = ["# Dense MNIST: absolute and 33 W-adjusted performance", "", "![Absolute and power-adjusted violin plot](mnist_dense_33w.png)", "",
             "Solid GPU bars show measured duration; dashed bars show duration adjusted to 33 W, "
             "using timing and running power from the same run. ActiveN violins include vertical "
             "25th–75th percentile bars and arithmetic-mean diamonds. RTX 3090's adjusted markers sit "
             "in a labeled strip above the axis and do not set its range; their labels retain the actual durations. "
             "Each absolute-point multiplier is the measured GPU mean / ActiveN mean for its group, "
             "distinct from the mean sample energy advantage reported below.", "",
             "Measured running board power: exactly five readings per GPU/rule, five seconds apart. "
             "Each case has one ~25-second timed training run after 10 seconds of warmup. "
             "Checkpoint writing and per-timestep GeNN event timing were disabled.", "",
             "| GPU | Rule | Five readings (W) | Mean (W) | Measured µs/step | Estimated µJ/step | Equivalent µs at 33 W |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for gpu in compare.GPUS:
        for rule in compare.RULES:
            row = raw[gpu][rule]
            readings = ", ".join(f"{p['power_w']:.2f}" for p in row["power_samples"])
            lines.append(f"| {compare.LABELS[gpu]} | {rule} | {readings} | {row['mean_power_w']:.3f} | "
                         f"{row['us_per_step']:.3f} | {row['energy_uJ_per_step']:.2f} | {row['energy_uJ_per_step']/33:.3f} |")
    lines += ["", "ActiveN advantage after power normalization, using the confirmed arithmetic mean of sample ratios:", "",
              "| GPU | 1-trace | 2-trace | 3-trace | Combined |", "| --- | ---: | ---: | ---: | ---: |"]
    for gpu in (*compare.GPUS, "All GPUs"):
        entries = [report["groups"][r]["all_gpus"] if gpu == "All GPUs" else report["groups"][r]["gpus"][gpu]
                   for r in compare.GROUPS]
        lines.append("| " + compare.LABELS.get(gpu, gpu) + " | " + " | ".join(
            f"{r['mean_sample_energy_advantage']:.3f}×" for r in entries) + " |")
    lines += ["", "Combined values (rules and hardware samples equally weighted):", "",
              "| GPU | Estimated mean µJ/step | Equivalent mean µs at 33 W | Ratio of mean energies | Historical timings × new power: mean sample advantage |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for gpu in compare.GPUS:
        row = report["groups"]["Combined"]["gpus"][gpu]
        lines.append(f"| {compare.LABELS[gpu]} | {row['estimated_mean_energy_uJ_per_step']:.2f} | "
                     f"{row['equivalent_mean_us_at_33w']:.3f} | {row['ratio_of_mean_energies']:.3f}× | "
                     f"{row['historical_timing_projection_mean_sample_advantage']:.3f}× |")
    all_gpus = report["groups"]["Combined"]["all_gpus"]
    lines += ["", f"All-GPU mean sample energy advantage: **{all_gpus['mean_sample_energy_advantage']:.3f}×**. "
              f"Ratio of overall mean energies: **{all_gpus['ratio_of_mean_energies']:.3f}×**.", "",
              "ActiveN mean energy at the supplied 33 W is **113.813 µJ/step** "
              "(1-trace: 85.332; 2-trace: 151.123; 3-trace: 104.984 µJ/step).", "",
              "Energy per step is estimated as mean sampled watts × measured microseconds per step. "
              "The 33 W equivalent duration is that energy divided by 33. The primary advantage averages "
              "this equivalent duration / each hardware sample duration, keeping the measured 70:30 "
              "presentation/rest mixture. Each of the five power readings receives equal weight. "
              "Combined energy averages each rule's power × time; it does not multiply pooled power and pooled time.", "",
              "Fresh timing and power were collected together. These runs continue past a 10-second warmup, "
              "so their image ranges differ from the historical 100-image GPU timing runs. Every new run first "
              "reproduced the historical first-100-image attempts and spike counters. The historical-timing "
              "projection is listed separately and combines older timing with new measured power.", "",
              "Five sensor snapshots give a coarse running-power estimate, not a continuous energy integral. "
              "Ranges and raw samples are retained; no confidence interval is inferred from five correlated "
              "readings. NVIDIA reports board power, which excludes the host CPU. Idle board overhead is included. "
              "ActiveN's 33 W is the user-supplied design figure, not a new power measurement. "
              "This comparison neither uses TDP nor predicts actual GPU latency under a 33 W cap.", "",
              "Sensor semantics: [NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html#gpu-power-readings). "
              "Depending on GPU/driver, `power.draw` is an instantaneous or averaged board-power reading.", "",
              "[Raw power CSV](power_samples.csv) · [Comparison CSV](comparison.csv) · "
              "[Summary JSON](summary.json) · [Run manifests](manifest.json) · [SVG plot](mnist_dense_33w.svg)", ""]
    (output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
