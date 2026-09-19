#!/usr/bin/env python3
"""Compare dense MNIST hardware samples with the recorded GeNN GPU baseline."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RULES = ("1-trace", "2-trace", "3-trace")
GROUPS = (*RULES, "Combined")
GPUS = ("RTX3090", "A100", "A800", "H800")
LABELS = {"Hardware": "ActiveN", "RTX3090": "RTX 3090", "A100": "A100", "A800": "A800", "H800": "H800"}
COLORS = {"Hardware": "#07838B", "RTX3090": "#7952B3", "A100": "#2864B7",
          "A800": "#C27B13", "H800": "#CD4561"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_inputs(hardware_path, gpu_path):
    hardware = json.loads(hardware_path.read_text())
    baseline = json.loads(gpu_path.read_text())
    if not isinstance(hardware, list) or not hardware:
        raise ValueError("hardware must be a nonempty array")
    for row in hardware:
        if (set(row) != {"class", "presenting", "cycles"}
                or row["class"] not in {f"{rule} dense" for rule in RULES}
                or type(row["presenting"]) is not bool
                or type(row["cycles"]) not in (int, float)
                or not math.isfinite(row["cycles"]) or row["cycles"] <= 0):
            raise ValueError(f"invalid hardware record: {row}")
    if baseline["schema"] != "genn-gpu-baseline-v1" or baseline["precision"] != "float":
        raise ValueError("expected the FP32 GeNN baseline")
    if set(baseline["machines"]) != set(GPUS):
        raise ValueError("GPU set changed; update the plot labels and colors explicitly")
    gpu_records = {}
    for gpu in GPUS:
        machine = baseline["machines"][gpu]
        manifests = machine["timing_manifests"]
        if len(manifests) != 5:
            raise ValueError(f"expected five timing repetitions for {gpu}")
        for manifest in manifests:
            if (manifest["mode"] != "timing" or manifest["seed"] != 20260724
                    or manifest["samples"] != 100
                    or manifest["source_sha256"] != baseline["source_sha256"]
                    or manifest["input_manifest_sha256"] != baseline["input_manifest_sha256"]):
                raise ValueError(f"incompatible timing manifest for {gpu}")
        records = []
        for trace, rule in enumerate(RULES, 1):
            case = f"mnist_{trace}trace_dense"
            rows = machine["cases"][case]["raw_repetitions"]
            if len(rows) != 5:
                raise ValueError(f"expected five repetitions for {gpu}/{case}")
            if not baseline["cross_machine_dynamics"][case]["identical_population_totals"]:
                raise ValueError(f"GPU population totals differ for {case}")
            for row in rows:
                ns = row["wall_seconds"] * 1e9 / row["simulation_steps"]
                if (not math.isfinite(ns) or ns <= 0
                        or row["simulation_steps"] != row["attempts"] * 1000
                        or row["accepted_samples"] != 100
                        or not math.isclose(ns, row["us_per_step"] * 1000, rel_tol=1e-12)):
                    raise ValueError(f"invalid timing denominator for {gpu}/{case}")
                # Preserve the input field names. GPU 'cycles' is equivalent ns,
                # not the GPU's physical clock count; null means both phases.
                records.append({"class": f"{rule} dense", "presenting": None,
                                "cycles": row["us_per_step"] * 1000})
        gpu_records[gpu] = records
    return hardware, gpu_records, baseline


def summarize(hardware, gpu_records):
    hw = {rule: np.array([r["cycles"] / 1000 for r in hardware
                          if r["class"] == f"{rule} dense"]) for rule in RULES}
    if any(len(values) < 2 or np.ptp(values) == 0 for values in hw.values()):
        raise ValueError("each hardware rule needs at least two distinct timings for a violin")
    hw["Combined"] = np.concatenate([hw[rule] for rule in RULES])
    counts = np.array([len(hw[rule]) for rule in RULES])
    report = {"unit": "microseconds", "hardware_clock_hz": 1_000_000_000,
              "groups": {}}
    for group in GROUPS:
        rows = [r for r in hardware if group == "Combined" or r["class"] == f"{group} dense"]
        values = hw[group]
        report["groups"][group] = {"hardware": {
            "samples": len(values), "presentation_samples": sum(r["presenting"] for r in rows),
            "rest_samples": sum(not r["presenting"] for r in rows),
            "mean_us": float(np.mean(values)), "median_us": float(np.median(values)),
            "min_us": float(values.min()), "max_us": float(values.max()),
            "q25_us": float(np.quantile(values, .25)), "q75_us": float(np.quantile(values, .75)),
        }, "gpus": {}}
    for gpu, records in gpu_records.items():
        means, speedups = [], []
        for rule in RULES:
            repetitions = [r["cycles"] / 1000 for r in records if r["class"] == f"{rule} dense"]
            mean = float(np.mean(repetitions))
            # Exactly equals the mean of G_j / H_i over all sample/repetition
            # combinations within this rule. No timestep pairing is inferred.
            speedup = float(np.mean(mean / hw[rule]))
            means.append(mean)
            speedups.append(speedup)
            report["groups"][rule]["gpus"][gpu] = {
                "repetitions": len(repetitions), "raw_repetition_means_us": repetitions,
                "mean_us": mean, "mean_sample_speedup": speedup,
                "ratio_of_mean_durations": mean / float(np.mean(hw[rule])),
            }
        combined_mean = float(np.average(means, weights=counts))
        report["groups"]["Combined"]["gpus"][gpu] = {
            "repetitions": len(records), "mean_us": combined_mean,
            "mean_sample_speedup": float(np.average(speedups, weights=counts)),
            "ratio_of_mean_durations": combined_mean / float(np.mean(hw["Combined"])),
        }
    for group in GROUPS:
        entries = report["groups"][group]["gpus"].values()
        report["groups"][group]["all_gpus"] = {
            key: float(np.mean([entry[key] for entry in entries]))
            for key in ("mean_us", "mean_sample_speedup", "ratio_of_mean_durations")}
    return hw, report


def plot(hw, report, output, scale, *, labels=None, basename="mnist_dense_violin",
         absolute_gpu_us=None, top_adjusted_gpu=None):
    labels = labels or {}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "svg.hashsalt": "mnist-hardware-v1"})
    overlay = absolute_gpu_us is not None
    top_strip = overlay and top_adjusted_gpu in GPUS
    fig, ax = plt.subplots(figsize=(13.8, 9.0 if overlay else 8.5))
    fig.subplots_adjust(left=.08, right=.98,
                        top=.69 if top_strip else (.77 if overlay else .80), bottom=.18)
    fig.text(.08, .952, labels.get("title", "Dense MNIST: ActiveN and GPU timestep duration"),
             fontsize=21, weight="bold", color="#172333")
    fig.text(.08, .913, labels.get("subtitle", "400 excitatory neurons  ·  ActiveN at 1 GHz  ·  Presentation + rest"),
             fontsize=12, color="#526174")
    artists = ax.violinplot([hw[group] for group in GROUPS], positions=range(4),
                            widths=.58, points=512, bw_method="scott",
                            showextrema=False, showmedians=False)
    for group, body in zip(GROUPS, artists["bodies"]):
        body.set_gid(f"activen-violin-{group}")
        body.set_facecolor(COLORS["Hardware"])
        body.set_edgecolor(COLORS["Hardware"])
        body.set_alpha(.40)
        body.set_linewidth(1.2)
    for x, group in enumerate(GROUPS):
        data = report["groups"][group]
        avg = data["hardware"]["mean_us"]
        ax.plot([x, x], [data["hardware"]["q25_us"], data["hardware"]["q75_us"]],
                color="#034E53", lw=5, solid_capstyle="butt", zorder=4,
                gid=f"activen-iqr-{group}")
        # Keep the mean beside the center bar so it cannot hide a short IQR
        # when the linear axis compresses the ActiveN distribution.
        ax.scatter(x-.06, avg, marker="D", s=42, c=COLORS["Hardware"],
                   edgecolors="white", linewidths=1.1, zorder=5,
                   gid=f"activen-mean-{group}")
        ax.annotate(f"{avg:.2f}", (x, avg), xytext=(28, 0),
                    textcoords="offset points", va="center", color=COLORS["Hardware"],
                    fontsize=10, weight="bold")
        for gpu_index, gpu in enumerate(GPUS):
            # Separate GPU columns keep nearby absolute means readable on a
            # linear axis. Each GPU's absolute/adjusted pair shares its column.
            center = x + (gpu_index - 1.5) * .20 if overlay else x
            half_width = .08 if overlay else .15
            values = [("33w" if overlay else "absolute", data["gpus"][gpu]["mean_us"])]
            if overlay:
                values.insert(0, ("absolute", absolute_gpu_us[group][gpu]))
            for kind, avg in values:
                in_strip = top_strip and kind == "33w" and gpu == top_adjusted_gpu
                display_y = 1.065 if in_strip else avg
                transform = ax.get_xaxis_transform() if in_strip else ax.transData
                ax.plot([center-half_width, center+half_width], [display_y, display_y],
                        color=COLORS[gpu], lw=2.6 if overlay else 3.0,
                        linestyle=(0, (3, 2)) if kind == "33w" else "-",
                        solid_capstyle="round", dash_capstyle="butt", zorder=4,
                        transform=transform, clip_on=not in_strip,
                        gid=f"gpu-{kind}-{gpu}-{group}")
                label = f"{avg:.2f} µs" if in_strip else f"{avg:.2f}"
                if kind == "absolute":
                    ax.scatter(center, avg, c=COLORS[gpu], s=28 if overlay else 34, zorder=5)
                    multiplier = avg / data["hardware"]["mean_us"]
                    label += f"\n{multiplier:.1f}×" if overlay else f" ({multiplier:.1f}×)"
                ax.annotate(label, (center if overlay else center+half_width, display_y),
                            xycoords=transform, annotation_clip=not in_strip,
                            xytext=(0, 6) if overlay else (7, 0), textcoords="offset points",
                            ha="center" if overlay else "left", va="bottom" if overlay else "center",
                            color=COLORS[gpu], fontsize=9 if overlay else 10, weight="medium",
                            gid=f"gpu-label-{kind}-{gpu}-{group}")
    ax.set_yscale(scale)
    maximum = max(d["mean_us"] for g in report["groups"].values()
                  for gpu, d in g["gpus"].items() if not (top_strip and gpu == top_adjusted_gpu))
    if overlay:
        maximum = max(maximum, max(v for group in absolute_gpu_us.values() for v in group.values()))
    maximum = max(maximum, float(hw["Combined"].max()))
    if scale == "log":
        ax.set_ylim(float(hw["Combined"].min()) * .77, maximum * 1.15)
        ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 100, 200, 500]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(NullLocator())
    else:
        ax.set_ylim(0, maximum * 1.10)
    ax.set_xlim(-.48, 3.52)
    if top_strip:
        ax.text(0, 1.18, f"{LABELS[top_adjusted_gpu]} adjusted to 33 W (above axis range)",
                transform=ax.transAxes, color=COLORS[top_adjusted_gpu], fontsize=10,
                va="bottom", gid="gpu-top-strip-label")
        ax.plot([-.48, 3.52], [1.025, 1.025], transform=ax.get_xaxis_transform(),
                color="#E5E9EF", lw=.8, clip_on=False, gid="gpu-top-strip-divider")
    ax.set_xticks(range(4), [f"{group}\nn = {len(hw[group]):,}" for group in GROUPS])
    ax.tick_params(axis="x", length=0, pad=12, labelsize=12)
    ax.tick_params(axis="y", length=0, pad=8)
    suffix = "; log scale" if scale == "log" else ""
    ax.set_ylabel(labels.get("ylabel", f"Time per simulation step (µs{suffix})"), labelpad=12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E9EF", linewidth=.8)
    ax.axvline(2.5, color="#C6CED8", linewidth=1, linestyle=(0, (3, 5)))
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [Patch(facecolor=COLORS["Hardware"], alpha=.55, label="ActiveN (1 GHz)")]
    handles += [Line2D([0], [0], color=COLORS[gpu], linewidth=3, marker="o",
                       markersize=5, label=LABELS[gpu]) for gpu in GPUS]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.074, .881),
               ncol=5, frameon=False, columnspacing=2.2, handlelength=1.5)
    if overlay:
        styles = [Line2D([0], [0], color="#526174", lw=2.6, marker="o", markersize=4,
                         label="GPU absolute duration"),
                  Line2D([0], [0], color="#526174", lw=2.6, linestyle=(0, (3, 2)),
                         label="GPU duration adjusted to 33 W")]
        fig.legend(handles=styles, loc="upper left", bbox_to_anchor=(.074, .836),
                   ncol=2, frameon=False, columnspacing=2.2, handlelength=2.5)
    fig.text(.08, .097, "ActiveN violins: measured timesteps; vertical center bars: 25th–75th percentiles; diamonds: arithmetic means.",
             fontsize=10, color="#526174")
    fig.text(.08, .070, labels.get("note1", "GPU bars: mean of 5 run averages per rule. Combined pools all 3 rules equally."),
             fontsize=10, color="#526174")
    fig.text(.08, .043, labels.get("note2", "Multipliers = GPU mean / ActiveN mean. "
             "GPU wall time includes normalization, transfers and retry handling. Lower is faster."),
             fontsize=10, color="#526174")
    # Changing the display scale does not change statistics or KDE data.
    for extension in ("png", "svg"):
        metadata = {"Date": None} if extension == "svg" else None
        fig.savefig(output / f"{basename}.{extension}", dpi=220,
                    facecolor="white", metadata=metadata)
    plt.close(fig)


def write_tables(report, output):
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "device", "hardware_samples", "gpu_repetitions",
                         "mean_us", "mean_sample_speedup", "ratio_of_mean_durations"])
        for group, data in report["groups"].items():
            hw = data["hardware"]
            writer.writerow([group, "ActiveN", hw["samples"], "", hw["mean_us"], 1, 1])
            for gpu, row in data["gpus"].items():
                writer.writerow([group, LABELS[gpu], hw["samples"], row["repetitions"],
                                 row["mean_us"], row["mean_sample_speedup"], row["ratio_of_mean_durations"]])
    lines = ["# Dense MNIST ActiveN comparison", "", "![Violin plot](mnist_dense_violin.png)", "",
             "ActiveN violins include vertical 25th–75th percentile "
             "bars and arithmetic-mean diamonds; GPU bars show mean durations. "
             "Each plotted multiplier is the GPU mean duration divided by the ActiveN mean for that group "
             "(the ratio of means, distinct from the mean of individual speedup ratios below).", "",
             "Mean of individual speedup ratios, with each hardware sample and each GPU repetition "
             "equally weighted within its rule:", "",
             "| GPU | 1-trace | 2-trace | 3-trace | Combined |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for gpu in (*GPUS, "all_gpus"):
        values = [report["groups"][g]["all_gpus"] if gpu == "all_gpus" else
                  report["groups"][g]["gpus"][gpu] for g in GROUPS]
        label = "All GPUs" if gpu == "all_gpus" else LABELS[gpu]
        lines.append("| " + label + " | " + " | ".join(f"{v['mean_sample_speedup']:.3f}×" for v in values) + " |")
    lines += ["", "Arithmetic mean timestep durations:", "",
              "| Device | 1-trace (µs) | 2-trace (µs) | 3-trace (µs) | Combined (µs) |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for device in ("Hardware", *GPUS):
        values = [report["groups"][g]["hardware"] if device == "Hardware" else
                  report["groups"][g]["gpus"][device] for g in GROUPS]
        label = LABELS.get(device, device)
        lines.append("| " + label + " | " + " | ".join(f"{v['mean_us']:.3f}" for v in values) + " |")
    lines += ["", "Combined ratios of mean durations (a separate statistic):", "",
              "| GPU | Mean GPU duration / mean ActiveN duration |", "| --- | ---: |"]
    combined = report["groups"]["Combined"]
    for gpu in GPUS:
        lines.append(f"| {LABELS[gpu]} | {combined['gpus'][gpu]['ratio_of_mean_durations']:.3f}× |")
    lines += [f"| All GPUs | {combined['all_gpus']['ratio_of_mean_durations']:.3f}× |", "",
              "GPU durations include normalization, transfers, retry handling, presentation and rest. "
              "ActiveN durations are the supplied cycles at 1 GHz. These records do not establish "
              "identical hardware/GPU timing boundaries or paired spike trajectories.", "",
              "See [README](README.md) for formulas, schema, protocol, and reproduction commands; "
              "[manifest.json](manifest.json) identifies source artifacts and software.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", type=Path, default=ROOT / "mnist_dense_results.json")
    parser.add_argument("--gpu-baseline", type=Path,
                        default=ROOT / "genn-sweep/baseline-20260907-matrix.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--scale", choices=("log", "linear"), default="linear")
    args = parser.parse_args()
    hardware, gpus, baseline = load_inputs(args.hardware, args.gpu_baseline)
    hw, report = summarize(hardware, gpus)
    args.output.mkdir(parents=True, exist_ok=True)
    data_path = args.output / "data"
    data_path.mkdir(exist_ok=True)
    write_json(data_path / "hardware.json", hardware)
    for gpu, records in gpus.items():
        write_json(data_path / f"{gpu.lower()}.json", records)
    write_json(args.output / "summary.json", report)
    write_tables(report, args.output)
    plot(hw, report, args.output, args.scale)
    manifest = {
        "schema": "dense-mnist-hardware-comparison-v1",
        "command": [sys.executable, *sys.argv],
        "working_directory": str(Path.cwd()),
        "hardware": {"path": str(args.hardware.resolve()), "sha256": sha256(args.hardware),
                     "clock_hz": 1_000_000_000, "sample_unit": "simulation timestep",
                     "provenance": "User-supplied cycles; clock and workload supplied in conversation. "
                                   "Source/checkpoint/seed/timing boundaries absent from the JSON."},
        "gpu_baseline": {"path": str(args.gpu_baseline.resolve()), "sha256": sha256(args.gpu_baseline),
                         "input_manifest_sha256": baseline["input_manifest_sha256"],
                         "source_sha256": baseline["source_sha256"],
                         "record_location": "machines[GPU].cases[mnist_Ntrace_dense].raw_repetitions; "
                                            "extracted rows retain rule order 1,2,3 and repetition order 1..5",
                         "timing_commands": {g: [m["command"] for m in baseline["machines"][g]["timing_manifests"]]
                                             for g in GPUS}},
        "current_repository_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "current_simulation_source_matches_baseline": {p: (ROOT / p).exists() and sha256(ROOT / p) == digest
                                                       for p, digest in baseline["source_sha256"].items()},
        "script_sha256": sha256(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "plot": {"y_scale": args.scale, "kde_space": "linear microseconds", "bandwidth": "Scott",
                 "density_points": 512, "support": "observed min to max", "violin_width": "equal maximum width",
                 "center_bar": "25th to 75th percentile", "diamond": "arithmetic mean",
                 "hardware_label": "ActiveN", "absolute_gpu_multiplier": "GPU mean duration / ActiveN mean duration"},
        "export_schema": {"class": "N-trace dense", "cycles": "nanoseconds; GPU values are equivalent "
                          "1 GHz cycles, not actual GPU cycles", "presenting": "hardware boolean; GPU null = both phases"},
    }
    artifacts = [*data_path.glob("*.json"), *(args.output / n for n in (
        "summary.json", "comparison.csv", "report.md", "mnist_dense_violin.png", "mnist_dense_violin.svg"))]
    manifest["output_sha256"] = {str(p.relative_to(args.output)): sha256(p) for p in sorted(artifacts)}
    write_json(args.output / "manifest.json", manifest)
    print(f"Wrote comparison to {args.output}")
    print(json.dumps(report["groups"]["Combined"]["all_gpus"], indent=2))


if __name__ == "__main__":
    main()
