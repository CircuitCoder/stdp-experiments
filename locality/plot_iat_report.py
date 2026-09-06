from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


CASES = {
    "additive": ("brunel_additive", "#2463a6"),
    "morrison": ("brunel_morrison", "#b33a3a"),
}
DT_MS = 0.1


def dense_histogram(summary: dict[str, Any]) -> np.ndarray:
    values = np.asarray(summary["histogram_values_ticks"], dtype=np.int64)
    counts = np.asarray(summary["histogram_counts"], dtype=np.uint64)
    if values.size == 0:
        return np.zeros(1, dtype=np.uint64)
    histogram = np.zeros(int(values[-1]) + 1, dtype=np.uint64)
    histogram[values] = counts
    return histogram


def aggregate_histogram(histogram: np.ndarray, ticks_per_bin: int = 10) -> np.ndarray:
    values = np.asarray(histogram, dtype=np.uint64)
    bins = (values.size + ticks_per_bin - 1) // ticks_per_bin
    padded = np.zeros(bins * ticks_per_bin, dtype=np.uint64)
    padded[: values.size] = values
    return padded.reshape(bins, ticks_per_bin).sum(axis=1, dtype=np.uint64)


def probability_curve(histogram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aggregate = aggregate_histogram(histogram)
    probability = aggregate.astype(np.float64) / aggregate.sum(dtype=np.uint64)
    x_ms = (np.arange(aggregate.size, dtype=np.float64) + 0.5)
    keep = probability > 0.0
    return x_ms[keep], probability[keep]


def survival_curve(histogram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(histogram, dtype=np.uint64)
    survival = np.cumsum(counts[::-1], dtype=np.uint64)[::-1].astype(np.float64)
    survival /= survival[0]
    x_ms = np.arange(counts.size, dtype=np.float64) * DT_MS
    keep = (x_ms > 0.0) & (survival > 0.0)
    return x_ms[keep], survival[keep]


def load_summaries(run: Path) -> dict[str, dict[str, Any]]:
    return {
        label: json.loads((run / case / "summary.json").read_text())
        for label, (case, _) in CASES.items()
    }


def plot_report(run: Path, output: Path) -> None:
    summaries = load_summaries(run)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)

    for label, (_, color) in CASES.items():
        intervals = summaries[label]["temporal"]["interval_histograms"]
        for event_class, line_style in (("pre", "-"), ("post", "--")):
            histogram = dense_histogram(
                intervals[f"ee_{event_class}synaptic_per_synapse"]
            )
            x_ms, probability = probability_curve(histogram)
            axes[0, 0].plot(
                x_ms,
                probability,
                color=color,
                linestyle=line_style,
                linewidth=1.35,
                label=f"{label} {event_class}",
            )
            x_ms, survival = survival_curve(histogram)
            axes[0, 1].plot(
                x_ms,
                survival,
                color=color,
                linestyle=line_style,
                linewidth=1.35,
                label=f"{label} {event_class}",
            )

    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("IAT (ms)")
    axes[0, 0].set_ylabel("Probability mass per 1 ms bin")
    axes[0, 0].set_title("Full 10 s distribution")
    axes[0, 0].legend(frameon=False, ncol=2)

    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("IAT threshold (ms)")
    axes[0, 1].set_ylabel("P(IAT >= threshold)")
    axes[0, 1].set_title("Full 10 s survival function")
    axes[0, 1].legend(frameon=False, ncol=2)

    for axis, (label, (_, color)) in zip(axes[1], CASES.items()):
        summary = summaries[label]
        windows = summary["temporal"]["iat_windows_1000_ms"]
        curves = (
            (
                dense_histogram(
                    windows[0]["interval_histograms"][
                        "ee_presynaptic_per_synapse"
                    ]
                ),
                "first 1 s",
                "-",
            ),
            (
                dense_histogram(
                    windows[-1]["interval_histograms"][
                        "ee_presynaptic_per_synapse"
                    ]
                ),
                "last 1 s",
                ":",
            ),
        )
        for histogram, curve_label, line_style in curves:
            x_ms, probability = probability_curve(histogram)
            axis.plot(
                x_ms,
                probability,
                color=color,
                linestyle=line_style,
                linewidth=1.5,
                label=curve_label,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Presynaptic IAT (ms)")
        axis.set_ylabel("Probability mass per 1 ms bin")
        axis.set_title(f"{label.capitalize()}: time-window comparison")
        axis.legend(frameon=False)

    fig.suptitle("GeNN Brunel E-E update inter-arrival distributions")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot report-ready Brunel per-synapse IAT distributions"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_report(args.run, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
