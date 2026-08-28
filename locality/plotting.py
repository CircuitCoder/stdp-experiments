from __future__ import annotations

from pathlib import Path

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_temporal(
    output: Path,
    traversal: np.ndarray,
    updates: np.ndarray,
    *,
    dt_ms: float,
    title: str,
) -> None:
    plt = _pyplot()
    traversal = np.asarray(traversal)
    updates = np.asarray(updates)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)

    visible = min(traversal.size, max(1, round(200.0 / dt_ms)))
    time = np.arange(visible) * dt_ms
    axes[0].plot(time, traversal[:visible], color="#2463a6", linewidth=0.8, label="all traversals")
    axes[0].plot(time, updates[:visible], color="#b33a3a", linewidth=0.8, label="weight updates")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_ylabel(f"Synapse events per {dt_ms:g} ms")
    axes[0].legend(frameon=False)
    axes[0].set_title("First 200 ms")

    maximum = max(int(traversal.max(initial=0)), int(updates.max(initial=0)))
    bins = np.linspace(0, maximum + 1, 61) if maximum else np.arange(2)
    axes[1].hist(traversal, bins=bins, color="#2463a6", alpha=0.65, label="all traversals")
    axes[1].hist(updates, bins=bins, color="#b33a3a", alpha=0.65, label="weight updates")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(f"Synapse events per {dt_ms:g} ms")
    axes[1].set_ylabel("Number of timesteps")
    axes[1].set_title("Temporal frequency histogram")
    axes[1].legend(frameon=False)
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mnist_spatial(output: Path, arrays: dict[str, np.ndarray], *, title: str) -> None:
    plt = _pyplot()
    input_counts = np.asarray(arrays["input_counts"])
    edge_frequency = np.asarray(arrays["edge_frequency"])
    exc_counts = np.asarray(arrays["exc_counts"])
    exc_strength = np.asarray(arrays["exc_strength"])
    abs_delta = np.asarray(arrays["absolute_net_weight_change"])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    image = axes[0, 0].imshow(input_counts.reshape(28, 28), cmap="magma")
    axes[0, 0].set_title("Input spike count")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    fig.colorbar(image, ax=axes[0, 0], shrink=0.8)

    axes[0, 1].hist(edge_frequency, bins=min(60, max(2, int(edge_frequency.max(initial=0)) + 1)), color="#2d7f5e")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Weight-update events per synapse")
    axes[0, 1].set_ylabel("Synapses")
    axes[0, 1].set_title("Per-synapse update frequency")

    axes[1, 0].scatter(exc_counts, exc_strength, s=10, alpha=0.65, color="#315b8a")
    axes[1, 0].set_xlabel("Excitatory spikes")
    axes[1, 0].set_ylabel("Incident update events")
    axes[1, 0].set_title("Update graph vs firing")

    delta_image = axes[1, 1].imshow(abs_delta, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[1, 1].set_xlabel("Excitatory neuron")
    axes[1, 1].set_ylabel("Input pixel")
    axes[1, 1].set_title("Accumulated |net STDP change|")
    fig.colorbar(delta_image, ax=axes[1, 1], shrink=0.8)
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_brunel_spatial(output: Path, arrays: dict[str, np.ndarray], *, title: str) -> None:
    plt = _pyplot()
    histogram = np.asarray(arrays["edge_frequency_histogram"])
    firing = np.asarray(arrays["firing_counts"])
    out_strength = np.asarray(arrays["out_strength"])
    rank_matrix = np.asarray(arrays["rank_matrix"], dtype=np.float64)
    sample_frequency = np.asarray(arrays["sample_frequency"])
    sample_delta = np.asarray(arrays["sample_abs_weight_change"])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    axes[0, 0].bar(np.arange(histogram.size), histogram, width=0.9, color="#2d7f5e")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Weight-update events per E-E synapse")
    axes[0, 0].set_ylabel("Synapses")
    axes[0, 0].set_title("Per-synapse update frequency")

    axes[0, 1].scatter(firing, out_strength, s=7, alpha=0.45, color="#315b8a")
    axes[0, 1].set_xlabel("Excitatory spikes")
    axes[0, 1].set_ylabel("Outgoing update events")
    axes[0, 1].set_title("Update graph vs firing")

    rank_fraction = rank_matrix / max(1.0, rank_matrix.sum())
    heatmap = axes[1, 0].imshow(np.log10(rank_fraction + 1.0e-12), origin="lower", cmap="magma")
    axes[1, 0].set_xlabel("Postsynaptic firing-rank block")
    axes[1, 0].set_ylabel("Presynaptic firing-rank block")
    axes[1, 0].set_title("Update mass ordered by firing rate")
    fig.colorbar(heatmap, ax=axes[1, 0], shrink=0.8, label="log10 mass fraction")

    axes[1, 1].scatter(sample_frequency, sample_delta, s=6, alpha=0.3, color="#a13c43")
    axes[1, 1].set_xlabel("Update events on sampled synapse")
    axes[1, 1].set_ylabel("|final - initial weight| (pA)")
    axes[1, 1].set_title("Frequency vs net plastic change")
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)
