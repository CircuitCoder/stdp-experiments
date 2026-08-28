from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ticks_for_events(
    times_ms: np.ndarray,
    *,
    dt_ms: float,
    start_ms: float,
    duration_ticks: int,
    delay_ms: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return event indices and the mask selecting events inside a window."""
    relative = (np.asarray(times_ms, dtype=np.float64) + delay_ms - start_ms) / dt_ms
    ticks = np.rint(relative).astype(np.int64)
    keep = (ticks >= 0) & (ticks < duration_ticks)
    return ticks[keep], keep


def event_series(
    times_ms: np.ndarray,
    ids: np.ndarray,
    endpoint_degrees: np.ndarray,
    *,
    dt_ms: float,
    start_ms: float,
    duration_ticks: int,
    delay_ms: float = 0.0,
) -> np.ndarray:
    ticks, keep = ticks_for_events(
        times_ms,
        dt_ms=dt_ms,
        start_ms=start_ms,
        duration_ticks=duration_ticks,
        delay_ms=delay_ms,
    )
    weights = np.asarray(endpoint_degrees, dtype=np.uint64)[np.asarray(ids)[keep]]
    return np.bincount(ticks, weights=weights, minlength=duration_ticks).astype(np.uint64)


def endpoint_counts(
    times_ms: np.ndarray,
    ids: np.ndarray,
    n_neurons: int,
    *,
    start_ms: float,
    end_ms: float,
    delay_ms: float = 0.0,
) -> np.ndarray:
    shifted = np.asarray(times_ms, dtype=np.float64) + delay_ms
    keep = (shifted >= start_ms) & (shifted < end_ms)
    return np.bincount(np.asarray(ids, dtype=np.int64)[keep], minlength=n_neurons).astype(
        np.uint32
    )


def aggregate_series(series: np.ndarray, width_ticks: int) -> np.ndarray:
    if width_ticks <= 0:
        raise ValueError("width_ticks must be positive")
    values = np.asarray(series, dtype=np.uint64)
    padding = (-values.size) % width_ticks
    if padding:
        values = np.pad(values, (0, padding))
    return values.reshape(-1, width_ticks).sum(axis=1, dtype=np.uint64)


def series_summary(series: np.ndarray) -> dict[str, Any]:
    values = np.asarray(series, dtype=np.float64)
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 0.0
    if values.size > 2 and std > 0.0:
        lag1 = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    else:
        lag1 = None
    unique, counts = np.unique(values.astype(np.uint64), return_counts=True)
    return {
        "bins": int(values.size),
        "total": int(values.sum(dtype=np.float64)),
        "mean": mean,
        "std": std,
        "coefficient_of_variation": std / mean if mean > 0.0 else None,
        "fano_factor": float(values.var() / mean) if mean > 0.0 else None,
        "idle_fraction": float(np.mean(values == 0.0)) if values.size else None,
        "p50": float(np.quantile(values, 0.50)) if values.size else None,
        "p90": float(np.quantile(values, 0.90)) if values.size else None,
        "p99": float(np.quantile(values, 0.99)) if values.size else None,
        "maximum": int(values.max()) if values.size else 0,
        "lag1_autocorrelation": lag1,
        "histogram_values": unique.astype(np.uint64).tolist(),
        "histogram_counts": counts.astype(np.uint64).tolist(),
    }


def histogram_concentration(histogram: np.ndarray) -> dict[str, float | None]:
    counts = np.asarray(histogram, dtype=np.uint64)
    n = int(counts.sum(dtype=np.uint64))
    mass = int(np.dot(np.arange(counts.size, dtype=np.uint64), counts))
    if n == 0 or mass == 0:
        return {"gini": None, "top_1pct_mass": None, "top_10pct_mass": None}

    weighted_rank_sum = 0.0
    rank = 1
    for value, count in enumerate(counts):
        if count:
            last = rank + int(count) - 1
            rank_sum = (rank + last) * int(count) / 2.0
            weighted_rank_sum += value * rank_sum
            rank = last + 1
    gini = (2.0 * weighted_rank_sum) / (n * mass) - (n + 1.0) / n

    def top_share(fraction: float) -> float:
        remaining = max(1, math.ceil(n * fraction))
        selected_mass = 0
        for value in range(counts.size - 1, -1, -1):
            take = min(remaining, int(counts[value]))
            selected_mass += take * value
            remaining -= take
            if remaining == 0:
                break
        return selected_mass / mass

    return {
        "gini": float(gini),
        "top_1pct_mass": float(top_share(0.01)),
        "top_10pct_mass": float(top_share(0.10)),
    }


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    x_values = np.asarray(x)
    y_values = np.asarray(y)
    if x_values.size < 2 or np.all(x_values == x_values[0]) or np.all(y_values == y_values[0]):
        return None
    value = float(spearmanr(x_values, y_values).statistic)
    return value if np.isfinite(value) else None


def morans_i_4_neighbor(image: np.ndarray) -> float | None:
    values = np.asarray(image, dtype=np.float64)
    centered = values - values.mean()
    denominator = float(np.sum(centered * centered))
    if denominator == 0.0:
        return None
    cross = float(np.sum(centered[:, :-1] * centered[:, 1:]))
    cross += float(np.sum(centered[:-1, :] * centered[1:, :]))
    directed_weight = 2 * (values.shape[0] * (values.shape[1] - 1))
    directed_weight += 2 * ((values.shape[0] - 1) * values.shape[1])
    return float(values.size / directed_weight * (2.0 * cross) / denominator)


def firing_communities(
    times_ms: np.ndarray,
    ids: np.ndarray,
    n_neurons: int,
    *,
    start_ms: float,
    end_ms: float,
    bin_ms: float = 20.0,
    requested_communities: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.cluster.vq import kmeans2

    n_bins = max(1, math.ceil((end_ms - start_ms) / bin_ms))
    times = np.asarray(times_ms, dtype=np.float64)
    senders = np.asarray(ids, dtype=np.int64)
    keep = (times >= start_ms) & (times < end_ms)
    bin_ids = np.floor((times[keep] - start_ms) / bin_ms).astype(np.int64)
    activity = np.zeros((n_neurons, n_bins), dtype=np.float32)
    np.add.at(activity, (senders[keep], bin_ids), 1.0)
    counts = activity.sum(axis=1, dtype=np.float64).astype(np.uint32)

    centered = activity - activity.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    active = norms > 0.0
    features = centered[active] / norms[active, None]
    labels = np.full(n_neurons, -1, dtype=np.int16)
    k = min(requested_communities, features.shape[0])
    if k > 1:
        _, active_labels = kmeans2(features, k, minit="++", iter=30, seed=0)
        labels[active] = active_labels.astype(np.int16)
    elif k == 1:
        labels[active] = 0
    if np.any(~active):
        labels[~active] = k
    return labels, counts


def _top_mass_fraction(histogram: np.ndarray, fraction: float) -> float | None:
    counts = np.asarray(histogram, dtype=np.uint64)
    total_mass = int(np.dot(np.arange(counts.size, dtype=np.uint64), counts))
    if total_mass == 0:
        return None
    target = total_mass * fraction
    edge_count = 0
    accumulated = 0
    for value in range(counts.size - 1, -1, -1):
        count = int(counts[value])
        if value == 0 or count == 0:
            continue
        needed = math.ceil((target - accumulated) / value)
        take = min(count, max(0, needed))
        accumulated += take * value
        edge_count += take
        if accumulated >= target:
            break
    return edge_count / int(counts.sum(dtype=np.uint64))


def scan_brunel_graph(
    row_lengths: np.ndarray,
    raw_post_indices: np.ndarray,
    max_connections: int,
    pre_counts: np.ndarray,
    post_counts: np.ndarray,
    communities: np.ndarray,
    *,
    rank_blocks: int = 48,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    row_lengths = np.asarray(row_lengths, dtype=np.int64)
    raw_post_indices = np.asarray(raw_post_indices)
    pre_counts = np.asarray(pre_counts, dtype=np.uint32)
    post_counts = np.asarray(post_counts, dtype=np.uint32)
    communities = np.asarray(communities, dtype=np.int16)
    n = row_lengths.size
    if pre_counts.shape != (n,) or post_counts.shape != (n,):
        raise ValueError("endpoint count shape does not match connectivity")

    n_communities = int(communities.max()) + 1
    histogram = np.zeros(int(pre_counts.max()) + int(post_counts.max()) + 1, dtype=np.uint64)
    out_strength = np.zeros(n, dtype=np.uint64)
    in_strength = np.zeros(n, dtype=np.uint64)
    community_matrix = np.zeros((n_communities, n_communities), dtype=np.uint64)
    order = np.argsort(pre_counts, kind="stable")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    rank_group = np.minimum(rank * rank_blocks // n, rank_blocks - 1)
    rank_matrix = np.zeros((rank_blocks, rank_blocks), dtype=np.uint64)

    top_count = max(1, math.ceil(n * 0.10))
    top = np.zeros(n, dtype=bool)
    top[order[-top_count:]] = True
    top_incident_mass = 0

    for pre, length in enumerate(row_lengths):
        posts = raw_post_indices[
            pre * max_connections : pre * max_connections + int(length)
        ].astype(np.int64, copy=False)
        values = pre_counts[pre].astype(np.uint64) + post_counts[posts].astype(np.uint64)
        histogram += np.bincount(values, minlength=histogram.size).astype(np.uint64)
        row_mass = int(values.sum(dtype=np.uint64))
        out_strength[pre] = row_mass
        np.add.at(in_strength, posts, values)
        community_matrix[communities[pre]] += np.bincount(
            communities[posts], weights=values, minlength=n_communities
        ).astype(np.uint64)
        rank_matrix[rank_group[pre]] += np.bincount(
            rank_group[posts], weights=values, minlength=rank_blocks
        ).astype(np.uint64)
        if top[pre]:
            top_incident_mass += row_mass
        else:
            top_incident_mass += int(values[top[posts]].sum(dtype=np.uint64))

    total_mass = int(out_strength.sum(dtype=np.uint64))
    within_mass = int(np.trace(community_matrix, dtype=np.uint64))
    community_out = community_matrix.sum(axis=1, dtype=np.uint64)
    community_in = community_matrix.sum(axis=0, dtype=np.uint64)
    if total_mass:
        expected_within = float(np.dot(community_out.astype(np.float64), community_in)) / total_mass
        modularity = (within_mass - expected_within) / total_mass
    else:
        expected_within = 0.0
        modularity = None

    summary = {
        "structural_edges_including_multapses": int(row_lengths.sum()),
        "total_update_attempts": total_mass,
        "edge_frequency_histogram": histogram.tolist(),
        "edge_frequency_concentration": histogram_concentration(histogram),
        "edge_fraction_for_50pct_update_mass": _top_mass_fraction(histogram, 0.50),
        "firing_vs_out_strength_spearman": spearman(pre_counts, out_strength),
        "firing_vs_in_strength_spearman": spearman(post_counts, in_strength),
        "community_count": n_communities,
        "within_community_mass_fraction": within_mass / total_mass if total_mass else None,
        "configuration_null_within_mass_fraction": expected_within / total_mass
        if total_mass
        else None,
        "weighted_directed_modularity": float(modularity) if modularity is not None else None,
        "top_10pct_firing_neuron_incident_mass_fraction": top_incident_mass / total_mass
        if total_mass
        else None,
    }
    arrays = {
        "out_strength": out_strength,
        "in_strength": in_strength,
        "community_matrix": community_matrix,
        "rank_matrix": rank_matrix,
        "rank_group": rank_group.astype(np.int16),
    }
    return summary, arrays


def sample_sparse_state(synapses: Any, ordinals: np.ndarray) -> dict[str, np.ndarray]:
    row_lengths = np.asarray(synapses._row_lengths.view, dtype=np.int64)
    cumulative = np.cumsum(row_lengths, dtype=np.int64)
    ordinals = np.asarray(ordinals, dtype=np.int64)
    pre = np.searchsorted(cumulative, ordinals, side="right")
    previous = np.where(pre == 0, 0, cumulative[pre - 1])
    offset = ordinals - previous
    raw = pre * int(synapses.max_connections) + offset
    return {
        "ordinal": ordinals.copy(),
        "pre": pre.astype(np.uint32),
        "post": np.asarray(synapses._ind.view[raw], dtype=np.uint32).copy(),
        "weight": np.asarray(synapses.vars["g"]._view[raw], dtype=np.float64).copy(),
    }


def choose_edge_sample(total_edges: int, requested: int) -> np.ndarray:
    count = min(total_edges, requested)
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    return np.linspace(0, total_edges - 1, count, dtype=np.int64)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="ascii") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def interval_gaps(event_ticks: Iterable[np.ndarray]) -> np.ndarray:
    gaps = []
    for ticks in event_ticks:
        values = np.asarray(ticks, dtype=np.int64)
        if values.size > 1:
            gaps.append(np.diff(np.sort(values)))
    return np.concatenate(gaps) if gaps else np.empty(0, dtype=np.int64)
