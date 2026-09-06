#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parent
CHECKPOINTS = REPOSITORY / "genn-sweep" / "checkpoints"
sys.path.insert(0, str(REPOSITORY / "reimpl"))
sys.path.insert(0, str(REPOSITORY / "brunel"))
sys.path.insert(0, str(REPOSITORY))

from brunel.ports.common import (  # noqa: E402
    DT_MS as BRUNEL_DT_MS,
    make_genn_default_model,
)
from brunel.ports.genn_port import GeNNBrunel  # noqa: E402
from locality.analysis import (  # noqa: E402
    aggregate_series,
    choose_edge_sample,
    count_distribution_summary,
    endpoint_counts,
    event_series,
    firing_communities,
    histogram_concentration,
    id_gap_histogram_summary,
    interval_histogram_summary,
    interval_gaps,
    morans_i_4_neighbor,
    sample_sparse_state,
    scan_brunel_graph,
    series_summary,
    spearman,
    ticks_for_events,
    weighted_interval_histogram,
    within_tick_id_gap_histogram,
    write_json,
)
from locality.plotting import (  # noqa: E402
    plot_brunel_intervals,
    plot_brunel_spatial,
    plot_mnist_spatial,
    plot_temporal,
)
from reimpl.backends.genn_backend import GeNNNetwork  # noqa: E402
from reimpl.zd3.constants import MODEL  # noqa: E402
from reimpl.zd3.io import load_checkpoint, load_mnist, sha256_file  # noqa: E402
from reimpl.zd3.variants import get_variant, validate_checkpoint_topology  # noqa: E402


@dataclass(frozen=True)
class MnistCase:
    variant: str
    parallelism: str
    threads_per_spike: int
    checkpoint: Path


MNIST_CASES = {
    "mnist_triplet_dense": MnistCase(
        "triplet-dense", "postsynaptic", 1, CHECKPOINTS / "mnist_triplet_dense_010000.npz"
    ),
    "mnist_one_trace_dense": MnistCase(
        "one-trace-dense",
        "postsynaptic",
        1,
        CHECKPOINTS / "mnist_one_trace_dense_010000.npz",
    ),
    "mnist_one_trace_sparse_0125": MnistCase(
        "one-trace-bernoulli-0125",
        "presynaptic",
        32,
        CHECKPOINTS / "mnist_one_trace_sparse_0125_010000.npz",
    ),
}
BRUNEL_CASES = {"brunel_additive": "additive", "brunel_morrison": "morrison"}
ALL_CASES = tuple(MNIST_CASES) + tuple(BRUNEL_CASES)


def configure_cuda_compiler() -> None:
    flags = os.environ.get("NVCC_PREPEND_FLAGS", "").split()
    cuda_path = os.environ.get("CUDA_PATH")
    cuda_host_cxx = os.environ.get("CUDAHOSTCXX")
    required = []
    if cuda_path:
        required.append(f"-I{cuda_path}/include")
    if cuda_host_cxx:
        required.append(f"-ccbin={cuda_host_cxx}")
    os.environ["NVCC_PREPEND_FLAGS"] = " ".join(required + flags)


def source_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "status_short": status}


def base_manifest(args: argparse.Namespace, case_name: str) -> dict[str, Any]:
    import pygenn

    return {
        "schema": "synapse-locality-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case": case_name,
        "command": sys.argv,
        "source": source_state(),
        "host": {
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "genn_version": pygenn.__version__,
        "precision": "float",
        "measurement_scope": {
            "synapse_traversal": "execution of a synaptic pre or post event path",
            "weight_update_attempt": "execution of code assigning the plastic weight",
            "net_weight_change": "absolute before/after difference; cancelling events are not counted separately",
            "population_interarrival": "gaps between consecutive spikes in one population, including zero-tick simultaneous gaps",
            "per_neuron_interspike": "gaps between consecutive spikes from the same neuron, pooled by population",
            "per_synapse_pre_interval": "source-neuron interspike gaps weighted by each source's E-E outdegree",
            "per_synapse_post_interval": "target-neuron interspike gaps weighted by each target's E-E indegree",
            "windowed_interval": "both events fall within the named half-open window; intervals crossing a window boundary are excluded",
            "within_tick_neuron_id_gap": "absolute ID distance between consecutive entries in GeNN's recorded spike-list order, excluding cross-tick pairs",
        },
    }


def events_by_neuron(
    times_ms: np.ndarray,
    ids: np.ndarray,
    n_neurons: int,
    *,
    dt_ms: float,
    start_ms: float,
    duration_ticks: int,
    delay_ms: float = 0.0,
) -> list[np.ndarray]:
    ticks, keep = ticks_for_events(
        times_ms,
        dt_ms=dt_ms,
        start_ms=start_ms,
        duration_ticks=duration_ticks,
        delay_ms=delay_ms,
    )
    kept_ids = np.asarray(ids, dtype=np.int64)[keep]
    order = np.argsort(kept_ids, kind="stable")
    sorted_ids = kept_ids[order]
    sorted_ticks = ticks[order]
    starts = np.searchsorted(sorted_ids, np.arange(n_neurons), side="left")
    ends = np.searchsorted(sorted_ids, np.arange(n_neurons), side="right")
    return [np.sort(sorted_ticks[start:end]) for start, end in zip(starts, ends)]


def temporal_summaries(series: np.ndarray, dt_ms: float) -> dict[str, Any]:
    result = {f"{dt_ms:g}_ms": series_summary(series)}
    for width_ms in (1.0, 10.0, 100.0):
        width_ticks = max(1, round(width_ms / dt_ms))
        if width_ticks > 1:
            result[f"{width_ms:g}_ms"] = series_summary(aggregate_series(series, width_ticks))
    return result


def run_mnist(args: argparse.Namespace, case_name: str, output: Path) -> None:
    case = MNIST_CASES[case_name]
    variant = get_variant(case.variant)
    checkpoint = load_checkpoint(case.checkpoint)
    mask = validate_checkpoint_topology(checkpoint.weights, variant)
    data = load_mnist(args.data_path, "train")
    recording_steps = MODEL.attempt_ticks
    manifest = {
        **base_manifest(args, case_name),
        "dataset": "MNIST train",
        "data_path": str(args.data_path.resolve()),
        "checkpoint": str(case.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(case.checkpoint),
        "accepted_sample_start": checkpoint.accepted_samples,
        "requested_accepted_samples": args.mnist_samples,
        "rng_seed": args.mnist_seed,
        "variant": variant.as_dict(),
        "model": MODEL.as_dict(),
        "protocol": "plastic training; every attempt includes 350 ms stimulus and 150 ms rest",
    }
    write_json(output / "manifest.json", manifest)

    network = GeNNNetwork(
        weights=checkpoint.weights.copy(),
        theta_mv=checkpoint.theta_mv.copy(),
        plasticity=True,
        inhibition=MODEL.train_inhibition,
        seed=args.mnist_seed,
        backend="cuda",
        build_path=output / "build",
        variant=variant,
        structural_mask=mask,
        precision="float",
        parallelism=case.parallelism,
        num_threads_per_spike=case.threads_per_spike,
        timing_enabled=False,
        reuse_build=None,
        record_spikes=True,
        recording_steps=recording_steps,
    )
    absolute_net_change = np.zeros_like(checkpoint.weights, dtype=np.float64)
    signed_net_change = np.zeros_like(checkpoint.weights, dtype=np.float64)
    changed_attempts = np.zeros_like(checkpoint.weights, dtype=np.uint16)
    attempts: list[tuple[int, int, float, int, int]] = []
    spike_chunks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "input": [],
        "excitatory": [],
        "inhibitory": [],
    }
    accepted = 0
    try:
        while accepted < args.mnist_samples:
            intensity = MODEL.initial_intensity
            while True:
                if len(attempts) >= args.mnist_max_attempts:
                    raise RuntimeError("MNIST attempt count exceeded recording capacity")
                network.normalize(validate=False)
                before = network.weights()
                sample = checkpoint.accepted_samples + accepted
                network.set_image(data.images[sample % len(data.images)], intensity)
                counts = network.run_stimulus()
                stimulus_spikes = int(counts.sum())
                retry = stimulus_spikes < MODEL.minimum_exc_spikes
                network.run_rest()
                after = network.weights()
                recorded = network.recorded_spikes()
                for population, events in recorded.items():
                    spike_chunks[population].append(events)
                delta = after - before
                absolute_net_change += np.abs(delta)
                signed_net_change += delta
                changed_attempts += (delta != 0.0).astype(np.uint16)
                attempts.append((sample, int(data.labels[sample % len(data.labels)]), intensity, stimulus_spikes, int(retry)))
                if retry:
                    intensity += MODEL.intensity_increment
                else:
                    accepted += 1
                    break

        final_theta = network.theta_mv()
    finally:
        network.close()

    attempt_array = np.asarray(
        attempts,
        dtype=[
            ("sample", np.int64),
            ("label", np.int16),
            ("intensity", np.float32),
            ("stimulus_exc_spikes", np.int32),
            ("retry", np.int8),
        ],
    )
    spikes = {
        population: (
            np.concatenate([events[0] for events in chunks]),
            np.concatenate([events[1] for events in chunks]),
        )
        for population, chunks in spike_chunks.items()
    }
    duration_ticks = len(attempts) * MODEL.attempt_ticks
    input_times, input_ids = spikes["input"]
    exc_times, exc_ids = spikes["excitatory"]
    input_outdegree = mask.sum(axis=1, dtype=np.uint32)
    exc_indegree = mask.sum(axis=0, dtype=np.uint32)
    pre_series = event_series(
        input_times,
        input_ids,
        input_outdegree,
        dt_ms=MODEL.dt_ms,
        start_ms=0.0,
        duration_ticks=duration_ticks,
    )
    post_series = event_series(
        exc_times,
        exc_ids,
        exc_indegree,
        dt_ms=MODEL.dt_ms,
        start_ms=0.0,
        duration_ticks=duration_ticks,
    )
    traversal_series = pre_series + post_series
    update_series = traversal_series if variant.learning_rule == "three-trace" else post_series
    end_ms = duration_ticks * MODEL.dt_ms
    input_counts = endpoint_counts(input_times, input_ids, MODEL.n_input, start_ms=0.0, end_ms=end_ms)
    exc_counts = endpoint_counts(exc_times, exc_ids, MODEL.n_exc, start_ms=0.0, end_ms=end_ms)
    if variant.learning_rule == "three-trace":
        update_matrix = input_counts[:, None].astype(np.uint64) + exc_counts[None, :].astype(np.uint64)
    else:
        update_matrix = np.broadcast_to(exc_counts[None, :], mask.shape).astype(np.uint64)
    update_matrix = np.where(mask, update_matrix, 0)
    edge_frequency = update_matrix[mask]
    edge_histogram = np.bincount(edge_frequency.astype(np.int64)).astype(np.uint64)
    exc_strength = update_matrix.sum(axis=0, dtype=np.uint64)

    pre_events = events_by_neuron(
        input_times,
        input_ids,
        MODEL.n_input,
        dt_ms=MODEL.dt_ms,
        start_ms=0.0,
        duration_ticks=duration_ticks,
    )
    post_events = events_by_neuron(
        exc_times,
        exc_ids,
        MODEL.n_exc,
        dt_ms=MODEL.dt_ms,
        start_ms=0.0,
        duration_ticks=duration_ticks,
    )
    edge_pre, edge_post = np.nonzero(mask)
    sample_ordinals = choose_edge_sample(edge_pre.size, args.edge_sample_size)
    sampled_event_ticks = []
    for ordinal in sample_ordinals[: args.interval_edge_sample_size]:
        if variant.learning_rule == "three-trace":
            events = np.concatenate((pre_events[edge_pre[ordinal]], post_events[edge_post[ordinal]]))
        else:
            events = post_events[edge_post[ordinal]]
        sampled_event_ticks.append(events)
    gaps = interval_gaps(sampled_event_ticks)

    summary = {
        "case": case_name,
        "accepted_samples": accepted,
        "attempts": len(attempts),
        "retries": int(attempt_array["retry"].sum()),
        "simulated_ms": end_ms,
        "structural_synapses": int(mask.sum()),
        "input_spikes": int(input_counts.sum(dtype=np.uint64)),
        "excitatory_spikes": int(exc_counts.sum(dtype=np.uint64)),
        "temporal": {
            "all_synapse_traversals": temporal_summaries(traversal_series, MODEL.dt_ms),
            "weight_update_attempts": temporal_summaries(update_series, MODEL.dt_ms),
        },
        "spatial": {
            "edge_frequency_histogram": edge_histogram.tolist(),
            "edge_frequency_concentration": histogram_concentration(edge_histogram),
            "input_spike_morans_i_4_neighbor": morans_i_4_neighbor(input_counts.reshape(28, 28)),
            "excitatory_firing_vs_update_strength_spearman": spearman(exc_counts, exc_strength),
            "synapses_with_nonzero_net_change": int(np.count_nonzero(absolute_net_change[mask])),
            "mean_absolute_net_change": float(absolute_net_change[mask].mean()),
            "inter_update_gap_ticks": series_summary(gaps),
        },
        "final_theta_mv": {
            "minimum": float(final_theta.min()),
            "mean": float(final_theta.mean()),
            "maximum": float(final_theta.max()),
        },
    }
    write_json(output / "summary.json", summary)
    arrays = {
        "traversal_series": traversal_series,
        "update_series": update_series,
        "pre_series": pre_series,
        "post_series": post_series,
        "input_counts": input_counts,
        "exc_counts": exc_counts,
        "exc_strength": exc_strength,
        "edge_frequency": edge_frequency,
        "edge_frequency_histogram": edge_histogram,
        "absolute_net_weight_change": absolute_net_change.astype(np.float32),
        "signed_net_weight_change": signed_net_change.astype(np.float32),
        "changed_attempts": changed_attempts,
        "structural_mask": mask,
        "attempts": attempt_array,
        "inter_update_gap_ticks": gaps,
    }
    np.savez_compressed(output / "locality.npz", **arrays)
    plot_temporal(
        output / "temporal.png",
        traversal_series,
        update_series,
        dt_ms=MODEL.dt_ms,
        title=case_name,
    )
    plot_mnist_spatial(output / "spatial.png", arrays, title=case_name)


def _pull_sample(network: GeNNBrunel, ordinals: np.ndarray) -> dict[str, np.ndarray]:
    network.ee.vars["g"].pull_from_device()
    return sample_sparse_state(network.ee, ordinals)


def brunel_window(
    network: GeNNBrunel,
    times: np.ndarray,
    ids: np.ndarray,
    *,
    start_ms: float,
    duration_ms: float,
    delay_ms: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n = network.ee.src.num_neurons
    end_ms = start_ms + duration_ms
    pre_counts = endpoint_counts(
        times, ids, n, start_ms=start_ms, end_ms=end_ms, delay_ms=delay_ms
    )
    post_counts = endpoint_counts(times, ids, n, start_ms=start_ms, end_ms=end_ms)
    communities, firing_counts = firing_communities(
        times, ids, n, start_ms=start_ms, end_ms=end_ms
    )
    graph_summary, graph_arrays = scan_brunel_graph(
        network.ee._row_lengths.view,
        network.ee._ind.view,
        network.ee.max_connections,
        pre_counts,
        post_counts,
        communities,
    )
    arrays = {
        **graph_arrays,
        "pre_counts": pre_counts,
        "post_counts": post_counts,
        "firing_counts": firing_counts,
        "communities": communities,
        "edge_frequency_histogram": np.asarray(graph_summary["edge_frequency_histogram"], dtype=np.uint64),
    }
    return graph_summary, arrays


def run_brunel(args: argparse.Namespace, case_name: str, output: Path) -> None:
    rule = BRUNEL_CASES[case_name]
    spec = make_genn_default_model(rule)
    manifest = {
        **base_manifest(args, case_name),
        "rng_seed": args.brunel_seed,
        "state_seed": args.brunel_seed,
        "model": spec.as_dict(),
        "presimulation_ms": args.brunel_presim_ms,
        "requested_simulation_ms": args.brunel_sim_ms,
        "stdp_timing": "arrival",
        "stdp_tie_order": "nest_causal_boundary",
        "protocol": "plastic simulation from a fresh seeded state",
    }
    write_json(output / "manifest.json", manifest)
    chunk_steps = round(args.brunel_chunk_ms / BRUNEL_DT_MS)
    network = GeNNBrunel(
        spec=spec,
        seed=args.brunel_seed,
        state_seed=args.brunel_seed,
        backend="cuda",
        build_path=output / "build",
        recording_steps=chunk_steps,
        precision="float",
        ee_parallelism="postsynaptic",
        ee_num_threads_per_spike=1,
        stdp_timing="arrival",
        stdp_tie_order="nest_causal_boundary",
        timing_enabled=False,
        reuse_build=None,
        record_spikes=True,
        collect_connectivity_stats=True,
    )
    total_edges = int(np.asarray(network.ee._row_lengths.view, dtype=np.uint64).sum())
    sample_ordinals = choose_edge_sample(total_edges, args.edge_sample_size)
    spike_chunks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "excitatory": [],
        "inhibitory": [],
    }
    try:
        presim_chunks = round(args.brunel_presim_ms / args.brunel_chunk_ms)
        for _ in range(presim_chunks):
            network.step(chunk_steps, synchronize=False)
            recorded = network.recorded_population_spikes()
            for population, events in recorded.items():
                spike_chunks[population].append(events)
        population_baseline = network.population_spike_counts()
        baseline_sample = _pull_sample(network, sample_ordinals)
        elapsed_ms = 0.0
        snapshots = [(0.0, baseline_sample["weight"])]
        previous_exc = population_baseline[0]
        termination = "requested_duration_completed"
        while elapsed_ms < args.brunel_sim_ms - 1.0e-12:
            chunk_ms = min(args.brunel_chunk_ms, args.brunel_sim_ms - elapsed_ms)
            network.step(round(chunk_ms / BRUNEL_DT_MS), synchronize=False)
            elapsed_ms += chunk_ms
            recorded = network.recorded_population_spikes()
            for population, events in recorded.items():
                spike_chunks[population].append(events)
            exc, _ = network.population_spike_counts()
            chunk_spikes = int((exc - previous_exc).sum(dtype=np.uint64))
            previous_exc = exc
            chunk_rate_hz = chunk_spikes / (spec.ne * chunk_ms) * 1000.0
            if abs(elapsed_ms - round(elapsed_ms / 1000.0) * 1000.0) < 1.0e-9 or elapsed_ms == args.brunel_sim_ms:
                snapshots.append((elapsed_ms, _pull_sample(network, sample_ordinals)["weight"]))
            if chunk_rate_hz >= args.brunel_abort_rate_hz:
                termination = f"chunk excitatory rate {chunk_rate_hz:.6g} Hz reached guard"
                break

        final_sample = _pull_sample(network, sample_ordinals)
        population_final = network.population_spike_counts()
        population_spikes = {
            population: (
                np.concatenate([events[0] for events in chunks]),
                np.concatenate([events[1] for events in chunks]),
            )
            for population, chunks in spike_chunks.items()
        }
        times, ids = population_spikes["excitatory"]
        inhibitory_times, inhibitory_ids = population_spikes["inhibitory"]
        actual_steps = round(elapsed_ms / BRUNEL_DT_MS)
        pre_series = event_series(
            times,
            ids,
            np.asarray(network._outdegrees["ee"], dtype=np.uint64),
            dt_ms=BRUNEL_DT_MS,
            start_ms=args.brunel_presim_ms,
            duration_ticks=actual_steps,
            delay_ms=spec.delay_ms,
        )
        post_series = event_series(
            times,
            ids,
            np.full(spec.ne, spec.ce, dtype=np.uint64),
            dt_ms=BRUNEL_DT_MS,
            start_ms=args.brunel_presim_ms,
            duration_ticks=actual_steps,
        )
        traversal_series = pre_series + post_series
        graph_summary, graph_arrays = brunel_window(
            network,
            times,
            ids,
            start_ms=args.brunel_presim_ms,
            duration_ms=elapsed_ms,
            delay_ms=spec.delay_ms,
        )
        selected_windows: dict[str, Any] = {}
        if elapsed_ms > 1000.0:
            for label, start in (
                ("first_1000_ms", args.brunel_presim_ms),
                ("last_1000_ms", args.brunel_presim_ms + elapsed_ms - 1000.0),
            ):
                window_summary, _ = brunel_window(
                    network,
                    times,
                    ids,
                    start_ms=start,
                    duration_ms=1000.0,
                    delay_ms=spec.delay_ms,
                )
                selected_windows[label] = window_summary

        sample_frequency = (
            graph_arrays["pre_counts"][baseline_sample["pre"]]
            + graph_arrays["post_counts"][baseline_sample["post"]]
        ).astype(np.uint32)
        sample_abs_delta = np.abs(final_sample["weight"] - baseline_sample["weight"])

        sampled_event_ticks = []
        pre_events = events_by_neuron(
            times,
            ids,
            spec.ne,
            dt_ms=BRUNEL_DT_MS,
            start_ms=args.brunel_presim_ms,
            duration_ticks=actual_steps,
            delay_ms=spec.delay_ms,
        )
        post_events = events_by_neuron(
            times,
            ids,
            spec.ne,
            dt_ms=BRUNEL_DT_MS,
            start_ms=args.brunel_presim_ms,
            duration_ticks=actual_steps,
        )
        inhibitory_events = events_by_neuron(
            inhibitory_times,
            inhibitory_ids,
            spec.ni,
            dt_ms=BRUNEL_DT_MS,
            start_ms=args.brunel_presim_ms,
            duration_ticks=actual_steps,
        )

        excitatory_counts = np.asarray(graph_arrays["firing_counts"], dtype=np.uint32)
        inhibitory_counts = endpoint_counts(
            inhibitory_times,
            inhibitory_ids,
            spec.ni,
            start_ms=args.brunel_presim_ms,
            end_ms=args.brunel_presim_ms + elapsed_ms,
        )
        recorded_population_totals = {
            "excitatory": int(excitatory_counts.sum(dtype=np.uint64)),
            "inhibitory": int(inhibitory_counts.sum(dtype=np.uint64)),
        }
        counter_population_totals = {
            "excitatory": int(
                (population_final[0] - population_baseline[0]).sum(dtype=np.uint64)
            ),
            "inhibitory": int(
                (population_final[1] - population_baseline[1]).sum(dtype=np.uint64)
            ),
        }
        if recorded_population_totals != counter_population_totals:
            raise RuntimeError(
                "recorded population spikes do not match device counters: "
                f"recorded={recorded_population_totals} counters={counter_population_totals}"
            )

        ee_outdegrees = np.asarray(network._outdegrees["ee"], dtype=np.uint64)
        expected_pre_updates = int(
            np.dot(excitatory_counts.astype(np.uint64), ee_outdegrees)
        )
        expected_post_updates = recorded_population_totals["excitatory"] * spec.ce
        observed_pre_updates = int(pre_series.sum(dtype=np.uint64))
        observed_post_updates = int(post_series.sum(dtype=np.uint64))
        if observed_pre_updates != expected_pre_updates:
            raise RuntimeError(
                f"presynaptic update mismatch: {observed_pre_updates} != {expected_pre_updates}"
            )
        if observed_post_updates != expected_post_updates:
            raise RuntimeError(
                f"postsynaptic update mismatch: {observed_post_updates} != {expected_post_updates}"
            )

        excitatory_ticks, excitatory_keep = ticks_for_events(
            times,
            dt_ms=BRUNEL_DT_MS,
            start_ms=args.brunel_presim_ms,
            duration_ticks=actual_steps,
        )
        inhibitory_ticks, inhibitory_keep = ticks_for_events(
            inhibitory_times,
            dt_ms=BRUNEL_DT_MS,
            start_ms=args.brunel_presim_ms,
            duration_ticks=actual_steps,
        )
        interval_histograms = {
            "excitatory_population_spike_gap": weighted_interval_histogram(
                [excitatory_ticks]
            ),
            "inhibitory_population_spike_gap": weighted_interval_histogram(
                [inhibitory_ticks]
            ),
            "excitatory_per_neuron_isi": weighted_interval_histogram(post_events),
            "inhibitory_per_neuron_isi": weighted_interval_histogram(
                inhibitory_events
            ),
            "ee_presynaptic_per_synapse": weighted_interval_histogram(
                pre_events, ee_outdegrees
            ),
            "ee_postsynaptic_per_synapse": weighted_interval_histogram(
                post_events, np.full(spec.ne, spec.ce, dtype=np.uint64)
            ),
            "ee_presynaptic_active_tick_gap": weighted_interval_histogram(
                [np.flatnonzero(pre_series)]
            ),
            "ee_postsynaptic_active_tick_gap": weighted_interval_histogram(
                [np.flatnonzero(post_series)]
            ),
        }
        id_gap_histograms = {
            "excitatory_within_tick_id_gap": within_tick_id_gap_histogram(
                times,
                ids,
                dt_ms=BRUNEL_DT_MS,
                start_ms=args.brunel_presim_ms,
                duration_ticks=actual_steps,
            ),
            "inhibitory_within_tick_id_gap": within_tick_id_gap_histogram(
                inhibitory_times,
                inhibitory_ids,
                dt_ms=BRUNEL_DT_MS,
                start_ms=args.brunel_presim_ms,
                duration_ticks=actual_steps,
            ),
        }

        expected_pre_intervals = int(
            np.dot(
                np.maximum(excitatory_counts.astype(np.int64) - 1, 0).astype(
                    np.uint64
                ),
                ee_outdegrees,
            )
        )
        expected_post_intervals = int(
            np.maximum(excitatory_counts.astype(np.int64) - 1, 0).sum(
                dtype=np.int64
            )
            * spec.ce
        )
        observed_pre_intervals = int(
            interval_histograms["ee_presynaptic_per_synapse"].sum(dtype=np.uint64)
        )
        observed_post_intervals = int(
            interval_histograms["ee_postsynaptic_per_synapse"].sum(dtype=np.uint64)
        )
        if observed_pre_intervals != expected_pre_intervals:
            raise RuntimeError(
                f"presynaptic interval mismatch: {observed_pre_intervals} != {expected_pre_intervals}"
            )
        if observed_post_intervals != expected_post_intervals:
            raise RuntimeError(
                f"postsynaptic interval mismatch: {observed_post_intervals} != {expected_post_intervals}"
            )

        iat_windows_1000_ms = []
        for window_index in range(int(elapsed_ms // 1000.0)):
            window_start_ms = args.brunel_presim_ms + window_index * 1000.0
            window_pre_events = events_by_neuron(
                times,
                ids,
                spec.ne,
                dt_ms=BRUNEL_DT_MS,
                start_ms=window_start_ms,
                duration_ticks=round(1000.0 / BRUNEL_DT_MS),
                delay_ms=spec.delay_ms,
            )
            window_post_events = events_by_neuron(
                times,
                ids,
                spec.ne,
                dt_ms=BRUNEL_DT_MS,
                start_ms=window_start_ms,
                duration_ticks=round(1000.0 / BRUNEL_DT_MS),
            )
            window_histograms = {
                "ee_presynaptic_per_synapse": weighted_interval_histogram(
                    window_pre_events, ee_outdegrees
                ),
                "ee_postsynaptic_per_synapse": weighted_interval_histogram(
                    window_post_events,
                    np.full(spec.ne, spec.ce, dtype=np.uint64),
                ),
            }
            window_expected_pre = int(
                np.dot(
                    np.asarray(
                        [max(events.size - 1, 0) for events in window_pre_events],
                        dtype=np.uint64,
                    ),
                    ee_outdegrees,
                )
            )
            window_expected_post = int(
                sum(max(events.size - 1, 0) for events in window_post_events)
                * spec.ce
            )
            window_observed_pre = int(
                window_histograms["ee_presynaptic_per_synapse"].sum(
                    dtype=np.uint64
                )
            )
            window_observed_post = int(
                window_histograms["ee_postsynaptic_per_synapse"].sum(
                    dtype=np.uint64
                )
            )
            if window_observed_pre != window_expected_pre:
                raise RuntimeError(
                    "windowed presynaptic interval mismatch: "
                    f"{window_observed_pre} != {window_expected_pre}"
                )
            if window_observed_post != window_expected_post:
                raise RuntimeError(
                    "windowed postsynaptic interval mismatch: "
                    f"{window_observed_post} != {window_expected_post}"
                )
            iat_windows_1000_ms.append(
                {
                    "window_index": window_index,
                    "start_ms": window_start_ms - args.brunel_presim_ms,
                    "end_ms": window_start_ms
                    - args.brunel_presim_ms
                    + 1000.0,
                    "interval_histograms": {
                        name: interval_histogram_summary(
                            histogram, BRUNEL_DT_MS
                        )
                        for name, histogram in window_histograms.items()
                    },
                    "conservation": {
                        "expected_ee_presynaptic_intervals": window_expected_pre,
                        "observed_ee_presynaptic_intervals": window_observed_pre,
                        "expected_ee_postsynaptic_intervals": window_expected_post,
                        "observed_ee_postsynaptic_intervals": window_observed_post,
                    },
                }
            )

        for pre, post in zip(
            baseline_sample["pre"][: args.interval_edge_sample_size],
            baseline_sample["post"][: args.interval_edge_sample_size],
        ):
            sampled_event_ticks.append(np.concatenate((pre_events[pre], post_events[post])))
        gaps = interval_gaps(sampled_event_ticks)
    finally:
        network.close()

    summary = {
        "case": case_name,
        "actual_simulation_ms": elapsed_ms,
        "termination": termination,
        "excitatory_spikes": recorded_population_totals["excitatory"],
        "inhibitory_spikes": recorded_population_totals["inhibitory"],
        "total_spikes": sum(recorded_population_totals.values()),
        "individual_neurons": {
            "excitatory": count_distribution_summary(excitatory_counts),
            "inhibitory": count_distribution_summary(inhibitory_counts),
        },
        "temporal": {
            "zero_ee_update_tick_fraction": float(np.mean(traversal_series == 0)),
            "all_ee_synapse_traversals": temporal_summaries(traversal_series, BRUNEL_DT_MS),
            "weight_update_attempts": temporal_summaries(traversal_series, BRUNEL_DT_MS),
            "ee_presynaptic_updates": temporal_summaries(pre_series, BRUNEL_DT_MS),
            "ee_postsynaptic_updates": temporal_summaries(post_series, BRUNEL_DT_MS),
            "interval_histograms": {
                name: interval_histogram_summary(histogram, BRUNEL_DT_MS)
                for name, histogram in interval_histograms.items()
            },
            "within_tick_neuron_id_gaps": {
                name: id_gap_histogram_summary(histogram)
                for name, histogram in id_gap_histograms.items()
            },
            "iat_windows_1000_ms": iat_windows_1000_ms,
        },
        "spatial": {
            **graph_summary,
            "sample_weight_change_vs_frequency_spearman": spearman(sample_frequency, sample_abs_delta),
            "sample_mean_absolute_net_weight_change_pa": float(sample_abs_delta.mean()),
            "inter_update_gap_ticks": series_summary(gaps),
        },
        "selected_windows": selected_windows,
        "weight_sample_snapshots_ms": [float(item[0]) for item in snapshots],
        "conservation": {
            "recorded_population_spikes": recorded_population_totals,
            "device_counter_population_spikes": counter_population_totals,
            "expected_ee_presynaptic_updates": expected_pre_updates,
            "observed_ee_presynaptic_updates": observed_pre_updates,
            "expected_ee_postsynaptic_updates": expected_post_updates,
            "observed_ee_postsynaptic_updates": observed_post_updates,
            "expected_ee_presynaptic_intervals": expected_pre_intervals,
            "observed_ee_presynaptic_intervals": observed_pre_intervals,
            "expected_ee_postsynaptic_intervals": expected_post_intervals,
            "observed_ee_postsynaptic_intervals": observed_post_intervals,
        },
    }
    write_json(output / "summary.json", summary)
    arrays = {
        "traversal_series": traversal_series,
        "update_series": traversal_series,
        "pre_series": pre_series,
        "post_series": post_series,
        "excitatory_firing_counts": excitatory_counts,
        "inhibitory_firing_counts": inhibitory_counts,
        "excitatory_spike_times_ms": times[excitatory_keep],
        "excitatory_spike_ids": ids[excitatory_keep],
        "inhibitory_spike_times_ms": inhibitory_times[inhibitory_keep],
        "inhibitory_spike_ids": inhibitory_ids[inhibitory_keep],
        **graph_arrays,
        "sample_ordinal": baseline_sample["ordinal"],
        "sample_pre": baseline_sample["pre"],
        "sample_post": baseline_sample["post"],
        "sample_initial_weight": baseline_sample["weight"],
        "sample_final_weight": final_sample["weight"],
        "sample_frequency": sample_frequency,
        "sample_abs_weight_change": sample_abs_delta,
        "snapshot_times_ms": np.asarray([item[0] for item in snapshots]),
        "snapshot_weights": np.asarray([item[1] for item in snapshots]),
        "inter_update_gap_ticks": gaps,
        **{
            f"interval_histogram_{name}": histogram
            for name, histogram in interval_histograms.items()
        },
        **{
            f"id_gap_histogram_{name}": histogram
            for name, histogram in id_gap_histograms.items()
        },
    }
    np.savez_compressed(output / "locality.npz", **arrays)
    plot_temporal(
        output / "temporal.png",
        traversal_series,
        traversal_series,
        dt_ms=BRUNEL_DT_MS,
        title=case_name,
    )
    plot_brunel_spatial(output / "spatial.png", arrays, title=case_name)
    plot_brunel_intervals(
        output / "intervals.png",
        excitatory_counts,
        inhibitory_counts,
        interval_histograms,
        dt_ms=BRUNEL_DT_MS,
        title=case_name,
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Capture temporal and graph locality of STDP updates")
    command.add_argument("--cases", nargs="+", choices=ALL_CASES, default=ALL_CASES)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--data-path", type=Path, default=REPOSITORY / "data" / "mnist")
    command.add_argument("--mnist-seed", type=int, default=0)
    command.add_argument("--mnist-samples", type=int, default=100)
    command.add_argument("--mnist-max-attempts", type=int, default=200)
    command.add_argument("--brunel-seed", type=int, default=20260724)
    command.add_argument("--brunel-presim-ms", type=float, default=100.0)
    command.add_argument("--brunel-sim-ms", type=float, default=1000.0)
    command.add_argument("--brunel-chunk-ms", type=float, default=100.0)
    command.add_argument("--brunel-abort-rate-hz", type=float, default=100.0)
    command.add_argument("--edge-sample-size", type=int, default=200000)
    command.add_argument("--interval-edge-sample-size", type=int, default=2000)
    command.add_argument("--build-jobs", type=int, default=1)
    return command


def main() -> int:
    args = parser().parse_args()
    configure_cuda_compiler()
    from pygenn import genn_model

    genn_model.cpu_count = lambda logical=False: args.build_jobs
    if args.mnist_samples <= 0 or args.mnist_max_attempts < args.mnist_samples:
        raise SystemExit("invalid MNIST sample or attempt budget")
    if args.build_jobs <= 0:
        raise SystemExit("build jobs must be positive")
    if args.brunel_sim_ms <= 0.0 or args.brunel_presim_ms < 0.0:
        raise SystemExit("invalid Brunel duration")
    if args.brunel_chunk_ms <= 0.0 or args.brunel_abort_rate_hz <= 0.0:
        raise SystemExit("invalid Brunel chunk or rate guard")
    presim_chunks = args.brunel_presim_ms / args.brunel_chunk_ms
    simulation_chunks = args.brunel_sim_ms / args.brunel_chunk_ms
    if not math.isclose(presim_chunks, round(presim_chunks)) or not math.isclose(
        simulation_chunks, round(simulation_chunks)
    ):
        raise SystemExit("Brunel presimulation and simulation must be whole chunk multiples")
    args.output.mkdir(parents=True, exist_ok=False)
    for case_name in args.cases:
        case_output = args.output / case_name
        case_output.mkdir()
        print(f"START case={case_name} output={case_output}", flush=True)
        if case_name in MNIST_CASES:
            run_mnist(args, case_name, case_output)
        else:
            run_brunel(args, case_name, case_output)
        print(f"DONE case={case_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
