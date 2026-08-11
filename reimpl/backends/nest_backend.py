from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from zd3.constants import MODEL
from zd3.evaluation import simple_demo_accuracy
from zd3.io import (
    load_checkpoint,
    load_mnist,
    load_reference_triplets,
    save_checkpoint,
    sha256_file,
)
from zd3.stats import IntervalStats


# NEST cannot transport a spike with a zero-step delay. One NEST resolution is
# the fixed-latency representation of the reference model's zero-delay event,
# which becomes visible to the target on the immediately following cycle.
NEST_SYNAPTIC_DELAY_MS = MODEL.dt_ms


def _import_nest() -> Any:
    try:
        import nest
    except ImportError as error:
        raise SystemExit(
            "PyNEST is unavailable. Use the environment of a source-built NEST installation."
        ) from error
    return nest


class NestNetwork:
    def __init__(
        self,
        *,
        weights: np.ndarray,
        theta_mv: np.ndarray,
        plasticity: bool,
        inhibition: float,
        seed: int,
        module: str,
        threads: int,
    ) -> None:
        self.nest = _import_nest()
        nest = self.nest
        nest.ResetKernel()
        nest.SetKernelStatus(
            {
                "resolution": MODEL.dt_ms,
                "local_num_threads": threads,
                "rng_seed": (seed % (2**32 - 2)) + 1,
                "print_time": False,
            }
        )
        nest.set_verbosity("M_WARNING")
        nest.Install(module)

        self.inputs = nest.Create("poisson_generator", MODEL.n_input)
        exc_parameters = {
            "tau_m": MODEL.exc_tau_m_ms,
            "v_rest": MODEL.exc_v_rest_mv,
            "V_reset": MODEL.exc_v_reset_mv,
            "V_th": MODEL.exc_v_threshold_mv,
            "e_exc": MODEL.exc_e_exc_mv,
            "e_inh": MODEL.exc_e_inh_mv,
            "tau_ge": MODEL.tau_ge_ms,
            "tau_gi": MODEL.tau_gi_ms,
            "t_ref": MODEL.exc_refractory_ms,
            "theta_offset": MODEL.theta_offset_mv,
            "theta_plus": MODEL.theta_plus_mv,
            "theta_tau": MODEL.theta_tau_ms,
            "plasticity": float(plasticity),
            "V_m": MODEL.exc_v_rest_mv - 40.0,
        }
        self.exc = nest.Create("zd3_midpoint_neuron", MODEL.n_exc, exc_parameters)
        self.exc.theta = np.asarray(theta_mv, dtype=np.float64)
        self.inh = nest.Create(
            "zd3_midpoint_neuron",
            MODEL.n_inh,
            {
                "tau_m": MODEL.inh_tau_m_ms,
                "v_rest": MODEL.inh_v_rest_mv,
                "V_reset": MODEL.inh_v_reset_mv,
                "V_th": MODEL.inh_v_threshold_mv,
                "e_exc": MODEL.inh_e_exc_mv,
                "e_inh": MODEL.inh_e_inh_mv,
                "tau_ge": MODEL.tau_ge_ms,
                "tau_gi": MODEL.tau_gi_ms,
                "t_ref": MODEL.inh_refractory_ms,
                "theta_offset": MODEL.theta_offset_mv,
                "plasticity": 0.0,
                "V_m": MODEL.inh_v_rest_mv - 40.0,
            },
        )

        feedforward_model = "zd3_triplet_synapse" if plasticity else "static_synapse"
        nest.Connect(
            self.inputs,
            self.exc,
            "all_to_all",
            {
                "synapse_model": feedforward_model,
                "weight": 0.1,
                "delay": NEST_SYNAPTIC_DELAY_MS,
            },
        )
        input_first = int(np.asarray(self.inputs)[0])
        exc_first = int(np.asarray(self.exc)[0])

        nest.Connect(
            self.exc,
            self.inh,
            "one_to_one",
            {
                "synapse_model": "static_synapse",
                "weight": MODEL.exc_to_inh_weight,
                "delay": NEST_SYNAPTIC_DELAY_MS,
            },
        )
        nest.Connect(
            self.inh,
            self.exc,
            "all_to_all",
            {
                "synapse_model": "static_synapse",
                "weight": -inhibition,
                "delay": NEST_SYNAPTIC_DELAY_MS,
            },
        )
        recurrent = nest.GetConnections(self.inh, self.exc)
        recurrent_sources = np.asarray(recurrent.source)
        recurrent_targets = np.asarray(recurrent.target)
        inh_first = int(np.asarray(self.inh)[0])
        recurrent.weight = np.where(
            recurrent_sources - inh_first == recurrent_targets - exc_first,
            0.0,
            -inhibition,
        )
        # NEST invalidates SynapseCollection descriptors when new connections
        # are added, so retain a fresh handle after topology construction.
        self.feedforward = nest.GetConnections(self.inputs, self.exc)
        one_to_one = nest.GetConnections(self.exc, self.inh)
        for name, connections in (
            ("input_to_exc", self.feedforward),
            ("exc_to_inh", one_to_one),
            ("inh_to_exc", recurrent),
        ):
            delays = np.asarray(connections.delay, dtype=np.float64)
            if not np.all(delays == NEST_SYNAPTIC_DELAY_MS):
                raise RuntimeError(
                    f"{name} delays do not equal the fixed one-cycle latency "
                    f"{NEST_SYNAPTIC_DELAY_MS} ms"
                )
        sources = np.asarray(self.feedforward.source)
        targets = np.asarray(self.feedforward.target)
        self._feedforward_indices = (
            (sources - input_first) * MODEL.n_exc + (targets - exc_first)
        ).astype(np.int64)
        if np.unique(self._feedforward_indices).size != MODEL.n_input * MODEL.n_exc:
            raise RuntimeError("NEST feedforward connection mapping is not dense and unique")
        self.feedforward.weight = np.asarray(weights, dtype=np.float64).reshape(-1)[
            self._feedforward_indices
        ]
        self.exc.ff_raw_sum = np.asarray(weights, dtype=np.float64).sum(axis=0)
        self.exc.ff_scale = 1.0
        self._count_baseline = np.zeros(MODEL.n_exc, dtype=np.int64)

    def set_image(self, pixels: np.ndarray, intensity: float) -> None:
        self.inputs.rate = pixels.astype(np.float64) / 8.0 * intensity

    def set_zero_input(self) -> None:
        self.inputs.rate = 0.0

    def run_stimulus(self) -> np.ndarray:
        self.nest.Simulate(MODEL.stimulus_ms)
        current = np.asarray(self.exc.spike_count, dtype=np.int64)
        counts = current - self._count_baseline
        self._count_baseline = current
        return counts

    def run_rest(self) -> None:
        self.set_zero_input()
        self.nest.Simulate(MODEL.rest_ms)

    def weights(self) -> np.ndarray:
        flat = np.empty(MODEL.n_input * MODEL.n_exc, dtype=np.float64)
        flat[self._feedforward_indices] = np.asarray(
            self.feedforward.weight, dtype=np.float64
        )
        raw = flat.reshape(MODEL.n_input, MODEL.n_exc)
        scales = np.asarray(self.exc.ff_scale, dtype=np.float64)
        return raw * scales[None, :]

    def normalize(self) -> None:
        raw_sums = np.asarray(self.exc.ff_raw_sum, dtype=np.float64)
        if np.any(raw_sums <= 0.0) or not np.all(np.isfinite(raw_sums)):
            raise RuntimeError("invalid NEST feedforward raw column sum")
        self.exc.ff_scale = MODEL.normalization_target / raw_sums

    def theta_mv(self) -> np.ndarray:
        return np.asarray(self.exc.theta, dtype=np.float64)

    def runtime_diagnostics(self) -> dict[str, float]:
        e_v = np.asarray(self.exc.V_m, dtype=np.float64)
        i_v = np.asarray(self.inh.V_m, dtype=np.float64)
        return {
            "e_v_min_mv": float(e_v.min()),
            "e_v_max_mv": float(e_v.max()),
            "i_v_min_mv": float(i_v.min()),
            "i_v_max_mv": float(i_v.max()),
            "e_ge_max": float(np.max(self.exc.ge)),
            "e_gi_max": float(np.max(self.exc.gi)),
            "i_ge_max": float(np.max(self.inh.ge)),
        }

    def validate_runtime(self, counts: np.ndarray, runaway_spikes: int) -> None:
        if int(counts.sum()) >= runaway_spikes:
            raise RuntimeError(f"runaway activity: {int(counts.sum())} E spikes")
        values = [
            self.exc.V_m,
            self.exc.ge,
            self.exc.gi,
            self.exc.theta,
            self.inh.V_m,
            self.inh.ge,
            self.inh.gi,
        ]
        if not all(np.all(np.isfinite(value)) for value in values):
            raise RuntimeError("non-finite NEST neuron state")


def _create_run_directory(path: Path, manifest: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    (path / "checkpoints").mkdir()
    (path / "results").mkdir()
    with (path / "manifest.json").open("x", encoding="ascii") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _initial_state(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        return (
            checkpoint.weights.copy(),
            checkpoint.theta_mv.copy(),
            checkpoint.accepted_samples,
            {"resume_checkpoint": str(args.resume), "resume_sha256": sha256_file(args.resume)},
        )
    return (
        load_reference_triplets(args.initial_weights, MODEL.n_input, MODEL.n_exc),
        np.full(MODEL.n_exc, MODEL.theta_initial_mv, dtype=np.float64),
        0,
        {
            "initial_weights": str(args.initial_weights),
            "initial_weights_sha256": sha256_file(args.initial_weights),
        },
    )


def train(args: argparse.Namespace) -> int:
    weights, theta, start_sample, provenance = _initial_state(args)
    nest = _import_nest()
    manifest = {
        "backend": "nest-cpu",
        "command": sys.argv,
        "dataset": "mnist",
        "data_path": str(args.data_path),
        "host": platform.uname()._asdict(),
        "model": MODEL.as_dict(),
        "nest_version": nest.__version__,
        "module": args.module,
        "requested_samples": args.samples,
        "rng_seed": args.seed,
        "nest_rng_seed": (args.seed % (2**32 - 2)) + 1,
        "start_sample": start_sample,
        "threads": args.threads,
        "configured_synaptic_delay_ms": NEST_SYNAPTIC_DELAY_MS,
        "synaptic_delay_steps": 1,
        "transport_delay_ms": NEST_SYNAPTIC_DELAY_MS,
        **provenance,
    }
    _create_run_directory(args.output, manifest)
    data = load_mnist(args.data_path, "train")
    network = NestNetwork(
        weights=weights,
        theta_mv=theta,
        plasticity=True,
        inhibition=MODEL.train_inhibition,
        seed=args.seed,
        module=args.module,
        threads=args.threads,
    )
    print(
        f"CONFIG backend=nest-cpu mode=train dt_ms={MODEL.dt_ms:.6f} "
        f"synaptic_delay_ms={NEST_SYNAPTIC_DELAY_MS:.6f} synaptic_delay_steps=1 "
        f"samples={args.samples} seed={args.seed} threads={args.threads}",
        flush=True,
    )
    interval = IntervalStats()
    attempts = 0
    wall = 0.0
    accepted = start_sample
    while accepted < args.samples:
        intensity = MODEL.initial_intensity
        while True:
            started = time.perf_counter()
            network.normalize()
            network.set_image(data.images[accepted % len(data.images)], intensity)
            counts = network.run_stimulus()
            network.validate_runtime(counts, args.runaway_spikes)
            retry = int(counts.sum()) < MODEL.minimum_exc_spikes
            network.run_rest()
            wall += time.perf_counter() - started
            attempts += 1
            interval.record_attempt(retry)
            if retry:
                intensity += MODEL.intensity_increment
                if intensity > args.max_intensity:
                    raise RuntimeError(f"sample {accepted} exceeded max intensity")
                continue
            accepted += 1
            interval.record_accepted(counts, intensity)
            break
        if accepted % args.stats_interval == 0:
            print(
                interval.format(
                    accepted=accepted,
                    weights=network.weights(),
                    theta_mv=network.theta_mv(),
                    interval=args.stats_interval,
                    backend="nest-cpu",
                    runtime=network.runtime_diagnostics(),
                ),
                flush=True,
            )
            interval = IntervalStats()
        if accepted % args.checkpoint_interval == 0 or accepted == args.samples:
            path = args.output / "checkpoints" / f"checkpoint_{accepted:06d}.npz"
            save_checkpoint(
                path,
                weights=network.weights(),
                theta_mv=network.theta_mv(),
                accepted_samples=accepted,
                manifest={
                    "backend": "nest-cpu",
                    "rng_seed": args.seed,
                    "source_run": str(args.output),
                    "runtime_state_scope": "weights-and-theta-only",
                },
            )
            print(f"CHECKPOINT backend=nest-cpu accepted={accepted} path={path}", flush=True)
    ticks = attempts * MODEL.attempt_ticks
    result = {
        "accepted_samples": accepted - start_sample,
        "attempts": attempts,
        "retries": attempts - (accepted - start_sample),
        "simulated_ticks": ticks,
        "simulation_wall_seconds": wall,
        "seconds_per_timestep_cycle": wall / ticks,
        "accepted_samples_per_second": (accepted - start_sample) / wall,
        "biological_realtime_factor": ticks * MODEL.dt_ms / 1000.0 / wall,
    }
    with (args.output / "results" / "performance.json").open("x", encoding="ascii") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("PERFORMANCE " + " ".join(f"{key}={value}" for key, value in result.items()), flush=True)
    print(f"RUN_COMPLETE backend=nest-cpu accepted={accepted}", flush=True)
    return 0


def evaluate(args: argparse.Namespace) -> int:
    checkpoint = load_checkpoint(args.checkpoint)
    nest = _import_nest()
    manifest = {
        "backend": "nest-cpu",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "command": sys.argv,
        "mode": "evaluate",
        "model": MODEL.as_dict(),
        "module": args.module,
        "nest_version": nest.__version__,
        "probe_samples": args.samples,
        "rng_seed": args.seed,
        "nest_rng_seed": (args.seed % (2**32 - 2)) + 1,
        "threads": args.threads,
        "configured_synaptic_delay_ms": NEST_SYNAPTIC_DELAY_MS,
        "synaptic_delay_steps": 1,
        "transport_delay_ms": NEST_SYNAPTIC_DELAY_MS,
    }
    _create_run_directory(args.output, manifest)
    data = load_mnist(args.data_path, "test")
    network = NestNetwork(
        weights=checkpoint.weights,
        theta_mv=checkpoint.theta_mv,
        plasticity=False,
        inhibition=MODEL.inference_inhibition,
        seed=args.seed,
        module=args.module,
        threads=args.threads,
    )
    activity = np.zeros((args.samples, MODEL.n_exc), dtype=np.uint16)
    attempts = 0
    wall = 0.0
    for sample in range(args.samples):
        intensity = MODEL.initial_intensity
        while True:
            started = time.perf_counter()
            network.set_image(data.images[sample], intensity)
            counts = network.run_stimulus()
            network.validate_runtime(counts, args.runaway_spikes)
            retry = int(counts.sum()) < MODEL.minimum_exc_spikes
            network.run_rest()
            wall += time.perf_counter() - started
            attempts += 1
            if retry:
                intensity += MODEL.intensity_increment
                if intensity > args.max_intensity:
                    raise RuntimeError(f"sample {sample} exceeded max intensity")
                continue
            activity[sample] = counts
            break
    score = simple_demo_accuracy(activity, data.labels[: args.samples])
    result = {
        "accuracy_percent": score["accuracy_percent"],
        "assigned_neurons": score["assigned_neurons"],
        "assignment_counts": score["assignment_counts"],
        "backend": "nest-cpu",
        "checkpoint_samples": checkpoint.accepted_samples,
        "probe_samples": args.samples,
        "protocol": "same test activity used for assignment and scoring",
        "attempts": attempts,
        "retries": attempts - args.samples,
        "simulation_wall_seconds": wall,
        "simulated_ticks": attempts * MODEL.attempt_ticks,
        "seconds_per_timestep_cycle": wall / (attempts * MODEL.attempt_ticks),
        "spikes_mean": float(activity.sum(axis=1).mean()),
        "active_mean": float(np.count_nonzero(activity, axis=1).mean()),
    }
    np.savez(
        args.output / "results" / "activity.npz",
        activity=activity,
        labels=data.labels[: args.samples],
        predictions=score["predictions"],
        assignments=score["assignments"],
    )
    with (args.output / "results" / "score.json").open("x", encoding="ascii") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("CHECKPOINT_ACCURACY " + " ".join(f"{key}={value}" for key, value in result.items()), flush=True)
    print(f"RUN_COMPLETE backend=nest-cpu accepted={args.samples}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    repository = root.parent
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--data-path", type=Path, default=repository / "data" / "mnist")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--module", default="zd3module")
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--runaway-spikes", type=int, default=5000)
        command.add_argument("--max-intensity", type=float, default=20.0)

    train_parser = subparsers.add_parser("train")
    common(train_parser)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--samples", type=int, required=True)
    train_parser.add_argument("--stats-interval", type=int, default=1000)
    train_parser.add_argument("--checkpoint-interval", type=int, default=10000)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument(
        "--initial-weights",
        type=Path,
        default=repository / "ref" / "zero_delay_midpoint_v1" / "inhib150_full" / "random" / "XeAe.npy",
    )
    train_parser.set_defaults(function=train)

    eval_parser = subparsers.add_parser("evaluate")
    common(eval_parser)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--samples", type=int, default=1000)
    eval_parser.set_defaults(function=evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples <= 0 or args.threads <= 0:
        raise SystemExit("--samples and --threads must be positive")
    return args.function(args)
