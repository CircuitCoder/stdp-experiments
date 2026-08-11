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
    PortableCheckpoint,
    load_checkpoint,
    load_mnist,
    load_reference_triplets,
    normalize_columns,
    save_checkpoint,
    sha256_file,
)
from zd3.stats import IntervalStats
from zd3.variants import (
    NetworkVariant,
    connectivity_mask,
    get_variant,
    validate_normalized_weight_bound,
)


def _import_brian2() -> Any:
    try:
        import brian2 as b
    except ImportError as error:
        raise SystemExit(
            "Brian2 is unavailable. Run this command in the reimpl development environment."
        ) from error
    return b


class Brian2Model:
    def __init__(
        self,
        *,
        weights: np.ndarray,
        theta_mv: np.ndarray,
        plasticity: bool,
        inhibition: float,
        seed: int,
        codegen_target: str,
        standalone_codegen: bool = False,
        variant: NetworkVariant | None = None,
        structural_mask: np.ndarray | None = None,
    ) -> None:
        self.b = _import_brian2()
        b = self.b
        b.start_scope()
        b.defaultclock.dt = MODEL.dt_ms * b.ms
        b.seed(seed)
        b.prefs.codegen.target = codegen_target
        self.variant = variant if variant is not None else get_variant("triplet-dense")
        self.structural_mask = (
            connectivity_mask(self.variant)
            if structural_mask is None
            else np.asarray(structural_mask, dtype=bool)
        )
        if self.structural_mask.shape != (MODEL.n_input, MODEL.n_exc):
            raise ValueError("invalid feedforward structural mask shape")
        self._feedforward_pre, self._feedforward_post = np.nonzero(
            self.structural_mask
        )

        integration_dt = MODEL.dt_ms * b.ms
        tau_ge = MODEL.tau_ge_ms * b.ms
        tau_gi = MODEL.tau_gi_ms * b.ms
        theta_tau = MODEL.theta_tau_ms * b.ms
        plasticity_value = 1.0 if plasticity else 0.0

        exc_equations = """
        dv/dt = (v_step - v) / integration_dt : volt (unless refractory)
        v_step = v_inf_mid + (v - v_inf_mid) * exp(-integration_dt * g_mid / tau_m) : volt
        v_inf_mid = (v_rest + ge_mid * e_exc + gi_mid * e_inh) / g_mid : volt
        g_mid = 1.0 + ge_mid + gi_mid : 1
        ge_mid = ge * exp(-0.5 * integration_dt / tau_ge) : 1
        gi_mid = gi * exp(-0.5 * integration_dt / tau_gi) : 1
        dge/dt = (ge_step - ge) / integration_dt : 1 (unless refractory)
        dgi/dt = (gi_step - gi) / integration_dt : 1 (unless refractory)
        ge_step = ge * exp(-integration_dt / tau_ge) : 1
        gi_step = gi * exp(-integration_dt / tau_gi) : 1
        dtheta/dt = (theta_step - theta) / integration_dt : volt (unless refractory)
        theta_step = theta * exp(-plasticity_value * integration_dt / theta_tau) : volt
        plasticity_value : 1 (constant)
        integration_dt : second (constant)
        tau_m : second (constant)
        tau_ge : second (constant)
        tau_gi : second (constant)
        theta_tau : second (constant)
        v_rest : volt (constant)
        e_exc : volt (constant)
        e_inh : volt (constant)
        threshold_base : volt (constant)
        threshold_offset : volt (constant)
        theta_plus : volt (constant)
        """
        inh_equations = """
        dv/dt = (v_step - v) / integration_dt : volt (unless refractory)
        v_step = v_inf_mid + (v - v_inf_mid) * exp(-integration_dt * g_mid / tau_m) : volt
        v_inf_mid = (v_rest + ge_mid * e_exc + gi_mid * e_inh) / g_mid : volt
        g_mid = 1.0 + ge_mid + gi_mid : 1
        ge_mid = ge * exp(-0.5 * integration_dt / tau_ge) : 1
        gi_mid = gi * exp(-0.5 * integration_dt / tau_gi) : 1
        dge/dt = (ge_step - ge) / integration_dt : 1 (unless refractory)
        dgi/dt = (gi_step - gi) / integration_dt : 1 (unless refractory)
        ge_step = ge * exp(-integration_dt / tau_ge) : 1
        gi_step = gi * exp(-integration_dt / tau_gi) : 1
        integration_dt : second (constant)
        tau_m : second (constant)
        tau_ge : second (constant)
        tau_gi : second (constant)
        v_rest : volt (constant)
        e_exc : volt (constant)
        e_inh : volt (constant)
        threshold_base : volt (constant)
        """

        self.inputs = b.PoissonGroup(MODEL.n_input, rates=0 * b.Hz, name="input")
        self.exc = b.NeuronGroup(
            MODEL.n_exc,
            exc_equations,
            threshold="v > theta - threshold_offset + threshold_base",
            reset="v = v_reset; theta += plasticity_value * theta_plus",
            refractory=MODEL.exc_refractory_ms * b.ms,
            method="euler",
            namespace={"v_reset": MODEL.exc_v_reset_mv * b.mV},
            name="exc",
        )
        self.inh = b.NeuronGroup(
            MODEL.n_inh,
            inh_equations,
            threshold="v > threshold_base",
            reset="v = v_reset",
            refractory=MODEL.inh_refractory_ms * b.ms,
            method="euler",
            namespace={"v_reset": MODEL.inh_v_reset_mv * b.mV},
            name="inh",
        )

        self.exc.integration_dt = integration_dt
        self.exc.tau_m = MODEL.exc_tau_m_ms * b.ms
        self.exc.tau_ge = tau_ge
        self.exc.tau_gi = tau_gi
        self.exc.theta_tau = theta_tau
        self.exc.plasticity_value = plasticity_value
        self.exc.v_rest = MODEL.exc_v_rest_mv * b.mV
        self.exc.e_exc = MODEL.exc_e_exc_mv * b.mV
        self.exc.e_inh = MODEL.exc_e_inh_mv * b.mV
        self.exc.threshold_base = MODEL.exc_v_threshold_mv * b.mV
        self.exc.threshold_offset = MODEL.theta_offset_mv * b.mV
        self.exc.theta_plus = MODEL.theta_plus_mv * b.mV
        self.exc.v = (MODEL.exc_v_rest_mv - 40.0) * b.mV
        self.exc.ge = 0.0
        self.exc.gi = 0.0
        self.exc.theta = theta_mv * b.mV

        self.inh.integration_dt = integration_dt
        self.inh.tau_m = MODEL.inh_tau_m_ms * b.ms
        self.inh.tau_ge = tau_ge
        self.inh.tau_gi = tau_gi
        self.inh.v_rest = MODEL.inh_v_rest_mv * b.mV
        self.inh.e_exc = MODEL.inh_e_exc_mv * b.mV
        self.inh.e_inh = MODEL.inh_e_inh_mv * b.mV
        self.inh.threshold_base = MODEL.inh_v_threshold_mv * b.mV
        self.inh.v = (MODEL.inh_v_rest_mv - 40.0) * b.mV
        self.inh.ge = 0.0
        self.inh.gi = 0.0

        if not plasticity:
            stdp_equations = "w : 1"
            on_pre = "ge_post += w"
            on_post = None
            stdp_namespace = {}
        elif self.variant.learning_rule == "three-trace":
            stdp_equations = """
            w : 1
            dpre_trace/dt = -pre_trace / pre_tau : 1 (event-driven)
            dpost1/dt = -post1 / post1_tau : 1 (event-driven)
            dpost2/dt = -post2 / post2_tau : 1 (event-driven)
            """
            on_pre = """
            ge_post += w
            pre_trace = 1.0
            w = clip(w - depression_rate * post1, weight_min, weight_max)
            """
            on_post = """
            w = clip(w + potentiation_rate * pre_trace * post2, weight_min, weight_max)
            post1 = 1.0
            post2 = 1.0
            """
            stdp_namespace = {
                "pre_tau": MODEL.pre_tau_ms * b.ms,
                "post1_tau": MODEL.post1_tau_ms * b.ms,
                "post2_tau": MODEL.post2_tau_ms * b.ms,
                "depression_rate": MODEL.depression_rate,
                "potentiation_rate": MODEL.potentiation_rate,
                "weight_min": MODEL.weight_min,
                "weight_max": self.variant.weight_max,
            }
        elif self.variant.learning_rule == "one-trace-power":
            stdp_equations = """
            w : 1
            dpre_trace/dt = -pre_trace / pre_tau : 1 (event-driven)
            """
            on_pre = """
            ge_post += w
            pre_trace += 1.0
            """
            on_post = """
            w = clip(w + potentiation_rate * (pre_trace - pre_target) * ((weight_max - w) ** post_exponent), weight_min, weight_max)
            """
            stdp_namespace = {
                "pre_tau": MODEL.pre_tau_ms * b.ms,
                "potentiation_rate": self.variant.potentiation_rate,
                "pre_target": self.variant.pre_trace_target,
                "post_exponent": self.variant.post_weight_exponent,
                "weight_min": MODEL.weight_min,
                "weight_max": self.variant.weight_max,
            }
        else:
            raise ValueError(f"unsupported learning rule: {self.variant.learning_rule}")
        self.feedforward = b.Synapses(
            self.inputs,
            self.exc,
            model=stdp_equations,
            on_pre=on_pre,
            on_post=on_post,
            method="exact",
            namespace=stdp_namespace,
            name="feedforward",
        )
        self.feedforward.connect(i=self._feedforward_pre, j=self._feedforward_post)
        if not standalone_codegen:
            if not np.array_equal(
                self.feedforward.i[:], self._feedforward_pre
            ) or not np.array_equal(
                self.feedforward.j[:], self._feedforward_post
            ):
                raise RuntimeError("Brian2 synapse order is not input-major row order")
        self.feedforward.w = np.asarray(
            weights[self.structural_mask], dtype=np.float64
        )
        if plasticity:
            self.feedforward.pre.order = 0
            self.feedforward.post.order = 1

        self.exc_to_inh = b.Synapses(
            self.exc,
            self.inh,
            on_pre="ge_post += exc_to_inh_weight",
            namespace={"exc_to_inh_weight": MODEL.exc_to_inh_weight},
            name="exc_to_inh",
        )
        self.exc_to_inh.connect(j="i")
        self.inh_to_exc = b.Synapses(
            self.inh,
            self.exc,
            on_pre="gi_post += inhibition_weight",
            namespace={"inhibition_weight": inhibition},
            name="inh_to_exc",
        )
        self.inh_to_exc.connect(condition="i != j")

        self.spikes = b.SpikeMonitor(self.exc, record=False, name="exc_spikes")
        self.network = b.Network(
            self.inputs,
            self.exc,
            self.inh,
            self.feedforward,
            self.exc_to_inh,
            self.inh_to_exc,
            self.spikes,
        )
        self.network.run(0 * b.ms, namespace={})
        self._count_baseline = (
            np.zeros(MODEL.n_exc, dtype=np.int64)
            if standalone_codegen
            else np.asarray(self.spikes.count[:], dtype=np.int64)
        )

    def set_image(self, pixels: np.ndarray, intensity: float) -> None:
        self.inputs.rates = (pixels.astype(np.float64) / 8.0 * intensity) * self.b.Hz

    def set_zero_input(self) -> None:
        self.inputs.rates = 0 * self.b.Hz

    def normalize(self) -> None:
        weights = self.weights()
        normalize_columns(weights)
        validate_normalized_weight_bound(weights, self.variant)
        self.feedforward.w = weights[self.structural_mask]

    def run_stimulus(self) -> np.ndarray:
        self.network.run(MODEL.stimulus_ms * self.b.ms, namespace={})
        current = np.asarray(self.spikes.count[:], dtype=np.int64)
        counts = current - self._count_baseline
        # Brian 1 advances its counter snapshot here, before the rest period.
        # The following presentation therefore includes preceding-rest spikes.
        self._count_baseline = current
        return counts

    def run_rest(self) -> None:
        self.set_zero_input()
        self.network.run(MODEL.rest_ms * self.b.ms, namespace={})

    def weights(self) -> np.ndarray:
        weights = np.zeros((MODEL.n_input, MODEL.n_exc), dtype=np.float64)
        weights[self._feedforward_pre, self._feedforward_post] = np.asarray(
            self.feedforward.w[:], dtype=np.float64
        )
        return weights

    def theta_mv(self) -> np.ndarray:
        return np.asarray(self.exc.theta[:] / self.b.mV, dtype=np.float64)

    def runtime_diagnostics(self) -> dict[str, float]:
        return {
            "e_v_min_mv": float(np.min(self.exc.v[:] / self.b.mV)),
            "e_v_max_mv": float(np.max(self.exc.v[:] / self.b.mV)),
            "i_v_min_mv": float(np.min(self.inh.v[:] / self.b.mV)),
            "i_v_max_mv": float(np.max(self.inh.v[:] / self.b.mV)),
            "e_ge_max": float(np.max(self.exc.ge[:])),
            "e_gi_max": float(np.max(self.exc.gi[:])),
            "i_ge_max": float(np.max(self.inh.ge[:])),
        }

    def validate_runtime(self, stimulus_counts: np.ndarray, runaway_spikes: int) -> None:
        if int(stimulus_counts.sum()) >= runaway_spikes:
            raise RuntimeError(
                f"runaway activity: {int(stimulus_counts.sum())} stimulus E spikes"
            )
        arrays = [
            self.exc.v[:] / self.b.mV,
            self.exc.ge[:],
            self.exc.gi[:],
            self.inh.v[:] / self.b.mV,
            self.inh.ge[:],
            self.inh.gi[:],
        ]
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise RuntimeError("non-finite neuron state")


def _create_run_directory(path: Path, manifest: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    (path / "checkpoints").mkdir()
    (path / "results").mkdir()
    with (path / "manifest.json").open("x", encoding="ascii") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _initial_state(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume)
        return (
            checkpoint.weights.copy(),
            checkpoint.theta_mv.copy(),
            checkpoint.accepted_samples,
            {"resume_checkpoint": str(args.resume), "resume_sha256": sha256_file(args.resume)},
        )
    weights = load_reference_triplets(
        args.initial_weights, MODEL.n_input, MODEL.n_exc, dtype=np.float64
    )
    theta = np.full(MODEL.n_exc, MODEL.theta_initial_mv, dtype=np.float64)
    return (
        weights,
        theta,
        0,
        {
            "initial_weights": str(args.initial_weights),
            "initial_weights_sha256": sha256_file(args.initial_weights),
        },
    )


def train(args: argparse.Namespace) -> int:
    weights, theta, start_sample, provenance = _initial_state(args)
    if start_sample >= args.samples:
        raise SystemExit("resume checkpoint already reached the requested sample count")
    b = _import_brian2()
    manifest = {
        "backend": "brian2-runtime",
        "brian2_version": getattr(b, "__version__", "unknown"),
        "codegen_target": args.codegen_target,
        "command": sys.argv,
        "dataset": "mnist",
        "data_path": str(args.data_path),
        "evaluation_protocol": "first-1000-test same-activity assignment and scoring",
        "host": platform.uname()._asdict(),
        "model": MODEL.as_dict(),
        "requested_samples": args.samples,
        "rng_seed": args.seed,
        "start_sample": start_sample,
        **provenance,
    }
    _create_run_directory(args.output, manifest)
    data = load_mnist(args.data_path, "train")
    model = Brian2Model(
        weights=weights,
        theta_mv=theta,
        plasticity=True,
        inhibition=MODEL.train_inhibition,
        seed=args.seed,
        codegen_target=args.codegen_target,
    )
    print(
        "CONFIG backend=brian2-runtime mode=train "
        f"dt_ms={MODEL.dt_ms:.6f} input_delay_ms=0.000000 "
        f"transport_delay_ms=0.000000 samples={args.samples} seed={args.seed}",
        flush=True,
    )

    interval = IntervalStats()
    total_attempts = 0
    simulation_wall = 0.0
    accepted = start_sample
    while accepted < args.samples:
        intensity = MODEL.initial_intensity
        while True:
            started = time.perf_counter()
            model.normalize()
            model.set_image(data.images[accepted % data.images.shape[0]], intensity)
            counts = model.run_stimulus()
            model.validate_runtime(counts, args.runaway_spikes)
            retry = int(counts.sum()) < MODEL.minimum_exc_spikes
            model.run_rest()
            simulation_wall += time.perf_counter() - started
            total_attempts += 1
            interval.record_attempt(retry)
            if retry:
                intensity += MODEL.intensity_increment
                if intensity > args.max_intensity:
                    raise RuntimeError(
                        f"sample {accepted} exceeded maximum intensity {args.max_intensity}"
                    )
                continue
            accepted += 1
            interval.record_accepted(counts, intensity)
            break

        if accepted % args.stats_interval == 0:
            current_weights = model.weights()
            current_theta = model.theta_mv()
            print(
                interval.format(
                    accepted=accepted,
                    weights=current_weights,
                    theta_mv=current_theta,
                    interval=args.stats_interval,
                    backend="brian2-runtime",
                    runtime=model.runtime_diagnostics(),
                ),
                flush=True,
            )
            interval = IntervalStats()

        if accepted % args.checkpoint_interval == 0 or accepted == args.samples:
            checkpoint_path = args.output / "checkpoints" / f"checkpoint_{accepted:06d}.npz"
            save_checkpoint(
                checkpoint_path,
                weights=model.weights(),
                theta_mv=model.theta_mv(),
                accepted_samples=accepted,
                manifest={
                    "backend": "brian2-runtime",
                    "rng_seed": args.seed,
                    "source_run": str(args.output),
                    "runtime_state_scope": "weights-and-theta-only",
                },
            )
            print(
                f"CHECKPOINT backend=brian2-runtime accepted={accepted} path={checkpoint_path}",
                flush=True,
            )

    simulated_ticks = total_attempts * MODEL.attempt_ticks
    performance = {
        "accepted_samples": accepted - start_sample,
        "attempts": total_attempts,
        "retries": total_attempts - (accepted - start_sample),
        "simulated_ticks": simulated_ticks,
        "simulation_wall_seconds": simulation_wall,
        "seconds_per_timestep_cycle": simulation_wall / simulated_ticks,
        "accepted_samples_per_second": (accepted - start_sample) / simulation_wall,
        "biological_realtime_factor": (
            simulated_ticks * MODEL.dt_ms / 1000.0 / simulation_wall
        ),
    }
    with (args.output / "results" / "performance.json").open(
        "x", encoding="ascii"
    ) as stream:
        json.dump(performance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("PERFORMANCE " + " ".join(f"{k}={v}" for k, v in performance.items()), flush=True)
    print(f"RUN_COMPLETE backend=brian2-runtime accepted={accepted}", flush=True)
    return 0


def evaluate(args: argparse.Namespace) -> int:
    checkpoint: PortableCheckpoint = load_checkpoint(args.checkpoint)
    b = _import_brian2()
    manifest = {
        "backend": "brian2-runtime",
        "brian2_version": getattr(b, "__version__", "unknown"),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "command": sys.argv,
        "mode": "evaluate",
        "model": MODEL.as_dict(),
        "probe_samples": args.samples,
        "rng_seed": args.seed,
    }
    _create_run_directory(args.output, manifest)
    data = load_mnist(args.data_path, "test")
    model = Brian2Model(
        weights=checkpoint.weights,
        theta_mv=checkpoint.theta_mv,
        plasticity=False,
        inhibition=MODEL.inference_inhibition,
        seed=args.seed,
        codegen_target=args.codegen_target,
    )
    activity = np.zeros((args.samples, MODEL.n_exc), dtype=np.uint16)
    interval = IntervalStats()
    attempts = 0
    simulation_wall = 0.0
    for sample in range(args.samples):
        intensity = MODEL.initial_intensity
        while True:
            started = time.perf_counter()
            model.set_image(data.images[sample], intensity)
            counts = model.run_stimulus()
            model.validate_runtime(counts, args.runaway_spikes)
            retry = int(counts.sum()) < MODEL.minimum_exc_spikes
            model.run_rest()
            simulation_wall += time.perf_counter() - started
            attempts += 1
            interval.record_attempt(retry)
            if retry:
                intensity += MODEL.intensity_increment
                if intensity > args.max_intensity:
                    raise RuntimeError(
                        f"sample {sample} exceeded maximum intensity {args.max_intensity}"
                    )
                continue
            activity[sample] = counts
            interval.record_accepted(counts, intensity)
            break
    score = simple_demo_accuracy(activity, data.labels[: args.samples])
    score_record = {
        key: value
        for key, value in score.items()
        if key not in {"predictions", "assignments"}
    }
    score_record.update(
        {
            "backend": "brian2-runtime",
            "checkpoint_samples": checkpoint.accepted_samples,
            "probe_samples": args.samples,
            "protocol": "same test activity used for assignment and scoring",
            "attempts": attempts,
            "retries": attempts - args.samples,
            "simulation_wall_seconds": simulation_wall,
            "simulated_ticks": attempts * MODEL.attempt_ticks,
            "seconds_per_timestep_cycle": simulation_wall
            / (attempts * MODEL.attempt_ticks),
            "spikes_mean": float(activity.sum(axis=1).mean()),
            "active_mean": float(np.count_nonzero(activity, axis=1).mean()),
        }
    )
    np.savez(
        args.output / "results" / "activity.npz",
        activity=activity,
        labels=data.labels[: args.samples],
        predictions=score["predictions"],
        assignments=score["assignments"],
    )
    with (args.output / "results" / "score.json").open("x", encoding="ascii") as stream:
        json.dump(score_record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        "CHECKPOINT_ACCURACY "
        + " ".join(f"{key}={value}" for key, value in score_record.items()),
        flush=True,
    )
    print(f"RUN_COMPLETE backend=brian2-runtime accepted={args.samples}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    repository = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--data-path", type=Path, default=repository / "data" / "mnist"
        )
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--codegen-target", choices=("cython", "numpy"), default="cython")
        command.add_argument("--runaway-spikes", type=int, default=5000)
        command.add_argument("--max-intensity", type=float, default=20.0)

    train_parser = subparsers.add_parser("train")
    add_common(train_parser)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--samples", type=int, required=True)
    train_parser.add_argument("--stats-interval", type=int, default=1000)
    train_parser.add_argument("--checkpoint-interval", type=int, default=10000)
    train_parser.add_argument("--resume", type=Path)
    train_parser.add_argument(
        "--initial-weights",
        type=Path,
        default=(
            repository
            / "ref"
            / "zero_delay_midpoint_v1"
            / "inhib150_full"
            / "random"
            / "XeAe.npy"
        ),
    )
    train_parser.set_defaults(function=train)

    eval_parser = subparsers.add_parser("evaluate")
    add_common(eval_parser)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--samples", type=int, default=1000)
    eval_parser.set_defaults(function=evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
