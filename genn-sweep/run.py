#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


SWEEP_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = SWEEP_DIRECTORY.parent
CHECKPOINT_DIRECTORY = SWEEP_DIRECTORY / "checkpoints"
sys.path.insert(0, str(REPOSITORY / "reimpl"))
sys.path.insert(0, str(REPOSITORY / "brunel"))

from backends.genn_backend import GeNNNetwork  # noqa: E402
from ports.common import DT_MS as BRUNEL_DT_MS  # noqa: E402
from ports.common import GENN_DEFAULT_DELAY_MS as BRUNEL_DELAY_MS  # noqa: E402
from ports.common import (  # noqa: E402
    GENN_DEFAULT_EXTERNAL_RATE_SCALES as BRUNEL_EXTERNAL_RATE_SCALES,
)
from ports.common import (  # noqa: E402
    GENN_DEFAULT_INDEGREE_SCALE as BRUNEL_INDEGREE_SCALE,
)
from ports.common import (  # noqa: E402
    GENN_DEFAULT_RECURRENT_DELIVERY_SCALE as BRUNEL_RECURRENT_DELIVERY_SCALE,
)
from ports.common import make_model  # noqa: E402
from ports.genn_port import GeNNBrunel  # noqa: E402
from zd3.constants import MODEL  # noqa: E402
from zd3.io import load_checkpoint, load_mnist  # noqa: E402
from zd3.variants import get_variant, validate_checkpoint_topology  # noqa: E402


@dataclass(frozen=True)
class MnistCase:
    variant: str
    parallelism: str
    threads_per_spike: int
    checkpoint_argument: str


MNIST_CASES = {
    "mnist_triplet_dense": MnistCase(
        variant="triplet-dense",
        parallelism="postsynaptic",
        threads_per_spike=1,
        checkpoint_argument="triplet_checkpoint",
    ),
    "mnist_one_trace_dense": MnistCase(
        variant="one-trace-dense",
        parallelism="postsynaptic",
        threads_per_spike=1,
        checkpoint_argument="dense_checkpoint",
    ),
    "mnist_one_trace_sparse_0125": MnistCase(
        variant="one-trace-bernoulli-0125",
        parallelism="presynaptic",
        threads_per_spike=32,
        checkpoint_argument="sparse_checkpoint",
    ),
}
BRUNEL_CASES = {
    "brunel_additive": "additive",
    "brunel_morrison": "morrison",
}
ALL_CASES = tuple(MNIST_CASES) + tuple(BRUNEL_CASES)


@dataclass(frozen=True)
class Result:
    spike_count: int
    wall_seconds: float
    simulation_steps: int


@contextmanager
def redirect_process_output(path: Path) -> Iterator[None]:
    """Redirect Python, native-library, and compiler output for one case."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(log_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def build_location(
    args: argparse.Namespace, case_name: str, precision: str
) -> tuple[Path, Path | None]:
    name = f"{case_name}_{precision}"
    if args.reuse_build_root is not None:
        reusable = args.reuse_build_root.resolve() / name
        return reusable, reusable
    return args.work_dir.resolve() / "builds" / name, None


def run_mnist(
    args: argparse.Namespace, case_name: str, precision: str
) -> Result:
    case = MNIST_CASES[case_name]
    variant = get_variant(case.variant)
    checkpoint = load_checkpoint(getattr(args, case.checkpoint_argument))
    structural_mask = validate_checkpoint_topology(checkpoint.weights, variant)
    data = load_mnist(args.data_path, "train")
    build_path, reuse_build = build_location(args, case_name, precision)
    network = GeNNNetwork(
        weights=checkpoint.weights.copy(),
        theta_mv=checkpoint.theta_mv.copy(),
        plasticity=True,
        inhibition=MODEL.train_inhibition,
        seed=args.mnist_seed,
        backend="cuda",
        build_path=build_path,
        variant=variant,
        structural_mask=structural_mask,
        precision=precision,
        parallelism=case.parallelism,
        num_threads_per_spike=case.threads_per_spike,
        timing_enabled=False,
        reuse_build=reuse_build,
    )
    accepted = 0
    attempts = 0
    try:
        started = time.perf_counter()
        while accepted < args.mnist_samples:
            intensity = MODEL.initial_intensity
            while True:
                network.normalize(validate=False)
                sample = checkpoint.accepted_samples + accepted
                network.set_image(data.images[sample % len(data.images)], intensity)
                excitatory_counts = network.run_stimulus()
                retry = int(excitatory_counts.sum()) < MODEL.minimum_exc_spikes
                network.run_rest(synchronize=False)
                attempts += 1
                if retry:
                    intensity += MODEL.intensity_increment
                else:
                    accepted += 1
                    break
        spike_count = network.total_spike_count()
        wall_seconds = time.perf_counter() - started
    finally:
        network.close()
    return Result(
        spike_count=spike_count,
        wall_seconds=wall_seconds,
        simulation_steps=attempts * MODEL.attempt_ticks,
    )


def run_brunel(
    args: argparse.Namespace, case_name: str, precision: str
) -> Result:
    rule = BRUNEL_CASES[case_name]
    external_rate_scale = getattr(
        args, f"brunel_{rule}_external_rate_scale"
    )
    spec = make_model(
        rule,
        1.0,
        args.brunel_indegree_scale,
        delay_ms=args.brunel_delay_ms,
        recurrent_delivery_scale=args.brunel_recurrent_delivery_scale,
        external_rate_scale=external_rate_scale,
    )
    build_path, reuse_build = build_location(args, case_name, precision)
    network = GeNNBrunel(
        spec=spec,
        seed=args.brunel_seed,
        state_seed=args.brunel_seed,
        backend="cuda",
        build_path=build_path,
        recording_steps=0,
        precision=precision,
        ee_parallelism="postsynaptic",
        ee_num_threads_per_spike=1,
        stdp_timing="arrival",
        stdp_tie_order="nest_causal_boundary",
        timing_enabled=False,
        reuse_build=reuse_build,
        record_spikes=False,
        collect_connectivity_stats=False,
    )
    presimulation_steps = round(args.brunel_presim_ms / BRUNEL_DT_MS)
    simulation_steps = round(args.brunel_sim_ms / BRUNEL_DT_MS)
    try:
        network.step(presimulation_steps, synchronize=False)
        baseline_exc, baseline_inh = network.population_spike_counts()
        started = time.perf_counter()
        network.step(simulation_steps, synchronize=False)
        final_exc, final_inh = network.population_spike_counts()
        spike_count = int(
            (final_exc - baseline_exc).sum(dtype=np.uint64)
            + (final_inh - baseline_inh).sum(dtype=np.uint64)
        )
        wall_seconds = time.perf_counter() - started
    finally:
        network.close()
    return Result(
        spike_count=spike_count,
        wall_seconds=wall_seconds,
        simulation_steps=simulation_steps,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the selected GeNN CUDA workloads with minimal measurement overhead."
    )
    parser.add_argument("--cases", nargs="+", choices=ALL_CASES, default=ALL_CASES)
    parser.add_argument(
        "--precision", choices=("float", "double", "both"), default="float"
    )
    parser.add_argument("--mnist-seed", type=int, default=0)
    parser.add_argument("--brunel-seed", type=int, default=20260724)
    parser.add_argument("--mnist-samples", type=int, default=100)
    parser.add_argument("--brunel-presim-ms", type=float, default=100.0)
    parser.add_argument("--brunel-sim-ms", type=float, default=1000.0)
    parser.add_argument(
        "--brunel-indegree-scale", type=float, default=BRUNEL_INDEGREE_SCALE
    )
    parser.add_argument("--brunel-delay-ms", type=float, default=BRUNEL_DELAY_MS)
    parser.add_argument(
        "--brunel-recurrent-delivery-scale",
        type=float,
        default=BRUNEL_RECURRENT_DELIVERY_SCALE,
    )
    parser.add_argument(
        "--brunel-additive-external-rate-scale",
        type=float,
        default=BRUNEL_EXTERNAL_RATE_SCALES["additive"],
    )
    parser.add_argument(
        "--brunel-morrison-external-rate-scale",
        type=float,
        default=BRUNEL_EXTERNAL_RATE_SCALES["morrison"],
    )
    parser.add_argument("--data-path", type=Path, default=REPOSITORY / "data" / "mnist")
    parser.add_argument(
        "--triplet-checkpoint",
        type=Path,
        default=CHECKPOINT_DIRECTORY / "mnist_triplet_dense_010000.npz",
    )
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=CHECKPOINT_DIRECTORY / "mnist_one_trace_dense_010000.npz",
    )
    parser.add_argument(
        "--sparse-checkpoint",
        type=Path,
        default=CHECKPOINT_DIRECTORY / "mnist_one_trace_sparse_0125_010000.npz",
    )
    parser.add_argument("--work-dir", type=Path, default=REPOSITORY / "genn-sweep" / "work")
    parser.add_argument(
        "--reuse-build-root",
        type=Path,
        help="Reuse the builds directory from an earlier matching sweep.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mnist_samples <= 0:
        raise SystemExit("--mnist-samples must be positive")
    if args.brunel_presim_ms < 0.0 or args.brunel_sim_ms <= 0.0:
        raise SystemExit("Brunel presimulation must be non-negative and simulation positive")
    for name in (
        "brunel_indegree_scale",
        "brunel_recurrent_delivery_scale",
        "brunel_additive_external_rate_scale",
        "brunel_morrison_external_rate_scale",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.brunel_delay_ms) or args.brunel_delay_ms < 0.0:
        raise SystemExit("--brunel-delay-ms must be finite and non-negative")
    for name in ("brunel_delay_ms", "brunel_presim_ms", "brunel_sim_ms"):
        steps = getattr(args, name) / BRUNEL_DT_MS
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise SystemExit(
                f"--{name.replace('_', '-')} must be an integer number of timesteps"
            )
    if args.reuse_build_root is not None and not args.reuse_build_root.is_dir():
        raise SystemExit(f"reusable build root does not exist: {args.reuse_build_root}")
    args.work_dir.mkdir(parents=True, exist_ok=False)
    (args.work_dir / "logs").mkdir()
    if args.reuse_build_root is None:
        (args.work_dir / "builds").mkdir()

    precisions = ("float", "double") if args.precision == "both" else (args.precision,)
    for precision in precisions:
        for case_name in args.cases:
            log_path = args.work_dir / "logs" / f"{case_name}_{precision}.log"
            try:
                with redirect_process_output(log_path):
                    if case_name in MNIST_CASES:
                        result = run_mnist(args, case_name, precision)
                    else:
                        result = run_brunel(args, case_name, precision)
            except BaseException:
                print(f"{case_name} ({precision}) failed; see {log_path}", file=sys.stderr)
                raise
            brunel_config = ""
            if case_name in BRUNEL_CASES:
                rule = BRUNEL_CASES[case_name]
                external_rate_scale = getattr(
                    args, f"brunel_{rule}_external_rate_scale"
                )
                brunel_config = (
                    f"indegree_scale={args.brunel_indegree_scale} "
                    f"delay_ms={args.brunel_delay_ms} "
                    f"recurrent_delivery_scale={args.brunel_recurrent_delivery_scale} "
                    f"external_rate_scale={external_rate_scale} "
                )
            print(
                f"case={case_name} precision={precision} {brunel_config}"
                f"spike_count={result.spike_count} "
                f"wall_seconds={result.wall_seconds:.9f} "
                f"seconds_per_step={result.wall_seconds / result.simulation_steps:.12e}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
