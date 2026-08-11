#!/usr/bin/env python3
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
from zd3.io import load_checkpoint, load_mnist, sha256_file
from zd3.evaluation import simple_demo_accuracy
from zd3.variants import VARIANTS, get_variant, validate_checkpoint_topology


def _flatten_spikes(spike_lists: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    senders: list[int] = []
    for sender, neuron_times in enumerate(spike_lists):
        times.extend(neuron_times)
        senders.extend([sender] * len(neuron_times))
    return np.asarray(times, dtype=np.float64), np.asarray(senders, dtype=np.int64)


def _neuron_params(*, excitatory: bool) -> dict[str, Any]:
    if excitatory:
        return {
            "V_th": MODEL.exc_v_threshold_mv,
            "Delta_T": 0.001,
            "g_L": 1.0,
            "E_L": MODEL.exc_v_rest_mv,
            "C_m": MODEL.exc_tau_m_ms,
            "a": 0.0,
            "b": 0.0,
            "tau_w": MODEL.theta_tau_ms,
            "I_e": 0.0,
            "V_peak": MODEL.exc_v_threshold_mv,
            "V_reset": MODEL.exc_v_reset_mv,
            "t_ref": MODEL.exc_refractory_ms,
            "E_rev": [MODEL.exc_e_exc_mv, MODEL.exc_e_inh_mv],
            "tau_rise": [0.05, 0.05],
            "tau_decay": [MODEL.tau_ge_ms, MODEL.tau_gi_ms],
        }
    return {
        "V_th": MODEL.inh_v_threshold_mv,
        "Delta_T": 0.001,
        "g_L": 1.0,
        "E_L": MODEL.inh_v_rest_mv,
        "C_m": MODEL.inh_tau_m_ms,
        "a": 0.0,
        "b": 0.0,
        "tau_w": MODEL.theta_tau_ms,
        "I_e": 0.0,
        "V_peak": MODEL.inh_v_threshold_mv,
        "V_reset": MODEL.inh_v_reset_mv,
        "t_ref": MODEL.inh_refractory_ms,
        "E_rev": [MODEL.inh_e_exc_mv, MODEL.inh_e_inh_mv],
        "tau_rise": [0.05, 0.05],
        "tau_decay": [MODEL.tau_ge_ms, MODEL.tau_gi_ms],
    }


class NESTGPUCheckpointNetwork:
    def __init__(
        self,
        ngpu: Any,
        *,
        weights: np.ndarray,
        theta_mv: np.ndarray,
        structural_mask: np.ndarray,
        seed: int,
        record_capacity: int,
    ) -> None:
        self.ngpu = ngpu
        self.mask = structural_mask
        ngpu.SetKernelStatus(
            {
                "verbosity_level": 0,
                "rnd_seed": seed,
                "min_allowed_delay": MODEL.dt_ms,
                "time_resolution": MODEL.dt_ms,
                "spike_buffer_algo": 0,
            }
        )
        self.inputs = ngpu.Create("poisson_generator", MODEL.n_input, 1, {"rate": 0.0})
        self.exc = ngpu.Create(
            "aeif_cond_beta_multisynapse", MODEL.n_exc, 2, _neuron_params(excitatory=True)
        )
        self.inh = ngpu.Create(
            "aeif_cond_beta_multisynapse", MODEL.n_inh, 2, _neuron_params(excitatory=False)
        )
        thresholds = MODEL.exc_v_threshold_mv + theta_mv - MODEL.theta_offset_mv
        ngpu.SetStatus(
            self.exc,
            {
                "V_th": {"array": thresholds.tolist()},
                "V_peak": {"array": thresholds.tolist()},
            },
        )

        pre, post = np.nonzero(structural_mask)
        ngpu.Connect(
            pre.astype(np.int64).tolist(),
            (post + int(self.exc[0])).astype(np.int64).tolist(),
            {"rule": "one_to_one"},
            {
                "weight_array": weights[pre, post].astype(np.float32).tolist(),
                "delay": MODEL.dt_ms,
                "receptor": 0,
            },
        )
        ngpu.Connect(
            self.exc,
            self.inh,
            {"rule": "one_to_one"},
            {"weight": MODEL.exc_to_inh_weight, "delay": MODEL.dt_ms, "receptor": 0},
        )
        inh_pre = np.repeat(np.arange(MODEL.n_inh, dtype=np.int64), MODEL.n_exc - 1)
        exc_post = np.concatenate(
            [np.r_[0:i, i + 1 : MODEL.n_exc] for i in range(MODEL.n_inh)]
        ).astype(np.int64)
        ngpu.Connect(
            (inh_pre + int(self.inh[0])).tolist(),
            (exc_post + int(self.exc[0])).tolist(),
            {"rule": "one_to_one"},
            {
                "weight": MODEL.inference_inhibition,
                "delay": MODEL.dt_ms,
                "receptor": 1,
            },
        )
        ngpu.ActivateRecSpikeTimes(self.inputs, record_capacity)
        ngpu.ActivateRecSpikeTimes(self.exc, record_capacity)
        ngpu.ActivateRecSpikeTimes(self.inh, record_capacity)
        started = time.perf_counter()
        ngpu.Calibrate()
        self.calibration_wall_seconds = time.perf_counter() - started

    def set_image(self, image: np.ndarray, intensity: float) -> None:
        rates = image.astype(np.float64) / 8.0 * intensity
        self.ngpu.SetStatus(self.inputs, {"rate": {"array": rates.tolist()}})

    def set_zero_input(self) -> None:
        self.ngpu.SetStatus(self.inputs, {"rate": 0.0})

    def simulate(self, duration_ms: float) -> float:
        started = time.perf_counter()
        self.ngpu.Simulate(duration_ms)
        return time.perf_counter() - started

    def drain_spikes(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            "input": _flatten_spikes(self.ngpu.GetRecSpikeTimes(self.inputs)),
            "exc": _flatten_spikes(self.ngpu.GetRecSpikeTimes(self.exc)),
            "inh": _flatten_spikes(self.ngpu.GetRecSpikeTimes(self.inh)),
        }


def main() -> int:
    root = Path(__file__).resolve().parent
    repository = root.parent
    parser = argparse.ArgumentParser(
        description="Run a clearly labelled static-checkpoint MNIST validation on NEST-GPU."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--stats-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-path", type=Path, default=repository / "data" / "mnist")
    parser.add_argument("--max-intensity", type=float, default=20.0)
    parser.add_argument("--record-capacity", type=int, default=5000)
    args = parser.parse_args()
    if args.samples <= 0 or args.stats_interval <= 0 or args.record_capacity <= 0:
        raise SystemExit("samples, stats interval, and record capacity must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results").mkdir()

    try:
        import nestgpu as ngpu
    except ImportError as error:
        raise SystemExit("NEST-GPU is not available on PYTHONPATH/LD_LIBRARY_PATH") from error

    checkpoint = load_checkpoint(args.checkpoint)
    variant = get_variant(args.variant)
    mask = validate_checkpoint_topology(checkpoint.weights, variant)
    manifest = {
        "backend": "nest-gpu-static-checkpoint-validation",
        "command": sys.argv,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_samples": checkpoint.accepted_samples,
        "dataset": "mnist-test",
        "data_path": str(args.data_path),
        "host": platform.uname()._asdict(),
        "model": MODEL.as_dict(),
        "variant": variant.as_dict(),
        "samples": args.samples,
        "seed": args.seed,
        "structural_synapses": int(mask.sum()),
        "transport_delay_ms": MODEL.dt_ms,
        "plasticity": "disabled; checkpoint inference only",
        "non_equivalences": [
            "NEST-GPU synapse ABI has no accumulated source trace for exact one-trace STDP",
            "aeif_cond_beta_multisynapse beta conductances replace midpoint LIF exponential conductances",
            "beta-conductance rise time is 0.05 ms rather than an instantaneous conductance jump",
        ],
        "accuracy_protocol": "optimistic simple-demo assignment and scoring on the same test activity",
    }
    with (args.output / "manifest.json").open("x", encoding="ascii") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")

    data = load_mnist(args.data_path, "test")
    program_started = time.perf_counter()
    construction_started = time.perf_counter()
    network = NESTGPUCheckpointNetwork(
        ngpu,
        weights=checkpoint.weights,
        theta_mv=checkpoint.theta_mv,
        structural_mask=mask,
        seed=args.seed,
        record_capacity=args.record_capacity,
    )
    construction_wall = time.perf_counter() - construction_started
    activity = np.zeros((args.samples, MODEL.n_exc), dtype=np.uint16)
    attempts = 0
    simulation_wall = 0.0
    excitatory_spikes = 0
    inhibitory_spikes = 0
    stimulus_exc_spikes = 0
    interval_spikes = 0
    interval_active = 0
    interval_attempts = 0
    for sample in range(args.samples):
        intensity = MODEL.initial_intensity
        while True:
            attempts += 1
            interval_attempts += 1
            network.set_image(data.images[sample], intensity)
            simulation_wall += network.simulate(MODEL.stimulus_ms)
            stimulus = network.drain_spikes()
            exc_ids = stimulus["exc"][1]
            counts = np.bincount(exc_ids, minlength=MODEL.n_exc)
            stimulus_count = int(counts.sum())
            for name, (_, ids) in stimulus.items():
                if name == "exc":
                    excitatory_spikes += len(ids)
                elif name == "inh":
                    inhibitory_spikes += len(ids)
            network.set_zero_input()
            simulation_wall += network.simulate(MODEL.rest_ms)
            rest = network.drain_spikes()
            for name, (_, ids) in rest.items():
                if name == "exc":
                    excitatory_spikes += len(ids)
                elif name == "inh":
                    inhibitory_spikes += len(ids)
            if stimulus_count >= MODEL.minimum_exc_spikes:
                activity[sample] = counts
                stimulus_exc_spikes += stimulus_count
                interval_spikes += stimulus_count
                interval_active += int(np.count_nonzero(counts))
                break
            if intensity >= args.max_intensity:
                raise RuntimeError(f"sample {sample} exceeded max intensity")
            intensity += MODEL.intensity_increment
        if (sample + 1) % args.stats_interval == 0 or sample + 1 == args.samples:
            block = (sample % args.stats_interval) + 1
            row = {
                "accepted": sample + 1,
                "attempts": interval_attempts,
                "mean_stimulus_spikes": interval_spikes / block,
                "mean_active_exc": interval_active / block,
            }
            print("PERIODIC " + json.dumps(row, sort_keys=True), flush=True)
            interval_spikes = interval_active = interval_attempts = 0

    score = simple_demo_accuracy(activity, data.labels[: args.samples])
    ticks = attempts * MODEL.attempt_ticks
    whole_wall = time.perf_counter() - program_started
    recorded_firing = excitatory_spikes + inhibitory_spikes
    result = {
        "accuracy_percent": score["accuracy_percent"],
        "assigned_neurons": score["assigned_neurons"],
        "assignment_counts": score["assignment_counts"],
        "probe_samples": args.samples,
        "attempts": attempts,
        "retries": attempts - args.samples,
        "mean_stimulus_spikes": stimulus_exc_spikes / args.samples,
        "mean_active_exc": float(np.count_nonzero(activity, axis=1).mean()),
        "construction_wall_seconds": construction_wall,
        "calibration_wall_seconds": network.calibration_wall_seconds,
        "simulation_wall_seconds": simulation_wall,
        "whole_program_wall_seconds": whole_wall,
        "simulated_ticks": ticks,
        "seconds_per_timestep_cycle": simulation_wall / ticks,
        "input_spikes": None,
        "excitatory_spikes": excitatory_spikes,
        "inhibitory_spikes": inhibitory_spikes,
        "total_firing_count": recorded_firing,
        "firing_count_scope": "excitatory and inhibitory only; Poisson generator events are not recordable",
        "feedforward_pre_synapse_updates": None,
        "feedforward_post_synapse_updates": 0,
        "average_firing_count_per_timestep": recorded_firing / ticks,
        "average_pre_synapse_updates_per_timestep": None,
        "average_post_synapse_updates_per_timestep": 0.0,
        "seconds_per_pre_synapse_update": None,
        "seconds_per_post_synapse_update": None,
        "structural_synapses": int(mask.sum()),
        "variant": variant.name,
    }
    with (args.output / "results" / "performance.json").open("x", encoding="ascii") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    np.savez_compressed(
        args.output / "results" / "activity.npz",
        activity=activity,
        labels=data.labels[: args.samples],
    )
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
    print(f"RUN_COMPLETE backend=nest-gpu-static-checkpoint accepted={args.samples}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
