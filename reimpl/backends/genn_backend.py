from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from math import exp
from pathlib import Path
from typing import Any

import numpy as np

from zd3.constants import MODEL, ModelConstants
from zd3.evaluation import simple_demo_accuracy
from zd3.io import (
    load_checkpoint,
    load_mnist,
    load_reference_triplets,
    normalize_columns,
    save_checkpoint,
    sha256_file,
)
from zd3.stats import IntervalStats
from zd3.variants import (
    VARIANTS,
    NetworkVariant,
    connectivity_mask,
    get_variant,
    prepare_initial_weights,
    validate_checkpoint_topology,
    validate_normalized_weight_bound,
)


def _import_genn() -> dict[str, Any]:
    try:
        import pygenn
        from pygenn import (
            GeNNModel,
            ParallelismHint,
            create_neuron_model,
            create_weight_update_model,
            init_postsynaptic,
            init_sparse_connectivity,
            init_weight_update,
        )
        from pygenn.genn_model import backend_modules
    except ImportError as error:
        raise SystemExit(
            "PyGeNN is unavailable. Build the vendored GeNN package in the reimpl environment."
        ) from error
    return {
        "module": pygenn,
        "GeNNModel": GeNNModel,
        "ParallelismHint": ParallelismHint,
        "create_neuron_model": create_neuron_model,
        "create_weight_update_model": create_weight_update_model,
        "init_postsynaptic": init_postsynaptic,
        "init_sparse_connectivity": init_sparse_connectivity,
        "init_weight_update": init_weight_update,
        "backend_modules": backend_modules,
    }


def create_two_trace_model(create_weight_update_model: Any) -> Any:
    return create_weight_update_model(
        "ZD3TwoTracePower",
        params=["preTau", "postTau", "potentiationRate", "depressionRate",
                "preTarget", "preExponent", "postExponent", "weightMin",
                "weightMax", "plasticity"],
        vars=[("g", "scalar")],
        pre_vars=[("x", "scalar")],
        post_vars=[("y", "scalar")],
        pre_dynamics_code="x *= exp(-dt / preTau);",
        post_dynamics_code="y *= exp(-dt / postTau);",
        pre_spike_code="x += 1.0;",
        post_spike_code="y += 1.0;",
        pre_spike_syn_code="""
        addToPost(g);
        if (plasticity > 0.0) {
        // Neuron traces already include this tick's spikes. Brian processes
        // pre before post, so depression excludes a simultaneous post increment.
        const scalar postBefore = fmax(0.0, y - ((st_post == st_pre) ? 1.0 : 0.0));
        g = fmin(weightMax, fmax(weightMin, g - plasticity * depressionRate
            * postBefore * pow(g, preExponent)));
        }
        """,
        post_spike_syn_code="""
        if (plasticity > 0.0) {
        g = fmin(weightMax, fmax(weightMin, g + plasticity * potentiationRate
            * (x - preTarget) * pow(weightMax - g, postExponent)));
        }
        """,
    )


class GeNNNetwork:
    def __init__(
        self,
        *,
        weights: np.ndarray,
        theta_mv: np.ndarray,
        plasticity: bool,
        inhibition: float,
        seed: int,
        backend: str,
        build_path: Path,
        variant: NetworkVariant,
        structural_mask: np.ndarray,
        precision: str,
        parallelism: str,
        num_threads_per_spike: int,
        timing_enabled: bool,
        reuse_build: Path | None,
        record_spikes: bool = False,
        recording_steps: int = 0,
        weight_max_by_post: np.ndarray | None = None,
        constants: ModelConstants = MODEL,
        integer_timestamps: bool = False,
    ) -> None:
        self.constants = constants
        self.integer_timestamps = integer_timestamps
        if integer_timestamps and variant.learning_rule != "three-trace":
            raise ValueError("integer image timestamps currently require three-trace STDP")
        api = _import_genn()
        if backend not in api["backend_modules"]:
            available = ", ".join(api["backend_modules"])
            raise RuntimeError(
                f"GeNN backend {backend!r} is unavailable in this PyGeNN build; available: {available}"
            )
        create_neuron_model = api["create_neuron_model"]
        if integer_timestamps:
            def create_neuron_model(name, **definition):
                if name in ("ZD3PoissonInput", "ZD3Excitatory"):
                    definition["vars"] = [*definition.get("vars", []),
                        ("tick", "uint32_t"), ("lastSpikeTick", "uint32_t"),
                        ("previousSpikeTick", "uint32_t")]
                    # Zero means no spike history; emission ticks are one-based.
                    # These image connections all have the same next-tick delivery.
                    definition["sim_code"] = (
                        "assert(tick < 4294967295u); tick++;\n"
                        + definition.get("sim_code", ""))
                    definition["reset_code"] = (
                        "previousSpikeTick = lastSpikeTick; lastSpikeTick = tick;\n"
                        + definition.get("reset_code", ""))
                return api["create_neuron_model"](name, **definition)
        create_weight_update_model = api["create_weight_update_model"]
        init_postsynaptic = api["init_postsynaptic"]
        init_sparse_connectivity = api["init_sparse_connectivity"]
        init_weight_update = api["init_weight_update"]
        self.variant = variant
        self.scalar_dtype = np.float32 if precision == "float" else np.float64
        self.structural_mask = np.asarray(structural_mask, dtype=bool)
        if self.structural_mask.shape != (self.constants.n_input, self.constants.n_exc):
            raise ValueError("invalid feedforward structural mask shape")
        self._feedforward_pre, self._feedforward_post = np.nonzero(
            self.structural_mask
        )
        self._feedforward_pre = self._feedforward_pre.astype(np.uint32)
        self._feedforward_post = self._feedforward_post.astype(np.uint32)
        self._feedforward_outdegree = self.structural_mask.sum(axis=1).astype(np.int64)
        self._feedforward_indegree = self.structural_mask.sum(axis=0).astype(np.int64)
        self.weight_max_by_post = (None if weight_max_by_post is None else
                                   np.asarray(weight_max_by_post, dtype=self.scalar_dtype).copy())
        cap_initial = {}
        if self.weight_max_by_post is not None:
            if (self.weight_max_by_post.shape != (self.constants.n_exc,) or
                    not np.all(np.isfinite(self.weight_max_by_post)) or
                    np.any(self.weight_max_by_post < 0) or
                    np.any(self.weight_max_by_post[self._feedforward_indegree > 0] <= 0)):
                raise ValueError("invalid postsynaptic weight caps")
            cap_initial = {"weightMax": self.weight_max_by_post}
            # Share exactly the existing learning kernels; only the storage of
            # weightMax changes from a population parameter to a static post var.
            create_uncapped = create_weight_update_model
            def create_weight_update_model(name, **definition):
                if "weightMax" in definition.get("params", []):
                    definition["params"] = [p for p in definition["params"] if p != "weightMax"]
                    definition["post_vars"] = [*definition.get("post_vars", []),
                        ("weightMax", "scalar", api["module"].VarAccess.READ_ONLY)]
                return create_uncapped(name, **definition)

        input_model = create_neuron_model(
            "ZD3PoissonInput",
            vars=[("rateHz", "scalar"), ("spikeCount", "unsigned int")],
            threshold_condition_code="gennrand_uniform() < (rateHz * dt * 0.001)",
            reset_code="spikeCount++;",
        )
        exc_model = create_neuron_model(
            "ZD3Excitatory",
            params=[
                "tauM",
                "vRest",
                "vReset",
                "vThreshold",
                "eExc",
                "eInh",
                "geHalfDecay",
                "giHalfDecay",
                "geDecay",
                "giDecay",
                "thetaOffset",
                "thetaPlus",
                "thetaDecay",
                "refractorySteps",
            ],
            vars=[
                ("V", "scalar"),
                ("ge", "scalar"),
                ("gi", "scalar"),
                ("theta", "scalar"),
                ("refrac", "unsigned int"),
                ("spikeCount", "unsigned int"),
            ],
            additional_input_vars=[("geIn", "scalar", 0.0), ("giIn", "scalar", 0.0)],
            sim_code="""
            ge += geIn;
            gi += giIn;
            if (refrac > 0) {
                refrac--;
            }
            else {
                const scalar geMid = ge * geHalfDecay;
                const scalar giMid = gi * giHalfDecay;
                const scalar gMid = 1.0 + geMid + giMid;
                const scalar vInf = (vRest + (geMid * eExc) + (giMid * eInh)) / gMid;
                V = vInf + ((V - vInf) * exp(-(dt * gMid) / tauM));
                ge *= geDecay;
                gi *= giDecay;
                theta *= thetaDecay;
            }
            """,
            threshold_condition_code="(refrac == 0) && (V > (theta - thetaOffset + vThreshold))",
            reset_code="""
            V = vReset;
            refrac = (unsigned int)refractorySteps;
            theta += thetaPlus;
            spikeCount++;
            """,
        )
        inh_model = create_neuron_model(
            "ZD3Inhibitory",
            params=[
                "tauM",
                "vRest",
                "vReset",
                "vThreshold",
                "eExc",
                "eInh",
                "geHalfDecay",
                "giHalfDecay",
                "geDecay",
                "giDecay",
                "refractorySteps",
            ],
            vars=[
                ("V", "scalar"),
                ("ge", "scalar"),
                ("gi", "scalar"),
                ("refrac", "unsigned int"),
                ("spikeCount", "unsigned int"),
            ],
            additional_input_vars=[("geIn", "scalar", 0.0), ("giIn", "scalar", 0.0)],
            sim_code="""
            ge += geIn;
            gi += giIn;
            if (refrac > 0) {
                refrac--;
            }
            else {
                const scalar geMid = ge * geHalfDecay;
                const scalar giMid = gi * giHalfDecay;
                const scalar gMid = 1.0 + geMid + giMid;
                const scalar vInf = (vRest + (geMid * eExc) + (giMid * eInh)) / gMid;
                V = vInf + ((V - vInf) * exp(-(dt * gMid) / tauM));
                ge *= geDecay;
                gi *= giDecay;
            }
            """,
            threshold_condition_code="(refrac == 0) && (V > vThreshold)",
            reset_code="V = vReset; refrac = (unsigned int)refractorySteps; spikeCount++;",
        )
        triplet_model = create_weight_update_model(
            "ZD3Triplet",
            params=[
                "depressionRate",
                "potentiationRate",
                "weightMin",
                "weightMax",
                "preTau",
                "post1Tau",
                "post2Tau",
                "plasticity",
            ],
            vars=[("g", "scalar")],
            pre_spike_syn_code="""
            addToPost(g);
            if (plasticity > 0.0) {
            const scalar postTimeBeforePre = (st_post < st_pre) ? st_post : prev_st_post;
            const scalar post1Before = (postTimeBeforePre > -1.0e20)
                ? exp(-(st_pre - postTimeBeforePre) / post1Tau) : 0.0;
            g = fmin(weightMax, fmax(weightMin,
                g - (plasticity * depressionRate * post1Before)));
            }
            """,
            post_spike_syn_code="""
            if (plasticity > 0.0) {
            const scalar preTimeBeforePost = (st_pre <= st_post) ? st_pre : prev_st_pre;
            const scalar preBefore = (preTimeBeforePost > -1.0e20)
                ? exp(-(st_post - preTimeBeforePost) / preTau) : 0.0;
            const scalar post2Before = (prev_st_post > -1.0e20)
                ? exp(-(st_post - prev_st_post) / post2Tau) : 0.0;
            g = fmin(weightMax, fmax(weightMin,
                g + (plasticity * potentiationRate * preBefore * post2Before)));
            }
            """,
        )
        if integer_timestamps:
            triplet_model = create_weight_update_model(
                "ZD3TripletIntegerTicks",
                params=["depressionRate", "potentiationRate", "weightMin",
                        "weightMax", "preTau", "post1Tau", "post2Tau", "plasticity"],
                vars=[("g", "scalar")],
                pre_neuron_var_refs=[("preTick", "uint32_t"), ("previousPreTick", "uint32_t")],
                post_neuron_var_refs=[("postTick", "uint32_t"), ("previousPostTick", "uint32_t")],
                pre_spike_syn_code="""
                addToPost(g);
                if (plasticity > 0.0) {
                    assert(preTick > 0u);
                    const uint32_t precedingPost = (postTick < preTick) ? postTick : previousPostTick;
                    assert(precedingPost <= preTick);
                    const scalar lag = (scalar)(preTick - precedingPost) * dt;
                    const scalar post1Before = (precedingPost > 0u) ? exp(-lag / post1Tau) : 0.0;
                    g = fmin(weightMax, fmax(weightMin, g - plasticity * depressionRate * post1Before));
                }
                """,
                post_spike_syn_code="""
                if (plasticity > 0.0) {
                    assert(postTick > 0u);
                    const uint32_t precedingPre = (preTick <= postTick) ? preTick : previousPreTick;
                    assert(precedingPre <= postTick && previousPostTick < postTick);
                    const scalar preLag = (scalar)(postTick - precedingPre) * dt;
                    const scalar postLag = (scalar)(postTick - previousPostTick) * dt;
                    const scalar preBefore = (precedingPre > 0u) ? exp(-preLag / preTau) : 0.0;
                    const scalar post2Before = (previousPostTick > 0u) ? exp(-postLag / post2Tau) : 0.0;
                    g = fmin(weightMax, fmax(weightMin,
                        g + plasticity * potentiationRate * preBefore * post2Before));
                }
                """,
            )
        one_trace_model = create_weight_update_model(
            "ZD3OneTracePower",
            params=[
                "preTau",
                "potentiationRate",
                "preTarget",
                "postExponent",
                "weightMin",
                "weightMax",
                "plasticity",
            ],
            vars=[("g", "scalar")],
            pre_vars=[("x", "scalar")],
            pre_dynamics_code="x *= exp(-dt / preTau);",
            pre_spike_code="x += 1.0;",
            pre_spike_syn_code="addToPost(g);",
            post_spike_syn_code="""
            if (plasticity > 0.0) {
            const scalar delta = plasticity * potentiationRate * (x - preTarget)
                * pow(weightMax - g, postExponent);
            g = fmin(weightMax, fmax(weightMin, g + delta));
            }
            """,
        )
        variable_pulse_model = create_weight_update_model(
            "ZD3VariablePulse",
            vars=[("g", "scalar")],
            pre_spike_syn_code="addToPost(g);",
        )

        if backend == "cuda":
            cuda_path = os.environ.get("CUDA_PATH")
            cuda_host_cxx = os.environ.get("CUDAHOSTCXX")
            prepend_flags = []
            if cuda_path:
                prepend_flags.append(f"-I{cuda_path}/include")
            if cuda_host_cxx:
                if not Path(cuda_host_cxx).is_file():
                    raise ValueError(f"CUDAHOSTCXX does not exist: {cuda_host_cxx}")
                prepend_flags.append(f"-ccbin={cuda_host_cxx}")
            existing_flags = os.environ.get("NVCC_PREPEND_FLAGS", "")
            for flag in reversed(prepend_flags):
                if flag not in existing_flags.split():
                    existing_flags = f"{flag} {existing_flags}".strip()
            if existing_flags:
                os.environ["NVCC_PREPEND_FLAGS"] = existing_flags
        self.model = api["GeNNModel"](precision, "zd3_genn", backend=backend)
        self.model.dt = self.constants.dt_ms
        self.model.seed = seed
        self.timing_enabled = timing_enabled
        self.model.timing_enabled = timing_enabled
        self.inputs = self.model.add_neuron_population(
            "Input", self.constants.n_input, input_model, {},
            {"rateHz": 0.0, "spikeCount": 0,
             **({"tick": 0, "lastSpikeTick": 0, "previousSpikeTick": 0} if integer_timestamps else {})}
        )
        common_decay = {
            "geHalfDecay": exp(-0.5 * self.constants.dt_ms / self.constants.tau_ge_ms),
            "giHalfDecay": exp(-0.5 * self.constants.dt_ms / self.constants.tau_gi_ms),
            "geDecay": exp(-self.constants.dt_ms / self.constants.tau_ge_ms),
            "giDecay": exp(-self.constants.dt_ms / self.constants.tau_gi_ms),
        }
        exc_params = {
            "tauM": self.constants.exc_tau_m_ms,
            "vRest": self.constants.exc_v_rest_mv,
            "vReset": self.constants.exc_v_reset_mv,
            "vThreshold": self.constants.exc_v_threshold_mv,
            "eExc": self.constants.exc_e_exc_mv,
            "eInh": self.constants.exc_e_inh_mv,
            "thetaOffset": self.constants.theta_offset_mv,
            "thetaPlus": self.constants.theta_plus_mv if plasticity else 0.0,
            "thetaDecay": exp(-self.constants.dt_ms / self.constants.theta_tau_ms) if plasticity else 1.0,
            "refractorySteps": round(self.constants.exc_refractory_ms / self.constants.dt_ms),
            **common_decay,
        }
        inh_params = {
            "tauM": self.constants.inh_tau_m_ms,
            "vRest": self.constants.inh_v_rest_mv,
            "vReset": self.constants.inh_v_reset_mv,
            "vThreshold": self.constants.inh_v_threshold_mv,
            "eExc": self.constants.inh_e_exc_mv,
            "eInh": self.constants.inh_e_inh_mv,
            "refractorySteps": round(self.constants.inh_refractory_ms / self.constants.dt_ms),
            **common_decay,
        }
        self.exc = self.model.add_neuron_population(
            "Exc",
            self.constants.n_exc,
            exc_model,
            exc_params,
            {
                "V": self.constants.exc_v_rest_mv - 40.0,
                "ge": 0.0,
                "gi": 0.0,
                "theta": np.asarray(theta_mv, dtype=self.scalar_dtype),
                "refrac": 0,
                "spikeCount": 0,
                **({"tick": 0, "lastSpikeTick": 0, "previousSpikeTick": 0} if integer_timestamps else {}),
            },
        )
        self.inh = self.model.add_neuron_population(
            "Inh",
            self.constants.n_inh,
            inh_model,
            inh_params,
            {
                "V": self.constants.inh_v_rest_mv - 40.0,
                "ge": 0.0,
                "gi": 0.0,
                "refrac": 0,
                "spikeCount": 0,
            },
        )
        if record_spikes:
            if recording_steps <= 0:
                raise ValueError("recording_steps must be positive when recording spikes")
            self.inputs.spike_recording_enabled = True
            self.exc.spike_recording_enabled = True
            self.inh.spike_recording_enabled = True
        triplet_params = {
            "depressionRate": variant.depression_rate,
            "potentiationRate": variant.potentiation_rate,
            "weightMin": self.constants.weight_min,
            "weightMax": variant.weight_max,
            "preTau": self.constants.pre_tau_ms,
            "post1Tau": self.constants.post1_tau_ms,
            "post2Tau": self.constants.post2_tau_ms,
            "plasticity": 1.0 if plasticity else 0.0,
        }
        one_trace_params = {
            "preTau": self.constants.pre_tau_ms,
            "potentiationRate": variant.potentiation_rate,
            "preTarget": variant.pre_trace_target,
            "postExponent": variant.post_weight_exponent,
            "weightMin": self.constants.weight_min,
            "weightMax": variant.weight_max,
            "plasticity": 1.0 if plasticity else 0.0,
        }
        if cap_initial:
            del triplet_params["weightMax"]
            del one_trace_params["weightMax"]
        if variant.learning_rule == "three-trace":
            feedforward_update = init_weight_update(
                triplet_model,
                triplet_params,
                {"g": np.asarray(weights[self.structural_mask], dtype=self.scalar_dtype)},
                post_vars=cap_initial,
                **({"pre_var_refs": {"preTick": "lastSpikeTick", "previousPreTick": "previousSpikeTick"},
                    "post_var_refs": {"postTick": "lastSpikeTick", "previousPostTick": "previousSpikeTick"}}
                   if integer_timestamps else {}),
            )
        elif variant.learning_rule == "one-trace-power":
            feedforward_update = init_weight_update(
                one_trace_model,
                one_trace_params,
                {"g": np.asarray(weights[self.structural_mask], dtype=self.scalar_dtype)},
                pre_vars={"x": 0.0},
                post_vars=cap_initial,
            )
        elif variant.learning_rule == "two-trace-power":
            two_trace_model = create_two_trace_model(create_weight_update_model)
            feedforward_update = init_weight_update(
                two_trace_model,
                {**one_trace_params, "postTau": self.constants.post1_tau_ms,
                 "depressionRate": variant.depression_rate,
                 "preExponent": variant.pre_weight_exponent},
                {"g": np.asarray(weights[self.structural_mask], dtype=self.scalar_dtype)},
                pre_vars={"x": 0.0},
                post_vars={"y": 0.0, **cap_initial},
            )
        else:
            raise ValueError(f"unsupported learning rule: {variant.learning_rule}")
        self.feedforward = self.model.add_synapse_population(
            "Feedforward",
            "SPARSE",
            self.inputs,
            self.exc,
            feedforward_update,
            init_postsynaptic("DeltaCurr"),
        )
        self.feedforward.set_sparse_connections(
            self._feedforward_pre, self._feedforward_post
        )
        if parallelism == "presynaptic":
            self.feedforward.parallelism_hint = api["ParallelismHint"].PRESYNAPTIC
        elif parallelism == "postsynaptic":
            self.feedforward.parallelism_hint = api["ParallelismHint"].POSTSYNAPTIC
        else:
            raise ValueError(f"unsupported parallelism: {parallelism}")
        self.feedforward.num_threads_per_spike = num_threads_per_spike
        self.feedforward.post_target_var = "geIn"
        self.exc_to_inh = self.model.add_synapse_population(
            "ExcToInh",
            "SPARSE",
            self.exc,
            self.inh,
            init_weight_update("StaticPulseConstantWeight", {"g": self.constants.exc_to_inh_weight}),
            init_postsynaptic("DeltaCurr"),
            init_sparse_connectivity("OneToOne"),
        )
        self.exc_to_inh.post_target_var = "geIn"
        inhibitory_weights = np.full(
            (self.constants.n_inh, self.constants.n_exc), inhibition, dtype=self.scalar_dtype
        )
        np.fill_diagonal(inhibitory_weights, 0.0)
        self.inh_to_exc = self.model.add_synapse_population(
            "InhToExc",
            "DENSE",
            self.inh,
            self.exc,
            init_weight_update(
                variable_pulse_model, {}, {"g": inhibitory_weights.reshape(-1)}
            ),
            init_postsynaptic("DeltaCurr"),
        )
        self.inh_to_exc.post_target_var = "giIn"

        if reuse_build is not None:
            build_path = reuse_build.resolve()
            if not (build_path / "zd3_genn_CODE" / "librunner.so").is_file():
                raise ValueError(f"invalid reusable GeNN build: {build_path}")
        else:
            build_path.mkdir(parents=True, exist_ok=False)
        build_started = time.perf_counter()
        self.model.build(str(build_path), never_rebuild=reuse_build is not None)
        self.build_wall_seconds = time.perf_counter() - build_started
        if record_spikes:
            self.model.load(num_recording_timesteps=recording_steps)
        else:
            self.model.load()
        self._count_baseline = np.zeros(self.constants.n_exc, dtype=np.uint32)
        self._record_spikes = record_spikes

    def close(self) -> None:
        self.model.unload()

    def kernel_timing(self) -> dict[str, float] | None:
        if not self.timing_enabled:
            return None
        return {
            "neuron_update_seconds": self.model.neuron_update_time,
            "presynaptic_update_seconds": self.model.presynaptic_update_time,
            "postsynaptic_update_seconds": self.model.postsynaptic_update_time,
        }

    def set_image(self, pixels: np.ndarray, intensity: float) -> None:
        variable = self.inputs.vars["rateHz"]
        variable.view[:] = pixels.astype(self.scalar_dtype) / 8.0 * intensity
        variable.push_to_device()

    def set_zero_input(self) -> None:
        variable = self.inputs.vars["rateHz"]
        variable.view[:] = 0.0
        variable.push_to_device()

    def _step(self, ticks: int) -> None:
        for _ in range(ticks):
            self.model.step_time()

    def run_stimulus(self) -> np.ndarray:
        self._step(self.constants.stimulus_ticks)
        variable = self.exc.vars["spikeCount"]
        variable.pull_from_device()
        current = np.asarray(variable.view, dtype=np.uint32).copy()
        counts = current.astype(np.int64) - self._count_baseline.astype(np.int64)
        self._count_baseline = current
        return counts

    def run_rest(self, *, synchronize: bool = True) -> None:
        self.set_zero_input()
        self._step(self.constants.rest_ticks)
        if synchronize and not self.timing_enabled:
            # GeNN event timing synchronizes every tick. Without it, synchronize
            # once at the attempt boundary so wall timing includes the final rest.
            self.exc.vars["spikeCount"].pull_from_device()

    def weights(self) -> np.ndarray:
        variable = self.feedforward.vars["g"]
        variable.pull_from_device()
        dense = np.zeros((self.constants.n_input, self.constants.n_exc), dtype=np.float64)
        dense[self._feedforward_pre, self._feedforward_post] = np.asarray(
            variable.values, dtype=np.float64
        )
        return dense

    def normalize(self, *, validate: bool = True, check_weight_bound: bool = True) -> None:
        variable = self.feedforward.vars["g"]
        variable.pull_from_device()
        weights = np.zeros((self.constants.n_input, self.constants.n_exc), dtype=np.float64)
        weights[self._feedforward_pre, self._feedforward_post] = np.asarray(
            variable.values, dtype=np.float64
        )
        if validate:
            normalize_columns(weights, self.constants.normalization_target)
            if check_weight_bound:
                validate_normalized_weight_bound(weights, self.variant,
                                                 weight_max_by_post=self.weight_max_by_post)
        else:
            sums = weights.sum(axis=0, dtype=np.float64)
            weights *= (self.constants.normalization_target / sums)[None, :]
        variable.values = weights[self._feedforward_pre, self._feedforward_post]
        variable.push_to_device()

    def total_spike_count(self) -> int:
        return int(
            self._pulled(self.inputs, "spikeCount").sum(dtype=np.uint64)
            + self._pulled(self.exc, "spikeCount").sum(dtype=np.uint64)
            + self._pulled(self.inh, "spikeCount").sum(dtype=np.uint64)
        )

    def theta_mv(self) -> np.ndarray:
        variable = self.exc.vars["theta"]
        variable.pull_from_device()
        return np.asarray(variable.view, dtype=np.float64).copy()

    def recorded_spikes(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if not self._record_spikes:
            raise RuntimeError("spike recording was disabled for this model")
        self.model.pull_recording_buffers_from_device()
        result = {}
        for name, population in (
            ("input", self.inputs),
            ("excitatory", self.exc),
            ("inhibitory", self.inh),
        ):
            times, ids = population.spike_recording_data[0]
            result[name] = (
                np.asarray(times, dtype=np.float64).copy(),
                np.asarray(ids, dtype=np.int64).copy(),
            )
        return result

    @staticmethod
    def _pulled(population: Any, name: str) -> np.ndarray:
        variable = population.vars[name]
        variable.pull_from_device()
        return np.asarray(variable.view)

    def runtime_diagnostics(self) -> dict[str, float]:
        e_v = self._pulled(self.exc, "V")
        i_v = self._pulled(self.inh, "V")
        return {
            "e_v_min_mv": float(np.min(e_v)),
            "e_v_max_mv": float(np.max(e_v)),
            "i_v_min_mv": float(np.min(i_v)),
            "i_v_max_mv": float(np.max(i_v)),
            "e_ge_max": float(np.max(self._pulled(self.exc, "ge"))),
            "e_gi_max": float(np.max(self._pulled(self.exc, "gi"))),
            "i_ge_max": float(np.max(self._pulled(self.inh, "ge"))),
        }

    def event_counters(self) -> dict[str, int]:
        input_spikes = self._pulled(self.inputs, "spikeCount").astype(np.int64)
        exc_spikes = self._pulled(self.exc, "spikeCount").astype(np.int64)
        inh_spikes = self._pulled(self.inh, "spikeCount").astype(np.int64)
        return {
            "input_spikes": int(input_spikes.sum()),
            "excitatory_spikes": int(exc_spikes.sum()),
            "inhibitory_spikes": int(inh_spikes.sum()),
            "total_firing_count": int(
                input_spikes.sum() + exc_spikes.sum() + inh_spikes.sum()
            ),
            "feedforward_pre_synapse_updates": int(
                np.dot(input_spikes, self._feedforward_outdegree)
            ),
            "feedforward_post_synapse_updates": int(
                np.dot(exc_spikes, self._feedforward_indegree)
            ),
        }

    def validate_runtime(self, counts: np.ndarray, runaway_spikes: int) -> None:
        if int(counts.sum()) >= runaway_spikes:
            raise RuntimeError(f"runaway activity: {int(counts.sum())} E spikes")
        for population, names in (
            (self.exc, ("V", "ge", "gi", "theta")),
            (self.inh, ("V", "ge", "gi")),
        ):
            for name in names:
                variable = population.vars[name]
                variable.pull_from_device()
                if not np.all(np.isfinite(variable.view)):
                    raise RuntimeError(f"non-finite {population.name}.{name}")


def _create_run_directory(path: Path, manifest: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    (path / "checkpoints").mkdir()
    (path / "results").mkdir()
    with (path / "manifest.json").open("x", encoding="ascii") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _initial_state(
    args: argparse.Namespace, variant: NetworkVariant
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict[str, Any]]:
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        checkpoint_variant = checkpoint.manifest.get("variant", {}).get("name")
        if checkpoint_variant is not None and checkpoint_variant != variant.name:
            raise ValueError(
                f"checkpoint variant {checkpoint_variant!r} does not match {variant.name!r}"
            )
        mask = validate_checkpoint_topology(checkpoint.weights, variant)
        return (
            checkpoint.weights.copy(),
            checkpoint.theta_mv.copy(),
            mask,
            checkpoint.accepted_samples,
            {"resume_checkpoint": str(args.resume), "resume_sha256": sha256_file(args.resume)},
        )
    initial_weights = args.initial_weights
    if initial_weights is None:
        repository = Path(__file__).resolve().parents[2]
        initial_weights = (
            repository
            / "ref"
            / "zero_delay_one_trace_v1"
            / "full_nupost0005_30000"
            / "random"
            / "XeAe.npy"
            if variant.learning_rule == "one-trace-power"
            else repository
            / "ref"
            / "zero_delay_midpoint_v1"
            / "inhib150_full"
            / "random"
            / "XeAe.npy"
        )
    weights, mask = prepare_initial_weights(
        load_reference_triplets(initial_weights, MODEL.n_input, MODEL.n_exc), variant
    )
    return (
        weights,
        np.full(MODEL.n_exc, MODEL.theta_initial_mv, dtype=np.float64),
        mask,
        0,
        {
            "initial_weights": str(initial_weights),
            "initial_weights_sha256": sha256_file(initial_weights),
        },
    )


def _counter_metrics(counters: dict[str, int], ticks: int, wall: float) -> dict[str, float | int]:
    pre_updates = counters["feedforward_pre_synapse_updates"]
    post_updates = counters["feedforward_post_synapse_updates"]
    return {
        **counters,
        "average_firing_count_per_timestep": counters["total_firing_count"] / ticks,
        "average_pre_synapse_updates_per_timestep": pre_updates / ticks,
        "average_post_synapse_updates_per_timestep": post_updates / ticks,
        "seconds_per_pre_synapse_update": wall / pre_updates if pre_updates else None,
        "seconds_per_post_synapse_update": wall / post_updates if post_updates else None,
    }


def train(args: argparse.Namespace) -> int:
    program_started = time.perf_counter()
    variant = get_variant(args.variant)
    weights, theta, mask, start_sample, provenance = _initial_state(args, variant)
    api = _import_genn()
    manifest = {
        "backend": f"genn-{args.backend}",
        "command": sys.argv,
        "dataset": "mnist",
        "data_path": str(args.data_path),
        "genn_version": api["module"].__version__,
        "cuda_path": os.environ.get("CUDA_PATH"),
        "cuda_host_cxx": os.environ.get("CUDAHOSTCXX"),
        "nvcc_prepend_flags": os.environ.get("NVCC_PREPEND_FLAGS"),
        "host": platform.uname()._asdict(),
        "model": MODEL.as_dict(),
        "variant": variant.as_dict(),
        "structural_synapses": int(mask.sum()),
        "actual_connection_rate": float(mask.mean()),
        "feedforward_storage": "SPARSE",
        "precision": args.precision,
        "parallelism": args.parallelism,
        "num_threads_per_spike": args.num_threads_per_spike,
        "genn_timing_enabled": args.genn_timing,
        "profile_accounting_enabled": args.profile_accounting,
        "reuse_build": str(args.reuse_build.resolve()) if args.reuse_build else None,
        "requested_samples": args.samples,
        "rng_seed": args.seed,
        "start_sample": start_sample,
        **provenance,
    }
    _create_run_directory(args.output, manifest)
    data = load_mnist(args.data_path, "train")
    network = GeNNNetwork(
        weights=weights,
        theta_mv=theta,
        plasticity=True,
        inhibition=MODEL.train_inhibition,
        seed=args.seed,
        backend=args.backend,
        build_path=args.output / "build" if args.reuse_build is None else args.reuse_build,
        variant=variant,
        structural_mask=mask,
        precision=args.precision,
        parallelism=args.parallelism,
        num_threads_per_spike=args.num_threads_per_spike,
        timing_enabled=args.genn_timing,
        reuse_build=args.reuse_build,
    )
    print(
        f"CONFIG backend=genn-{args.backend} mode=train dt_ms={MODEL.dt_ms:.6f} "
        f"input_delay_ms=0.000000 transport_delay_ms={MODEL.dt_ms:.6f} "
        f"samples={args.samples} seed={args.seed} variant={variant.name} "
        f"precision={args.precision} parallelism={args.parallelism} "
        f"threads_per_spike={args.num_threads_per_spike} "
        f"structural_synapses={int(mask.sum())}",
        flush=True,
    )
    interval = IntervalStats()
    attempts = 0
    wall = 0.0
    component_wall = {
        "normalization": 0.0,
        "input_rate_push": 0.0,
        "stimulus_and_count_pull": 0.0,
        "runtime_validation": 0.0,
        "zero_rate_push_and_rest": 0.0,
        "periodic_diagnostics": 0.0,
        "checkpoint": 0.0,
    }
    stats_events = 0
    checkpoint_events = 0
    accepted = start_sample

    def measured(name: str, function: Any, *function_args: Any) -> Any:
        if not args.profile_accounting:
            return function(*function_args)
        component_started = time.perf_counter()
        value = function(*function_args)
        component_wall[name] += time.perf_counter() - component_started
        return value

    try:
        while accepted < args.samples:
            intensity = MODEL.initial_intensity
            while True:
                started = time.perf_counter()
                measured("normalization", network.normalize)
                measured(
                    "input_rate_push",
                    network.set_image,
                    data.images[accepted % len(data.images)],
                    intensity,
                )
                counts = measured("stimulus_and_count_pull", network.run_stimulus)
                measured("runtime_validation", network.validate_runtime, counts, args.runaway_spikes)
                retry = int(counts.sum()) < MODEL.minimum_exc_spikes
                measured("zero_rate_push_and_rest", network.run_rest)
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
                diagnostics_started = time.perf_counter()
                print(
                    interval.format(
                        accepted=accepted,
                        weights=network.weights(),
                        theta_mv=network.theta_mv(),
                        interval=args.stats_interval,
                        backend=f"genn-{args.backend}",
                        weight_max=variant.weight_max,
                        runtime=network.runtime_diagnostics(),
                    ),
                    flush=True,
                )
                if args.profile_accounting:
                    component_wall["periodic_diagnostics"] += (
                        time.perf_counter() - diagnostics_started
                    )
                stats_events += 1
                interval = IntervalStats()
            if accepted % args.checkpoint_interval == 0 or accepted == args.samples:
                checkpoint_started = time.perf_counter()
                path = args.output / "checkpoints" / f"checkpoint_{accepted:06d}.npz"
                save_checkpoint(
                    path,
                    weights=network.weights(),
                    theta_mv=network.theta_mv(),
                    accepted_samples=accepted,
                    manifest={
                        "backend": f"genn-{args.backend}",
                        "rng_seed": args.seed,
                        "source_run": str(args.output),
                        "runtime_state_scope": "weights-and-theta-only",
                        "variant": variant.as_dict(),
                        "structural_synapses": int(mask.sum()),
                    },
                )
                print(f"CHECKPOINT backend=genn-{args.backend} accepted={accepted} path={path}", flush=True)
                if args.profile_accounting:
                    component_wall["checkpoint"] += time.perf_counter() - checkpoint_started
                checkpoint_events += 1
        counters = network.event_counters()
        kernel_timing = network.kernel_timing()
    finally:
        network.close()
    ticks = attempts * MODEL.attempt_ticks
    scalar_bytes = np.dtype(np.float32 if args.precision == "float" else np.float64).itemsize
    uint_bytes = np.dtype(np.uint32).itemsize
    structural_synapses = int(mask.sum())
    transfer_bytes = {
        "normalization_device_to_host": attempts * structural_synapses * scalar_bytes,
        "normalization_host_to_device": attempts * structural_synapses * scalar_bytes,
        "input_rate_host_to_device": attempts * MODEL.n_input * scalar_bytes,
        "zero_rate_host_to_device": attempts * MODEL.n_input * scalar_bytes,
        "stimulus_spike_count_device_to_host": attempts * MODEL.n_exc * uint_bytes,
        "runtime_validation_device_to_host": attempts
        * ((4 * MODEL.n_exc) + (3 * MODEL.n_inh))
        * scalar_bytes,
        "periodic_diagnostics_device_to_host": stats_events
        * (
            (structural_synapses * scalar_bytes)
            + (MODEL.n_exc * scalar_bytes)
            + (((4 * MODEL.n_exc) + (3 * MODEL.n_inh)) * scalar_bytes)
        ),
        "checkpoint_device_to_host": checkpoint_events
        * ((structural_synapses + MODEL.n_exc) * scalar_bytes),
        "final_counter_device_to_host": (MODEL.n_input + MODEL.n_exc + MODEL.n_inh)
        * uint_bytes,
    }
    result = {
        "accepted_samples": accepted - start_sample,
        "attempts": attempts,
        "build_wall_seconds": network.build_wall_seconds,
        "simulated_ticks": ticks,
        "simulation_wall_seconds": wall,
        "seconds_per_timestep_cycle": wall / ticks,
        "accepted_samples_per_second": (accepted - start_sample) / wall,
        "biological_realtime_factor": ticks * MODEL.dt_ms / 1000.0 / wall,
        "whole_program_wall_seconds": time.perf_counter() - program_started,
        "variant": variant.name,
        "parallelism": args.parallelism,
        "num_threads_per_spike": args.num_threads_per_spike,
        "structural_synapses": int(mask.sum()),
        "genn_timing_enabled": args.genn_timing,
        "profile_accounting_enabled": args.profile_accounting,
        "kernel_timing": kernel_timing,
        "component_wall_seconds": component_wall if args.profile_accounting else None,
        "host_device_transfer_bytes": transfer_bytes,
        "host_device_transfer_bytes_total": int(sum(transfer_bytes.values())),
        **_counter_metrics(counters, ticks, wall),
    }
    with (args.output / "results" / "performance.json").open("x", encoding="ascii") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("PERFORMANCE " + " ".join(f"{k}={v}" for k, v in result.items()), flush=True)
    print(f"RUN_COMPLETE backend=genn-{args.backend} accepted={accepted}", flush=True)
    return 0


def evaluate(args: argparse.Namespace) -> int:
    program_started = time.perf_counter()
    checkpoint = load_checkpoint(args.checkpoint)
    variant = get_variant(args.variant)
    checkpoint_variant = checkpoint.manifest.get("variant", {}).get("name")
    if checkpoint_variant is not None and checkpoint_variant != variant.name:
        raise ValueError(
            f"checkpoint variant {checkpoint_variant!r} does not match {variant.name!r}"
        )
    mask = validate_checkpoint_topology(checkpoint.weights, variant)
    api = _import_genn()
    manifest = {
        "backend": f"genn-{args.backend}",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "command": sys.argv,
        "genn_version": api["module"].__version__,
        "cuda_path": os.environ.get("CUDA_PATH"),
        "cuda_host_cxx": os.environ.get("CUDAHOSTCXX"),
        "nvcc_prepend_flags": os.environ.get("NVCC_PREPEND_FLAGS"),
        "mode": "evaluate",
        "model": MODEL.as_dict(),
        "probe_samples": args.samples,
        "rng_seed": args.seed,
        "variant": variant.as_dict(),
        "structural_synapses": int(mask.sum()),
        "feedforward_storage": "SPARSE",
        "precision": args.precision,
        "parallelism": args.parallelism,
        "num_threads_per_spike": args.num_threads_per_spike,
        "genn_timing_enabled": args.genn_timing,
        "profile_accounting_enabled": args.profile_accounting,
        "reuse_build": str(args.reuse_build.resolve()) if args.reuse_build else None,
    }
    _create_run_directory(args.output, manifest)
    data = load_mnist(args.data_path, "test")
    network = GeNNNetwork(
        weights=checkpoint.weights,
        theta_mv=checkpoint.theta_mv,
        plasticity=False,
        inhibition=MODEL.inference_inhibition,
        seed=args.seed,
        backend=args.backend,
        build_path=args.output / "build" if args.reuse_build is None else args.reuse_build,
        variant=variant,
        structural_mask=mask,
        precision=args.precision,
        parallelism=args.parallelism,
        num_threads_per_spike=args.num_threads_per_spike,
        timing_enabled=args.genn_timing,
        reuse_build=args.reuse_build,
    )
    activity = np.zeros((args.samples, MODEL.n_exc), dtype=np.uint16)
    attempts = 0
    wall = 0.0
    try:
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
        counters = network.event_counters()
    finally:
        network.close()
    score = simple_demo_accuracy(activity, data.labels[: args.samples])
    ticks = attempts * MODEL.attempt_ticks
    result = {
        "accuracy_percent": score["accuracy_percent"],
        "assigned_neurons": score["assigned_neurons"],
        "assignment_counts": score["assignment_counts"],
        "backend": f"genn-{args.backend}",
        "build_wall_seconds": network.build_wall_seconds,
        "checkpoint_samples": checkpoint.accepted_samples,
        "probe_samples": args.samples,
        "protocol": "same test activity used for assignment and scoring",
        "attempts": attempts,
        "retries": attempts - args.samples,
        "simulation_wall_seconds": wall,
        "simulated_ticks": ticks,
        "seconds_per_timestep_cycle": wall / ticks,
        "whole_program_wall_seconds": time.perf_counter() - program_started,
        "variant": variant.name,
        "precision": args.precision,
        "parallelism": args.parallelism,
        "num_threads_per_spike": args.num_threads_per_spike,
        "structural_synapses": int(mask.sum()),
        **_counter_metrics(counters, ticks, wall),
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
    print("CHECKPOINT_ACCURACY " + " ".join(f"{k}={v}" for k, v in result.items()), flush=True)
    print(f"RUN_COMPLETE backend=genn-{args.backend} accepted={args.samples}", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    repository = root.parent
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--backend", choices=("single_threaded_cpu", "cuda"), default="single_threaded_cpu")
        command.add_argument(
            "--variant", choices=tuple(VARIANTS), default="triplet-dense"
        )
        command.add_argument("--precision", choices=("double", "float"), default="double")
        command.add_argument(
            "--parallelism",
            choices=("postsynaptic", "presynaptic"),
            default="postsynaptic",
        )
        command.add_argument("--num-threads-per-spike", type=int, default=1)
        command.add_argument("--data-path", type=Path, default=repository / "data" / "mnist")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--runaway-spikes", type=int, default=5000)
        command.add_argument("--max-intensity", type=float, default=20.0)
        command.add_argument(
            "--genn-timing",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="enable GeNN's per-stage CUDA event timers",
        )
        command.add_argument(
            "--profile-accounting",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="time host phases and report host/device transfer accounting",
        )
        command.add_argument(
            "--reuse-build",
            type=Path,
            help="load an existing immutable GeNN build directory without compiling",
        )

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
        default=None,
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
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.num_threads_per_spike <= 0:
        raise SystemExit("--num-threads-per-spike must be positive")
    if args.parallelism == "postsynaptic" and args.num_threads_per_spike != 1:
        raise SystemExit(
            "--num-threads-per-spike only applies to presynaptic parallelism"
        )
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
