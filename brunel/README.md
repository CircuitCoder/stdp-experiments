# Brunel STDP experiments

The current validated default is the FP32 **GeNN-only** configuration in
`genn-sweep/README.md`: 9,000 E and 2,250 I neurons, 5% fixed indegree (450 E
and 112 I inputs per neuron), zero GeNN delay steps, arrival-timed STDP,
recurrent delivery scale `sqrt(20)`, and external-rate scale 0.47 for additive
or 0.32 for Morrison. `run_genn.py` now uses these defaults. The Brian2,
Brian2CUDA, NEST, and NEST-GPU ports have not been validated with this
configuration and retain historical behavior.

The shared GeNN code now uses integer tick timestamps for STDP, with exact tie
comparisons and FP32 elapsed-time trace arithmetic for FP32 workloads. See the
[integer timing investigation](../workloads/MORRISON-INTEGER-TICKS-20260920.md).
Historical benchmark timings predate this timestamp change.

The reports below primarily describe the older full-indegree, 1.5 ms-delay
family. Their results remain valid for their recorded manifests but are not
direct performance or dynamics comparisons with the current GeNN default.

`STDP_REGIMES.md` documents why additive hard-bound STDP is expected to
produce a bimodal weight distribution while the Morrison power-law rule used
by NEST's HPC benchmark is expected to produce a unimodal distribution.
`RESULTS.md` records the paired scale-1 runs, timing per integration step, and
the limits of the behavior reproduced at desktop scale.
`STABILITY_TUNING.md` records the static controls and the joint
inhibition/drive/depression sweep used to prevent short-run Morrison runaway
at scale 1.

Cross-framework implementations are in `ports/`, with launchers
`run_brian2.py`, `run_brian2cuda.py`, `run_brian2cuda_codegen.py`,
`run_genn.py`, and `run_nestgpu.py`. `RESULTS.md` records the CPU and CUDA
runtime comparisons and backend-specific semantic limitations.

`nest_brunel_stdp.py` runs all three modes through one network implementation.
The only intended difference between paired plastic cases is the E-to-E
synapse model and its rule-specific parameters:

```sh
bash brunel/run_with_nest_env.sh \
  --rule additive --output brunel/runs/additive_example

bash brunel/run_with_nest_env.sh \
  --rule morrison --output brunel/runs/morrison_example
```

For stability experiments, `--rule static` freezes E-to-E weights at the
benchmark initial value. `--inhibitory-weight-ratio` controls the magnitude of
the inhibitory weight relative to the static excitatory weight (the stock
value is 5), while `--external-drive-eta` controls external drive relative to
threshold (the stock value is 1.685).

Long stability sweeps can set `--abort-rate-hz`. The runner then stops after a
reporting chunk reaches that rate, but still writes final samples, actual step
counts, timings, and an explicit termination reason.

Output directories are exclusive and cannot be reused. Each run writes its
full command and environment to `manifest.json`, periodic statistics and final
measurements to `results.json`, and deterministic initial/final weight samples
to `weight_samples.npz`.

The runner reports three time-per-step quantities:

- `simulation_seconds_per_step` covers the measured post-presimulation NEST
  calls only;
- `combined_simulate_call_seconds_per_step` includes both presimulation and
  measured NEST calls; and
- `dynamic_phase_seconds_per_step` also includes periodic Python-side
  statistics collected between chunks.

The documented NEST desktop workload is selected by the defaults. Smaller
diagnostic workloads must record both `--network-scale` and
`--indegree-scale`, because reducing either changes the network's correlation
structure and therefore its plastic equilibrium.

The Brian2 and GeNN launchers use the same scale options and emit the same
manifest/result/weight-sample layout:

```sh
bash brunel/run_with_reimpl_env.sh brunel/run_brian2.py \
  --rule additive --output brunel/runs/brian2_additive_example

bash brunel/run_with_reimpl_env.sh brunel/run_genn.py \
  --rule morrison --output brunel/runs/genn_morrison_example
```

Brian2 and GeNN default to `--stdp-timing nest_dendritic`. This delays the
postsynaptic learning path by 3 ms while retaining the 1.5 ms recurrent
transmission delay, reproducing the pair timing used by NEST's dendritic-delay
convention. Use `--stdp-timing arrival` for the original port behavior. These
modes are distinct model protocols and must not be mixed in a comparison.

They also default to `--stdp-tie-order nest_causal_boundary`. At NEST's causal
window boundary, the upper-bound post spike contributes to potentiation, while
the strictly-before post-trace lookup excludes it from depression. This mode
therefore applies both weight changes from the old traces before retaining both
trace increments. The `framework_pre_first`, `nest_post_first`, and
`nest_exclude_zero` modes are retained as explicit diagnostic protocols.

GeNN defaults to `single_threaded_cpu`; pass `--backend cuda` with a
CUDA-capable PyGeNN build. `run_brian2cuda.py` builds and runs the same Brian2
model through CUDA standalone. NEST-GPU requires a separately built vendored
library. Its nearest-pair plasticity and unavoidable-autapse differences are
documented in `RESULTS.md` and in every generated manifest, so its output is an
approximation rather than an equivalent NEST rule.

Brian2CUDA defaults to `--compile-jobs 1`. CUDA standalone otherwise asks GNU
make for one compiler process per visible CPU, which creates a large transient
host-memory spike even for a small model. Increase this only after measuring
available host memory. The scale-1 E-to-E graph has 81 million plastic
synapses; its generated host arrays and per-synapse trace state can exceed this
container's safe headroom independently of GPU memory.
