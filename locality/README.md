# Synapse update locality

The locality runner takes its Brunel model from the current FP32 GeNN default:
5% fixed indegree, zero delay, recurrent delivery scale `sqrt(20)`, and
rule-specific external-rate scaling. `RESULTS.md` reports the current capture
first and retains the older full-indegree, 1.5 ms-delay measurements only as
historical controls.

This directory captures temporal and graph locality for the five workloads in
`genn-sweep`. The capture is deliberately separate from the timing sweep: spike
recording, weight snapshots, connectivity reads, graph reductions, and plots
are diagnostic overhead and must not be included in performance measurements.

Two event definitions are reported. A **synapse traversal** executes a pre- or
postsynaptic handler. A **weight-update attempt** executes code assigning the
plastic weight, even when a zero trace or bound leaves the value unchanged.
The MNIST one-trace rule therefore has pre-driven transmission traversals but
only post-driven weight updates. The triplet MNIST and both Brunel rules update
weights on both event paths.

The default command uses real MNIST, the three committed 10,000-sample
checkpoints, 100 accepted training samples, and the current GeNN Brunel cases
with seed 20260724, 100 ms presimulation, and 1,000 ms measurement.

```sh
nix-shell \
  -p python313Packages.numpy python313Packages.scipy \
     python313Packages.matplotlib python313Packages.psutil \
  --run 'PYTHONPATH="$PWD/3rdparty/genn:$PYTHONPATH" \
    CUDA_PATH=/tmp/stdp-locality-cuda13-patched \
    CUDAHOSTCXX=/nix/store/5p9jf97rva6v7w56a372as90sgvw82jl-gcc-wrapper-13.4.0/bin/g++ \
    LD_LIBRARY_PATH=/nix/store/22qr7rzm4ba3z8dpai056lcxq6wh6w4a-stdp-cuda-toolkit-genn-12.8-complete/lib:/tmp/stdp-locality-cuda13-patched/lib:/usr/lib/wsl/lib \
    python locality/run.py --output locality/runs/selected_1s'
```

On this container, `/tmp/stdp-locality-cuda13-patched` is a writable copy of
the retained CUDA 13 toolchain with the `rsqrt` and `rsqrtf` declarations in
`include/crt/math_functions.h` disabled for glibc 2.42. CUDA 12.8's runtime is
also kept on `LD_LIBRARY_PATH` because the retained PyGeNN extension was built
against `libcudart.so.12`.

Each case directory contains `manifest.json`, `summary.json`, compressed raw
arrays in `locality.npz`, and `temporal.png`, `spatial.png`, and
`intervals.png`. Build products stay inside that case directory. Output
directories must be new.

For Brunel, the raw arrays and summary include:

- one firing count for every E and I neuron;
- population inter-arrival histograms and same-neuron interspike histograms;
- exact per-synapse presynaptic intervals, obtained by weighting each source
  neuron's interspike intervals by its realized E-to-E outdegree;
- exact per-synapse postsynaptic intervals, obtained by weighting each target
  neuron's interspike intervals by its fixed indegree;
- population variance, standard deviation, coefficient of variation, skewness,
  excess kurtosis, and quantiles derived directly from every interval
  histogram;
- separate per-synapse pre/post IAT histograms for every complete 1-second
  window, with boundary-crossing intervals excluded;
- intervals between ticks containing any presynaptic or postsynaptic update;
  and
- within-tick gaps between consecutively recorded neuron IDs.

All intervals are quantized to the 0.1 ms Brunel timestep. Neuron-ID gaps
describe adjacency in GeNN array order, not biological distance, GPU memory
transactions, or execution order within a kernel. The runner checks recorded
population counts and every derived pre/post update and interval total against
independent degree-weighted identities before accepting a result. Raw E and I
spike times and IDs from the measured interval are retained in `locality.npz`
for later reanalysis.

`zero_ee_update_tick_fraction` is the fraction of 0.1 ms ticks containing no
logical plastic E-to-E pre or post update. It does not mean the GPU or the whole
network was idle: neuron integration, external input, and other synapse groups
still execute.

An extended Brunel capture uses the same model but measures 10 seconds. It has
a 100 Hz excitatory-rate guard checked every 100 ms.

```sh
python locality/run.py \
  --cases brunel_additive brunel_morrison \
  --brunel-sim-ms 10000 \
  --output locality/runs/brunel_sparse005_zero_iat_10s_20260828_c
```

Generate the normalized report comparison from a completed paired run with:

```sh
python locality/plot_iat_report.py \
  --run locality/runs/brunel_sparse005_zero_iat_10s_20260828_c \
  --output locality/figures/brunel_sparse005_zero_iat_10s.png
```

The completed measurements and their interpretation are in `RESULTS.md`.
Generated run directories are excluded from Git by the repository `.gitignore`;
their manifests, raw arrays, and plots remain available in the workspace.

Run the deterministic analysis tests with:

```sh
pytest -q locality/tests
```
