# GeNN CUDA A100 sweep

`run.py` runs the five GeNN workloads selected during profiling with the
diagnostic paths removed. It defaults to FP32 and prints one result line per
case. GeNN event timing, spike recording, weight sampling, runtime-state
validation, connectivity accounting, periodic statistics, checkpoints, and
result serialization are disabled.

The shared Brunel/Morrison adapter now uses integer STDP timestamps; see the
[timing investigation](../workloads/MORRISON-INTEGER-TICKS-20260920.md).
The historical measurements below predate this change and retain their recorded
source hashes. Weights, traces, and neuron arithmetic still use the selected
FP32/FP64 scalar precision.

MNIST column normalization and retry decisions are retained because they are
part of the training workload. Excitatory spike counters are read after each
stimulus because the count controls retry behavior. No profiler should be
attached when collecting these wall-time results.

## Workloads

The default sweep runs these cases in order:

| Case | Starting state | GeNN scheduling |
|---|---|---|
| `mnist_triplet_dense` | triplet dense 10k checkpoint, seed 0 | PostSpan, x1 |
| `mnist_one_trace_dense` | one-trace dense 10k checkpoint, seed 0 | PostSpan, x1 |
| `mnist_one_trace_sparse_0125` | one-trace 12.5% sparse 10k checkpoint, seed 0 | PreSpan, x32 |
| `brunel_additive` | seed 20260724, 5% indegree, zero delay | E-E PostSpan, x1 |
| `brunel_morrison` | seed 20260724, 5% indegree, zero delay | E-E PostSpan, x1 |

MNIST trains 100 accepted samples, including any retry attempts, with 700
stimulus steps and 300 rest steps per attempt. Brunel uses arrival-timed STDP,
100 ms of untimed presimulation, and a 1,000 ms measured interval at 0.1 ms per
step. The Brunel graph retains 9,000 excitatory and 2,250 inhibitory neurons,
but uses 450 excitatory and 112 inhibitory inputs per neuron. It therefore has
6,322,500 recurrent synapses, including 4,050,000 plastic E-to-E synapses.

The recurrent transport delay is zero GeNN delay steps. Since synapses update
before neurons within each global step, a source spike is consumed by the
synapse kernel on the following 0.1 ms step. All recurrent delivered currents
are multiplied by `1 / sqrt(0.05) = 4.47213595499958`. The learned E-to-E
variable, additive bound, and both STDP update equations remain in their
original units. External input rate is scaled by 0.47 for additive and 0.32 for
Morrison. No periodic synaptic normalization is used.

These are the sweep defaults. The topology and compensation can be overridden
with `--brunel-indegree-scale`, `--brunel-delay-ms`,
`--brunel-recurrent-delivery-scale`,
`--brunel-additive-external-rate-scale`, and
`--brunel-morrison-external-rate-scale`.

## Sparse zero-delay recovery

The configuration was selected on 2026-08-28 UTC using FP32 GeNN 5.4.0 on the
RTX 3090 described below. Each diagnostic used seed and state seed `20260724`,
100 ms presimulation, arrival-timed STDP, 100 ms diagnostic chunks, and a 100 Hz
excitatory-rate cutoff. Disposable manifests and results are under
`copilot/tmp/brunel_sparse005_zero_*`.

The analytic variance-preserving external rate scale, `sqrt(0.05)`, made the
additive network silent. Holding recurrent delivery at `sqrt(20)` and tuning
only the rule-specific external rate recovered the active branch:

| Rule / external scale | Measured interval | E rate | Outcome |
|---|---:|---:|---|
| Additive / 0.223607 | 500 ms | 0 Hz | silent |
| Additive / 0.45 | 500 ms | 0.006 Hz | effectively silent |
| Additive / 0.47 | 1,000 ms | 4.381 Hz | selected |
| Additive / 0.475 | 1,000 ms | 5.472 Hz | active control |
| Additive / 0.48 | 1,000 ms | 5.443 Hz | active control |
| Additive / 0.50 | 300 ms | 8.900 Hz | high-rate control |
| Morrison / 0.25 | 500 ms | 1.968 Hz | low-rate control |
| Morrison / 0.30 | 1,000 ms | 4.969 Hz | low-rate control |
| Morrison / 0.32 | 1,000 ms | 5.967 Hz | selected |

The compatible full-density controls use the same neuron counts, seed,
presimulation and measurement lengths, and arrival-timed STDP. Their total
spikes come from `test-run-20260811-retry2.txt`; their E counts come from
`locality/runs/selected_1s_20260826_a`. Selected sparse counts come from the
full diagnostic artifacts.

| Rule / graph | E spikes | I spikes | Total spikes | Final sampled E-E weight | Boundary mass / modes |
|---|---:|---:|---:|---:|---:|
| Additive / full indegree | 41,417 | 12,405 | 53,822 | not sampled by sweep | not sampled by sweep |
| Additive / 5% zero delay | 39,755 | 10,141 | 49,896 | 45.474 +/- 1.013 pA | 0% / 1 |
| Morrison / full indegree | 54,856 | 14,587 | 69,443 | not sampled by sweep | not sampled by sweep |
| Morrison / 5% zero delay | 53,597 | 13,492 | 67,089 | 45.452 +/- 0.449 pA | 0% / 1 |

Thus total firing is 7.3% below the additive control and 3.4% below the
Morrison control. Both rules complete the requested interval without a cutoff,
weight boundary mass, or loss of the interior mode. This recovers firing level
and short-run weight stability, not the full temporal process. The sparse
diagnostics' first-1,000-E-neuron 3 ms population Fano factors are 122.51 for
additive and 5.09 for Morrison, so correlation structure remains sensitive to
the topology and should not be called numerically matched.

The subsequent all-neuron, 10-second capture in `locality/RESULTS.md` confirms
that distinction at native-tick and individual-neuron granularity. It is an
instrumented locality experiment and is not a replacement for the timing rows
below.

## RTX 3090 benchmark results

The full five-case FP32 sweep was run three times on 2026-08-28 UTC on an
NVIDIA GeForce RTX 3090 (24 GiB, driver 596.49), using CUDA 13.0.88 and the
current worktree based on Git `0a7299e73dcb1dea86e82c24236a8170680cbe65`.
Run 1 compiled fresh; runs 2 and 3 reused its exact generated models. Timed
regions and denominators follow the measurement contract below.

The benchmark source SHA-256 values were `35d79fbec6fad8660d9122d5e1c0f17121e849019d90438c5a101754e5b293d6`
for `genn-sweep/run.py`, `478e8b74ec3edf8bc7f28e4ceb800a010156005310f291430f0e51d8cd370bf1`
for `brunel/ports/common.py`, and
`83bf75b022edfd17c06497073e2623360390f24b4835f44f31ca3fcac6fc0ad0`
for `brunel/ports/genn_port.py`.

| Case | Spike count | Median us/step | Three-run range us/step |
|---|---:|---:|---:|
| `mnist_triplet_dense` | 246,685 | 34.884 | 34.814-36.118 |
| `mnist_one_trace_dense` | 244,155 | 28.875 | 28.546-29.736 |
| `mnist_one_trace_sparse_0125` | 244,318 | 26.344 | 25.786-26.794 |
| `brunel_additive` | 49,896 | 42.204 | 42.119-42.634 |
| `brunel_morrison` | 67,089 | 42.193 | 41.858-42.311 |

Against the retained full-density RTX 3090 sweep, additive improves from
95.157 to 42.204 us/step (2.25x, 55.6% less time) and Morrison improves from
121.039 to 42.193 us/step (2.87x, 65.1% less time). These speedups compare
different delay and topology configurations and are workload results, not
isolated synapse-kernel speedups.

The retained work directories are:

```text
copilot/tmp/genn_sweep_sparse005_zero_fp32_20260828_run1
copilot/tmp/genn_sweep_sparse005_zero_fp32_20260828_run2
copilot/tmp/genn_sweep_sparse005_zero_fp32_20260828_run3
```

Exact stdout for all three runs is retained in
`genn-sweep/sparse005-zero-results-20260828.txt`.

## Checkpoints

The three immutable MNIST starting states are committed under
`genn-sweep/checkpoints/` and used by default. Each is the 10,000-accepted-sample
checkpoint from a retained 30,000-sample GeNN CUDA training run on MNIST with
seed 0.

| Workload | Committed checkpoint | Original retained run checkpoint | SHA-256 |
|---|---|---|---|
| Triplet dense | `mnist_triplet_dense_010000.npz` | `reimpl/runs/genn_cuda_mnist_30k_20260725_a/checkpoints/checkpoint_010000.npz` | `e4cef93ef2ad8c8e93b7d3d1b93b28276d62616f1e3b59b4dbc0db99149becce` |
| One-trace dense | `mnist_one_trace_dense_010000.npz` | `reimpl/runs/genn_cuda_onetrace_dense_train30000_post_20260805_a/checkpoints/checkpoint_010000.npz` | `eab026d59dad20f1dd979f800e6a37e3f8a3e2b0386febb3f9b1fe672e44d289` |
| One-trace sparse 12.5% | `mnist_one_trace_sparse_0125_010000.npz` | `reimpl/runs/genn_cuda_onetrace_sparse0125_train30000_pre32_20260805_a/checkpoints/checkpoint_010000.npz` | `c32bf0269d8dd33879a7ecfadc096e078cb4e6967d0d58215a1fe061253cc82d` |

The original experiment directories remain on this machine for result
reproduction. Override the committed defaults with `--triplet-checkpoint`,
`--dense-checkpoint`, or `--sparse-checkpoint` when testing another state.
`--data-path` similarly overrides the MNIST directory.

## Environment

The tested source uses GeNN/PyGeNN 5.4.0 from `3rdparty/genn`. On a conventional
Linux installation, provide:

- an NVIDIA driver that supports the installed CUDA toolkit;
- the CUDA toolkit, including `nvcc`;
- a CUDA-compatible host C++ compiler;
- Python 3.10 or newer; and
- NumPy and SciPy.

One setup sequence is:

```sh
python3 -m venv .venv-genn
. .venv-genn/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pybind11 psutil pkgconfig 'setuptools>=61'

export CUDA_PATH=/usr/local/cuda
export CUDAHOSTCXX=/usr/bin/g++
python -m pip install --editable ./3rdparty/genn
```

Adjust `CUDA_PATH` and `CUDAHOSTCXX` to the installed toolkit and a compiler
version supported by that toolkit. Verify the device before building:

```sh
nvidia-smi
"${CUDA_PATH}/bin/nvcc" --version
python -c 'import pygenn; print(pygenn.__version__)'
```

No WSL-specific library paths or Nix compiler flags should be copied from the
original profiling machine.

## Run

From the repository root, run the maximum-throughput FP32 sweep:

```sh
python genn-sweep/run.py --work-dir genn-sweep/a100-fp32
```

The work directory must not already exist. It contains generated CUDA builds
and compiler logs, but no measurements or model-state recordings. Normal
stdout contains only the five result lines, for example:

```text
case=mnist_triplet_dense precision=float spike_count=12345 wall_seconds=3.123456789 seconds_per_step=3.061232146078e-05
```

Run both precisions when an FP64 comparison is useful on the A100:

```sh
python genn-sweep/run.py \
  --precision both \
  --work-dir genn-sweep/a100-fp32-fp64
```

Select a subset or change the workload lengths as follows:

```sh
python genn-sweep/run.py \
  --cases mnist_one_trace_sparse_0125 brunel_morrison \
  --mnist-samples 100 \
  --brunel-presim-ms 100 \
  --brunel-sim-ms 1000 \
  --work-dir genn-sweep/a100-subset
```

After a successful build, a repetition can reuse exactly matching generated
models while writing new logs to a fresh work directory:

```sh
python genn-sweep/run.py \
  --work-dir genn-sweep/a100-fp32-repeat-2 \
  --reuse-build-root genn-sweep/a100-fp32/builds
```

Only reuse a build root created by this script with the same source, case,
precision, all workload arguments (including Brunel topology, delay, delivery
and external-rate scales, seed, and scheduling), GeNN version, CUDA toolkit,
compiler, and GPU architecture. The script names build directories
`<case>_<precision>` and GeNN's `never_rebuild` mode does not validate all of
those conditions.

## Measurement contract

`wall_seconds` is workload wall time, not process startup time. Model creation,
CUDA compilation, device allocation, checkpoint and dataset loading, Brunel
presimulation, and model unloading are outside it. The final spike-counter read
and CUDA synchronization are inside it.

For MNIST, the timed region includes normalization, host/device transfers,
retry decisions, and every stimulus/rest step for accepted and retried
attempts. Its `seconds_per_step` denominator is therefore:

```text
number of attempts * 1,000 steps
```

For Brunel, the timed region is the single uninterrupted 1,000 ms measurement
after presimulation. Its default denominator is 10,000 steps.

`spike_count` is a basic trajectory check:

- MNIST: Input + Excitatory + Inhibitory spikes across all timed attempts,
  including stimulus and rest periods.
- Brunel: Excitatory + Inhibitory spikes during the measured interval only;
  presimulation spikes are subtracted.

Compare spike counts only between runs with the same source, precision, seed,
checkpoint, dataset order, and workload lengths. FP32 and FP64 are not expected
to produce identical trajectories; the earlier RTX 3090 controls found roughly
5% precision-dependent changes in Brunel firing rates. A spike-count mismatch
across GPU architectures is evidence to investigate, not by itself proof of an
incorrect run.
