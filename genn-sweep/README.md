# GeNN CUDA A100 sweep

`run.py` runs the five GeNN workloads selected during profiling with the
diagnostic paths removed. It defaults to FP32 and prints one result line per
case. GeNN event timing, spike recording, weight sampling, runtime-state
validation, connectivity accounting, periodic statistics, checkpoints, and
result serialization are disabled.

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
| `brunel_additive` | seed 20260724, scale 1 | E-E PostSpan, x1 |
| `brunel_morrison` | seed 20260724, scale 1 | E-E PostSpan, x1 |

MNIST trains 100 accepted samples, including any retry attempts, with 700
stimulus steps and 300 rest steps per attempt. Brunel uses arrival-timed STDP,
100 ms of untimed presimulation, and a 1,000 ms measured interval at 0.1 ms per
step. The Brunel graph is the full scale-1 network: 9,000 excitatory neurons,
2,250 inhibitory neurons, and 126,562,500 recurrent synapses.

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
precision, GeNN version, CUDA toolkit, compiler, and GPU architecture. The
script names build directories `<case>_<precision>` and GeNN's
`never_rebuild` mode does not validate all of those conditions.

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
