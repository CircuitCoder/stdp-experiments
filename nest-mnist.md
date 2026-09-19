# NEST MNIST latency benchmarks

Measurements collected on **2026-09-10**; consolidated on **2026-09-13**.

NEST completed the full 21-case MNIST matrix using one network with 16 native
threads. Median latency was **39.498–99.334 µs per timestep**, taking
**3.16–5.80 times the elapsed time of single-core GeNN** across the matched
100-image workloads. A separate, shorter dense-case comparison found that
32 NEST threads provided **1.18–1.40× median speedup over one NEST thread**.
Other host processes were running during these measurements.

These are elapsed-time measurements of one evolving network. Synchronization
is included; elapsed time is divided only by executed timesteps. Independent
replicas are not used. Power was not measured, so these results do not establish
a performance-per-watt ranking.

## Implementation and measurement protocol

- **Host:** Ryzen 9 7950X, reporting 16 physical cores and 32 logical CPUs to
  the WSL2/NixOS container. CPU affinity does not reserve host resources.
- **Simulator:** NEST `3.9.0-post0.dev14`, revision
  `182eba446a8b89108f21cd2ad54aa4c667afd86a`, with the custom
  [MNIST extension](cpu-benchmark/mnist_module/module.cpp) and
  [benchmark runner](cpu-benchmark/run_nest_mnist.py).
- **Build:** the extension uses GCC 14.3, `-O3 -march=native`, without
  fast-math. The reused NEST engine uses `-O3`; its earlier Nix build stripped
  `-march=native`. Neuron, trace and plastic state are FP32. Normalization
  sums/scales and NEST recurrent event buffers use FP64.
- **Network:** 784 inputs, 400 excitatory and 400 inhibitory neurons, using
  the GeNN benchmark's midpoint conductance integration and next-tick
  delivery. Native NEST nodes, spike routing and OpenMP scheduling evolve
  the network in one MPI process.
- **Learning:** one-, two- and three-trace rules. Plastic weights belong to
  their target neuron, allowing eager postsynaptic updates, including for
  silent inputs. Shared input traces use alternating buffers separated by
  NEST's per-tick barrier. One input spike is broadcast to its complete fan-out.
- **Dataset and initialization:** real MNIST training data in `data/mnist`,
  seed `20260724`, with the frozen per-case weights/theta checkpoints and
  masks from `copilot/tmp/baseline_inputs_20260907_v2`. Each repetition starts
  from its 10,000-image checkpoint, resets other runtime state, and presents
  images sequentially beginning at index 10000. This is a branched training
  continuation benchmark; no classification accuracy was evaluated.
- **Attempt:** 700 stimulus ticks plus 300 rest ticks, with `dt = 0.5 ms`.
  Columns are normalized to 78 before each attempt. Fewer than five E spikes
  triggers an intensity retry. The stimulus counter includes preceding rest
  spikes, as in the GeNN baseline. Timing includes retries and uses the actual
  number of executed ticks.
- **Timer:** includes simulation, synchronization, normalization, host
  control and final spike-count reads. Excludes construction, setup and
  result serialization. Diagnostic runs precede the timing repetitions.
- **Thread settings:** `OMP_WAIT_POLICY=ACTIVE`, `GOMP_SPINCOUNT=300000`,
  `OMP_PROC_BIND=spread`, `OMP_PLACES=cores`. Thread counts and repetitions
  run sequentially. NEST's `local_num_threads`, selected through `--threads`,
  controls simulation concurrency.

These results belong to the saved run snapshots, rather than an arbitrary
later worktree. The NEST runner and compiled extension were identical between
the full matrix and the later 1/32-thread test. Each run manifest preserves its
source hashes, command, environment, input identity and module hash. The older
`reimpl/run_nest.py` training port is a separate implementation.

## Full matrix: 16 threads

Each configuration used **100 accepted images and five timing repetitions**.
Values are median elapsed **µs/timestep**; lower is faster.

| Connectivity | 1 trace | 2 traces | 3 traces |
|---|---:|---:|---:|
| Dense | 79.303 | 99.334 | 97.152 |
| Random 50% | 59.156 | 59.339 | 57.953 |
| Random 25% | 47.424 | 45.796 | 48.042 |
| Random 12.5% | 39.498 | 42.107 | 40.549 |
| Fixed fan-out 50% | 59.887 | 60.310 | 57.364 |
| Fixed fan-out 25% | 47.937 | 45.941 | 48.043 |
| Fixed fan-out 12.5% | 39.968 | 42.047 | 42.105 |

Random connectivity uses Bernoulli masks. Fixed degree means **fan-out**:
each input connects to exactly 200, 100 or 50 E targets at the respective
densities. All masks are reused from the frozen checkpoints.

For context, the dense cases compare as follows, also in median µs/timestep:

| Rule | NEST, 16 threads | GeNN CPU, 1 thread | GeNN RTX 3090 | GeNN A100 |
|---|---:|---:|---:|---:|
| 1 trace | 79.303 | 13.678 | 26.473 | 18.848 |
| 2 traces | 99.334 | 29.496 | 27.300 | 19.598 |
| 3 traces | 97.152 | 19.914 | 32.674 | 21.770 |

The comparison uses the same frozen model definitions, masks, checkpoints and
100-image protocol. Native NEST random streams and accumulation order differ
from GeNN, so the trajectories are not identical. Across the full matrix,
NEST E spikes per timestep ranged from 0.955 to 1.072 times the corresponding
GeNN rate. GeNN was the faster CPU implementation measured here; this is not
a claim about every possible NEST implementation or an isolated host.

The [full timing CSV](cpu-benchmark/results-nest-mnist-20260910/timings.csv)
contains all 21 GeNN comparisons, all four recorded GPUs, and NEST ranges.

## Quick dense comparison: 1 versus 32 threads

Each case used **20 accepted images and three timing repetitions**. One thread
was tested first, followed by 32 threads. These have a different sample budget
from the full 16-thread matrix and should not be combined with it to calculate
a 1/16/32-thread scaling curve.

| Rule | 1 thread, median µs/timestep | 32 threads, median µs/timestep | Speedup |
|---|---:|---:|---:|
| 1 trace | 123.013 | 104.044 | 1.18× |
| 2 traces | 137.968 | 98.430 | 1.40× |
| 3 traces | 126.239 | 101.173 | 1.25× |

Speedup is the one-thread median divided by the 32-thread median. Observed
ranges across the three repetitions were:

| Rule | 1 thread, µs/timestep | 32 threads, µs/timestep |
|---|---:|---:|
| 1 trace | 118.880–126.838 | 98.766–105.615 |
| 2 traces | 137.257–139.687 | 95.119–99.435 |
| 3 traces | 126.162–126.591 | 100.751–1439.506 |

The first 32-thread three-trace repetition took **1439.506 µs/timestep**;
the other two took 100.751 and 101.173. The slow repetition is retained. Its
cause was not isolated on this shared host; the median hides this tail latency.

The observed native thread counts were exactly one and 32. The single worker
was affined to logical CPUs 0–1, representing one physical core. The 32 workers
were affined two per physical core, spanning logical CPUs 0–31.

NEST random streams depend on thread count. The one-thread cases each took
20 attempts; the 32-thread one-, two- and three-trace cases took 20, 21 and
21 attempts. Actual tick counts account for the retries, but differing spike
trains still make this an approximate concurrency comparison.

## Earlier thread-selection pilot

Before the full matrix, a dense pilot tested six thread counts with **five
accepted images and two timing repetitions** per case. Median µs/timestep:

| Threads | 1 trace | 2 traces | 3 traces |
|---|---:|---:|---:|
| 1 | 131.400 | 149.198 | 136.633 |
| 2 | 97.347 | 105.425 | 101.219 |
| 4 | 87.186 | 90.820 | 89.539 |
| 8 | 83.042 | 85.265 | 85.317 |
| 16 | 87.833 | 89.650 | 88.847 |
| 32 | 105.720 | 114.746 | 104.549 |

Eight threads were modestly faster than sixteen in this short pilot, while
32 were slower. Sixteen were selected for the full matrix to exercise all
physical cores as requested. The pilot also uses thread-dependent RNG and
does not establish that maximum concurrency minimizes latency or energy.

## Weight bounds and validation

Normalization can exceed the STDP learning cap when too few inputs must sum
to 78, or when weights are concentrated. NEST records normalization and final
cap overshoots without aborting, while preserving the configured STDP clipping
rule. The measured GeNN timing path likewise disabled normalization bound
validation; its diagnostic path could still abort on its tolerance. No cap
overshoot occurred in the 21 NEST full-run diagnostics: the largest observed
normalized weight divided by its cap was 1.0.

Finite-state, nonnegative-weight, structural-mask, normalization-sum, runaway
firing and retry guards remain. At the time of the experiment, **61 relevant
tests passed**, including 13 deterministic MNIST tests covering all three
rules, event ordering, shared traces, recurrent delivery, refractory behavior,
normalization boundaries and an intentionally impossible normalization cap.

All **105 full-matrix timing repetitions** and all **18 quick-test timing
repetitions** exactly matched their corresponding NEST diagnostic final FP32
weights and thresholds, population/event counts and executed synaptic visits.
These are checks within the same NEST configuration, not bitwise agreement
between simulators or evidence of long-training accuracy alignment.

## Reproduction and artifacts

Run from the repository root with the preserved NEST installation and module.
Use fresh output directories. The following commands reproduce the measured
configuration; the source snapshots in the manifests identify the exact code
used for the original measurements.

```sh
nix-shell cpu-benchmark/shell.nix
export NIX_ENFORCE_NO_NATIVE=0
export OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=300000
export OMP_PROC_BIND=spread OMP_PLACES=cores

python cpu-benchmark/run_nest_mnist.py \
  --nest-prefix copilot/tmp/cpu_nest_build_20260910_c/install \
  --module copilot/tmp/cpu_nest_mnist_module_20260910_b/cpumnistmodule.so \
  --inputs copilot/tmp/baseline_inputs_20260907_v2 \
  --data-path data/mnist --seed 20260724 \
  --output copilot/tmp/nest_mnist_full16_NEW \
  --threads 16 --samples 100 --repetitions 5

for threads in 1 32; do
  python cpu-benchmark/run_nest_mnist.py \
    --nest-prefix copilot/tmp/cpu_nest_build_20260910_c/install \
    --module copilot/tmp/cpu_nest_mnist_module_20260910_b/cpumnistmodule.so \
    --inputs copilot/tmp/baseline_inputs_20260907_v2 \
    --data-path data/mnist --seed 20260724 \
    --output "copilot/tmp/nest_mnist_t${threads}_NEW" \
    --threads "$threads" --samples 20 --repetitions 3 \
    --cases mnist_1trace_dense mnist_2trace_dense mnist_3trace_dense
done
```

- [Full matrix report](cpu-benchmark/results-nest-mnist-20260910/report.md),
  [diagnostics](cpu-benchmark/results-nest-mnist-20260910/diagnostics.csv) and
  [thread pilot](cpu-benchmark/results-nest-mnist-20260910/thread-pilot.csv).
- [Quick 1/32-thread report](cpu-benchmark/results-nest-mnist-threads-20260910/report.md),
  [timings](cpu-benchmark/results-nest-mnist-threads-20260910/timings.csv) and
  [provenance](cpu-benchmark/results-nest-mnist-threads-20260910/summary.json).
- [Full-run manifest](copilot/tmp/cpu_nest_mnist_full16_20260910_a/manifest.json),
  [validation](copilot/tmp/cpu_nest_mnist_full16_20260910_a/validation/validation.json)
  and [log](copilot/tmp/cpu_nest_mnist_full16_20260910_a.log).
- [Quick-test launch command](copilot/tmp/cpu_nest_mnist_t1_t32_20260910_a/launch.json)
  and [log](copilot/tmp/cpu_nest_mnist_t1_t32_20260910_a/run.log).
- [Build and runner instructions](cpu-benchmark/README.md) and
  [CPU/energy assessment](cpu-benchmark/assessment.md).
