# GPU Workload Baseline

This document records the fixed simulation workloads used to evaluate ActiveN
against GeNN CUDA on different GPUs. Results apply to the complete recorded
machine and software configuration, including the host CPU and operating
system. They are not measurements of an intrinsic GPU architectural limit.

## Workload Matrix

GPU floating-point state and timestep arithmetic use FP32. Portable checkpoint
arrays and host-side MNIST normalization use FP64 before transfer to the GPU.
A percentage below is
the fraction of possible feedforward connections that are present, not the
fraction removed and not the fraction of learned weights that are nonzero.

The requested suite contains 23 cases:

| Model | Connectivity | Learning rules | Cases |
| --- | --- | --- | ---: |
| MNIST | Dense | One-trace, two-trace, three-trace | 3 |
| MNIST | Independent Bernoulli, 50%, 25%, 12.5% | Each of the three rules | 9 |
| MNIST | Fixed fan-out, 50%, 25%, 12.5% | Each of the three rules | 9 |
| Brunel | Fixed indegree, 5% | Additive, Morrison multiplicative | 2 |

Fixed fan-out means exactly 200, 100 or 50 targets per input neuron. Historical
sparse reference experiments used fixed fan-in, a different topology that is
not substituted here. Masks use NumPy RandomState seed 20260723 and a shared
784-by-400 score matrix. Bernoulli masks threshold the scores; fixed fan-out
masks choose the lowest scores in each row. Masks are nested across rates and
identical across learning rules.

`genn-sweep/baseline.py` implements this complete matrix. The older
`genn-sweep/run.py` retains its five-case interface and supplies the unchanged
Brunel timing path. New sparse cases are performance workloads derived from
learned dense checkpoints, not independently trained sparse networks.

### MNIST Contract

- Real MNIST from `data/mnist`, sequential training-set order.
- 784 Poisson input neurons, 400 excitatory neurons, 400 inhibitory neurons.
- Conductance-based midpoint voltage integration; timestep 0.5 ms.
- Zero GeNN delay steps: synapses consume the preceding neuron update's spikes.
- 700 stimulus steps and 300 rest steps per attempt.
- Feedforward column normalization to 78 before every attempt.
- Retry with increased intensity when the excitatory count is below five.
- Plasticity and adaptive excitatory thresholds enabled during training.
- Training lateral inhibition 25.5; E-to-I weight 10.4.
- Fixed structural masks retain zero-valued present synapses during learning.

The MNIST cases continue each case's immutable
10,000-sample checkpoint for 100 accepted samples. New measurements use GeNN
seed **20260724**. Historical measurements used seed 0, which GeNN interprets
as nondeterministic initialization through `std::random_device`; zero must not
be used for new paired comparisons. A nonzero seed does not guarantee bitwise
agreement across GPU models, compiler versions, or different kernel layouts.
Portable checkpoints contain weights and thresholds; the continuation resets
other runtime state. Record this as a branched continuation, not an exact resume.
See [genn-sweep/README.md](genn-sweep/README.md) for checkpoint hashes and the
existing rule parameters.

Dense one-trace and three-trace starts retain the existing GeNN 10,000-sample
checkpoints. Dense two-trace imports the matched `XeAe10000.npy` and
`theta_A10000.npy` pair from the reference `full_nupost0005_30000` case, with
theta converted from volts to mV. Its portable copy is
`genn-sweep/checkpoints/mnist_two_trace_dense_010000.npz`. Bernoulli 12.5%
one-trace retains its independently trained sparse checkpoint. Every other
sparse start masks its rule's dense checkpoint and renormalizes each column to
78, retaining theta. If simple rescaling exceeds the weight bound, bounded
rescaling preserves relative positive weights until they reach the cap and
redistributes the remaining column mass. No extra training is hidden in the
preparation step.

For retained connection fraction `p`, `wmax = 1/p`. The one- and two-trace
potentiation coefficient is `0.0005 / p^0.8`; two-trace depression is
`0.0001 / p^0.8`. Both use additive traces with 20 ms decay, presynaptic target
0.4 and weight exponent 0.2. One-trace has no presynaptic depression.
Two-trace depression uses the post trace before any simultaneous post
increment; potentiation includes a simultaneous pre increment. Three-trace
uses the existing nearest-spike rule with coefficients `0.01/p` and
`0.0001/p` and trace time constants 20/20/40 ms. This scaling preserves the
fractional update of a weight expressed in dense-equivalent units.

All MNIST feedforward populations use GeNN sparse storage, including the
fully connected case. Dense cases use postsynaptic parallelism; sparse cases
use presynaptic parallelism with 32 threads per spike on every GPU. This fixes
the implementation policy across machines. It is not a fresh per-GPU
autotuning search over these scheduling choices.

### Brunel Contract

- 9,000 excitatory and 2,250 inhibitory neurons; timestep 0.1 ms.
- Fixed indegrees: 450 excitatory and 112 inhibitory inputs per neuron.
- 6,322,500 recurrent synapses, including 4,050,000 plastic E-to-E synapses.
- Procedural fixed indegree with replacement; recurrent autapses excluded.
- Zero GeNN delay steps, arrival-timed STDP, causal-boundary tie ordering.
- Recurrent delivered-current scale `1 / sqrt(0.05)`.
- External-rate scales: additive 0.47; Morrison 0.32.
- Seed and state seed 20260724; no periodic weight normalization.
- 100 ms presimulation followed by 1,000 ms measurement (10,000 steps).

These are the current GeNN defaults. Historical full-indegree, 1.5 ms-delay
Brunel results are a different workload and are excluded from the comparison.

## Measurement And Validation

For each new machine, preserve a source snapshot and hashes, checkpoint and
mask hashes, exact commands, dependency versions, generated CUDA, GPU model
and UUID, compute capability, host CPU, operating system, driver, clock and
power settings, and whether MIG or other GPU workloads are active.

Build for the actual target GPU. Reuse generated code only within the same
machine and exactly matching source, parameters, and build configuration.
Run cases sequentially on a selected GPU with one BLAS/OpenMP thread. Preserve
each run in a fresh directory and retain failures as well as successful runs.

The comparison uses a separate diagnostic pass followed by five
unprofiled timing repetitions per case. GeNN per-timestep event timing,
profilers, spike recording, and periodic diagnostic transfers are disabled
during timing. Compilation, allocation, dataset loading, and result writing
are outside the timed region. A final device read synchronizes completion
before the wall timer stops.

The principal metric is elapsed workload wall time divided by all executed
simulation steps. For MNIST, the numerator includes normalization, transfers,
retry decisions, stimulus, and rest; the denominator is attempts times 1,000.
For Brunel, the numerator covers only the measured simulation interval and its
completion read. Report median, range, and IQR across repetitions. Do not
combine profiler kernel durations with these wall-time measurements.

Diagnostic runs must check separate input/E/I spike totals, population firing
rates, retries, finite voltage/conductance/threshold state, feedforward column
sums, weight bounds, and concentration of activity across neurons. For Brunel,
also retain population temporal statistics and weight-distribution summaries.
Compare identical configurations across machines; investigate silence,
runaway activity, retry storms, or material changes in macroscopic dynamics
before accepting a timing row. A short diagnostic pass establishes numerical
and activity plausibility, not convergence or final classification accuracy.

## Recorded Results

### Four-GPU Matrix, Batch 20260907

This is the primary comparison: all 23 cases, five unprofiled repetitions per
case and GPU, **460 timing records** in total. Values below are median
microseconds per simulation step. No repetitions were discarded; the
[machine-readable record](genn-sweep/baseline-20260907-matrix.json) contains
all five values, minimum, maximum, IQR, diagnostics, commands and manifests
for every cell. IQR uses NumPy's linear quantile convention.

| Workload | RTX 3090 | A100 | A800 | H800 |
| --- | ---: | ---: | ---: | ---: |
| MNIST 1-trace, dense | 26.473 | 18.848 | 16.680 | 14.343 |
| MNIST 1-trace, Bernoulli 50% | 25.600 | 16.718 | 15.808 | 12.625 |
| MNIST 1-trace, Bernoulli 25% | 24.710 | 15.290 | 14.621 | 11.425 |
| MNIST 1-trace, Bernoulli 12.5% | 24.109 | 14.731 | 14.016 | 10.991 |
| MNIST 1-trace, fixed fan-out 50% | 25.673 | 16.682 | 15.908 | 12.674 |
| MNIST 1-trace, fixed fan-out 25% | 24.873 | 15.308 | 14.533 | 11.473 |
| MNIST 1-trace, fixed fan-out 12.5% | 24.284 | 14.702 | 13.990 | 11.007 |
| MNIST 2-trace, dense | 27.300 | 19.598 | 17.627 | 14.370 |
| MNIST 2-trace, Bernoulli 50% | 26.698 | 18.829 | 17.802 | 14.424 |
| MNIST 2-trace, Bernoulli 25% | 25.430 | 16.504 | 15.753 | 12.820 |
| MNIST 2-trace, Bernoulli 12.5% | 25.082 | 15.312 | 14.539 | 11.852 |
| MNIST 2-trace, fixed fan-out 50% | 27.257 | 19.117 | 17.751 | 14.541 |
| MNIST 2-trace, fixed fan-out 25% | 25.686 | 16.586 | 15.584 | 12.816 |
| MNIST 2-trace, fixed fan-out 12.5% | 25.112 | 15.278 | 14.466 | 11.812 |
| MNIST 3-trace, dense | 32.674 | 21.770 | 20.212 | 16.835 |
| MNIST 3-trace, Bernoulli 50% | 32.143 | 20.807 | 19.662 | 16.146 |
| MNIST 3-trace, Bernoulli 25% | 31.064 | 18.934 | 17.947 | 14.588 |
| MNIST 3-trace, Bernoulli 12.5% | 30.326 | 18.017 | 17.149 | 13.766 |
| MNIST 3-trace, fixed fan-out 50% | 32.109 | 20.812 | 19.648 | 16.302 |
| MNIST 3-trace, fixed fan-out 25% | 31.054 | 18.961 | 17.953 | 14.616 |
| MNIST 3-trace, fixed fan-out 12.5% | 30.405 | 17.990 | 17.179 | 13.752 |
| Brunel additive | 40.831 | 45.234 | 40.771 | 32.717 |
| Brunel Morrison | 41.247 | 53.687 | 52.253 | 42.238 |

Across the 21 MNIST cases, geometric-mean speedups relative to RTX 3090 are
1.562x for A100, 1.659x for A800 and 2.047x for H800. On the more closely
matched cluster hosts, H800 has **14-22% lower MNIST latency than A800**
(1.234x geometric-mean speedup), and 19-20% lower Brunel latency. In every
case, the slowest H800 repetition was faster than the fastest A800 repetition.
The A100 improves MNIST but is slower than RTX 3090 on both Brunel cases. These are
workload-specific system measurements, not a ranking by peak FP32 throughput.

#### Machines And Software

All four use GeNN 5.4.0, revision
`563c45c531eb6adce53ad3ff3f46d614a19abdb2`, NumPy 2.3.4 and SciPy 1.16.3.
The native runs use identical hashes for the nine recorded simulation/harness
source files, based on repository
`695015042a46d7766c83ede502e7ed8fa708da46` plus the uncommitted baseline
implementation. Source hashes are embedded in the result record. Kernels
were freshly generated and built for each GPU model; repetitions 2-5 reused
that model's first timing build. GeNN's default code-generation optimization
preferences were retained, with no CUDA graphs, kernel fusion or new per-GPU
tuning. This is a baseline of the specified implementation.

| GPU | Host CPU / OS | Driver | nvcc / GCC / Python | GPU power limit |
| --- | --- | --- | --- | ---: |
| RTX 3090, 24 GiB, cc 8.6 | Ryzen 9 7950X / NixOS container on WSL2 | 596.49 | 12.8.93 / 13.4 / 3.13.12 | 390 W |
| A100 PCIe, 40 GiB, cc 8.0 | EPYC 7742 / Ubuntu 24.04 | 570.211.01 | 12.8.61 / 13.3 / 3.12.3 | 250 W |
| A800 SXM4, 80 GiB, cc 8.0 | Xeon Platinum 8470 / Ubuntu 22.04 | 610.43.02 | 12.8.61 / 12.4 / 3.12.4 | 400 W |
| H800, 80 GB class, cc 9.0 | Xeon Platinum 8470 / Ubuntu 22.04 | 610.43.02 | 12.8.61 / 12.4 / 3.12.4 | 700 W |

GPU UUIDs and clock snapshots are in each manifest. Clocks were not locked;
a pre-run clock snapshot can show the idle clock and is not a measurement of
the clock during simulation. MIG was disabled on the datacenter GPUs. A100
used GPU 0 with host memory and CPU bound to NUMA node 0; other GPUs on that
host had unrelated work. RTX 3090 desktop contention cannot be excluded.

The A800 and H800 runs used idle GPUs on nodes `g55` and `g40`, respectively,
in separate eight-CPU Slurm steps within existing user allocations 499057
and 499509. Both used the same CPU model, OS, driver, CUDA compiler and shared
GeNN installation. Slurm's GPU visibility was preserved. This controls more
host variables than the RTX 3090/A100 comparison, although GPU clocks, power
limits and other node activity were not experimentally equalized.

Cluster SSH access used the configured alias `CRAFT-cluster`. All cluster
sources, dependencies, caches, temporary files and results reside below
`~/meow/stdp-experiments`, which resolves through `~/meow -> ~/WORK/meow`
to `/WORK/PUBLIC/zhangyouh_work/meow/stdp-experiments`. Dependencies, GeNN
sources, checkpoints and real MNIST were uploaded; no cluster network
downloads were used. The modules were `soft/anaconda3/config`,
`gpu/v12.8.1` and `compilers/gcc/v12.4.0`. Only the benchmark's Slurm
steps were completed; the user's parent allocations were left intact.

The batch label follows the conversation date. Raw manifests and logs retain
their machine UTC timestamps, including completed runs on 2026-09-06.

#### Dynamics Validation

All 23 cases have **identical separate population spike totals across all
four GPUs**. Every native repetition's total also matches its diagnostic
pass. All 21 MNIST cases additionally have identical per-neuron excitatory
spike-count vectors across the four GPUs. This is strong short-run activity
agreement, not a claim that every state variable is bitwise identical.

MNIST completed 100 accepted images in 100-103 attempts. Across the matrix,
there were 1,278-2,225 excitatory spikes and 143-293 active excitatory neurons
over each run. The largest single-neuron share was 15.56%. Mean final adaptive
thresholds ranged from 24.47 to 26.85 mV. No finite-state, runaway, retry-limit,
normalization or final weight-bound guard failed. Diagnostic state snapshots
were collected at the end of selected stimuli and after the final rest; they
do not bound every transient inside a stimulus. Sparse derived starts remain
performance inputs, without a new classification-accuracy or convergence claim.

| Brunel rule | E / I spikes, each GPU | E / I mean Hz | Mean ISI CV | All-E population Fano, 3 ms bins |
| --- | --- | --- | ---: | ---: |
| Additive | 39,755 / 10,141 | 4.417 / 4.507 | 0.564 | 1113.80 |
| Morrison | 53,597 / 13,492 | 5.955 / 5.996 | 0.713 | 40.25 |

Brunel recorded all 9,000 excitatory neurons for the measured second.
The additive case is substantially burstier than Morrison. These population
Fano factors do not establish an asynchronous-irregular regime or long-run
STDP equilibrium. They reproduce the current validated configuration's
short-run dynamics. Sampled plastic weights stayed interior and unimodal with
zero boundary mass. Across GPUs, final additive means were 45.469-45.475
with SD 1.013-1.019; Morrison means were 45.453-45.458 with SD 0.447-0.451.
The sampler uses 100,000 evenly spaced storage indices; full weight-array
bitwise equivalence was not tested.

The 30-test GeNN/Brunel/baseline suite passed on each of the four GPU models,
including generated-CUDA checks of two-trace event ordering, clipping and
plasticity-off behavior, and deterministic topology/projection tests.
The final 21-checkpoint input bundle and its manifest were regenerated
byte-for-byte from the portable parent checkpoints. Early RTX 3090 and A100
diagnostics used input bundle v1; its weights and theta arrays were verified
identical to v2. The intervening harness changes package the two-trace parent,
add state hashes and validate build reuse; the simulation loop and model
source are unchanged. All native repetitions and both cluster diagnostics
used v2.

#### Reproduction And Artifacts

The permanent result record embeds the prepared-input manifest, including
parent checkpoint, dataset, structural-mask, weight-array and theta-array
hashes. The input manifest SHA-256 is
`e94363abf5e97f786c7217c509ce79ce96e395329ff2d5dff0b1c26639d57e80`.
The four parent checkpoints live in `genn-sweep/checkpoints/`.

From the repository root, with the recorded GeNN/CUDA environment and a fresh
output prefix:

```sh
python genn-sweep/baseline.py prepare --output copilot/tmp/baseline_inputs_NEW
python genn-sweep/baseline.py run --inputs copilot/tmp/baseline_inputs_NEW \
  --output copilot/tmp/baseline_GPU_diagnostic --mode diagnostic
python genn-sweep/baseline.py run --inputs copilot/tmp/baseline_inputs_NEW \
  --output copilot/tmp/baseline_GPU_native_r1 --mode timing
# Repeat with fresh output names r2 through r5:
python genn-sweep/baseline.py run --inputs copilot/tmp/baseline_inputs_NEW \
  --output copilot/tmp/baseline_GPU_native_r2 --mode timing \
  --reuse-build-root copilot/tmp/baseline_GPU_native_r1/builds

python genn-sweep/summarize_baseline.py --inputs copilot/tmp/baseline_inputs_NEW \
  --machine GPU copilot/tmp/baseline_GPU_diagnostic copilot/tmp/baseline_GPU_native_r \
  --output copilot/tmp/baseline_GPU_summary.json
```

Add one `--machine` argument per GPU when aggregating. Defaults fix FP32,
seed 20260724, 100 accepted MNIST samples and all 23 cases. Use one selected
GPU and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`; on the cluster run inside
a Slurm GPU allocation and retain its `CUDA_VISIBLE_DEVICES`.

Full run artifacts use `copilot/tmp/baseline_<GPU>_20260907_*`, where
`<GPU>` is `3090_cuda128`, `a100`, `a800` or `h800`.
Downloaded cluster logs and wrappers are in
`copilot/tmp/cluster_logs_20260907` and
`copilot/tmp/cluster_helpers_20260907`. The wrappers keep compiler jobs within
the eight allocated CPUs; simulation BLAS/OpenMP threads remain one.
Local and A100 environment scripts are
`copilot/tmp/baseline_env_3090_cuda128_20260907.sh` and
`copilot/tmp/baseline_env_a100_20260907.sh`.

A100 source snapshots and full generated artifacts remain at
`/data/meow/stdp-baseline-20260907-v1` (diagnostics and Python environment)
and `/data/meow/stdp-baseline-20260907-v2` (native runs and profiles).
The original `/data/meow/stdp-experiments` checkout was preserved.
Full A800/H800 generated artifacts remain under the requested cluster root.
The permanent JSON preserves essential measurements and provenance even
though local `copilot/tmp/` artifacts are ignored by Git.

### Earlier RTX 3090 Fixed-Seed Control, Batch 20260907

Five sequential FP32 repetitions, using the contracts above and seed 20260724
for both models. Source: repository `695015042a46d7766c83ede502e7ed8fa708da46`,
GeNN `563c45c531eb6adce53ad3ff3f46d614a19abdb2` (5.4.0), with no simulation
source changes. Hardware: RTX 3090 24 GiB, Ryzen 9 7950X, NixOS container on
WSL2. Software: driver 596.49, nvcc 13.0.88, GCC 13.4.0, Python 3.13.12,
NumPy 2.3.4. The restored environment uses CUDA runtime 13.0.96 alongside
GeNN libraries linked against runtime 12.8.90; exact paths are in the manifest.
The GPU power limit was 390 W; clocks were not locked and desktop contention
cannot be excluded. The batch label follows the conversation date; diagnostic
manifests record the machine's independent UTC date as 2026-09-06.

| Workload | Median us/step | Range us/step | IQR us/step | Steps/run | Total spikes/run |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST dense, three-trace | 32.318 | 31.878-33.684 | 0.630 | 103,000 | 246,757 |
| MNIST dense, one-trace | 26.350 | 26.137-27.377 | 0.807 | 102,000 | 244,828 |
| MNIST Bernoulli 12.5%, one-trace | 23.673 | 23.603-24.422 | 0.259 | 100,000 | 242,939 |
| Brunel additive, 5% indegree, zero delay | 40.375 | 39.946-40.965 | 0.313 | 10,000 | 49,896 |
| Brunel Morrison, 5% indegree, zero delay | 40.767 | 40.599-41.620 | 0.074 | 10,000 | 67,089 |

Every repetition's total spike count matched its separate diagnostic pass.
Run 1 built fresh; runs 2-5 reused its generated models. All five repetitions
are included, including the slower first run. IQR is the 75th percentile minus
the 25th percentile using NumPy's linear quantile convention.

These medians are lower than the August control below, but the MNIST seed and
restored software environment differ. This is a refreshed comparison point,
not evidence of an implementation optimization or a change in GPU hardware.

The permanent [result record](genn-sweep/baseline-20260907-rtx3090.json)
contains every raw timing, diagnostics, exact commands, environment, source,
checkpoint, dataset and structural-mask hashes. Full generated CUDA, build
logs, diagnostic checkpoints and logs remain under
`copilot/tmp/gpu_baseline_20260907_3090_*`. These disposable directories are
ignored by Git; the JSON record preserves the essential comparison data.

Commands, from the repository root with the recorded environment:

```sh
python genn-sweep/run.py --precision float \
  --mnist-seed 20260724 --brunel-seed 20260724 \
  --work-dir copilot/tmp/gpu_baseline_20260907_3090_native_r1

# Repeat with fresh work directories r2 through r5.
python genn-sweep/run.py --precision float \
  --mnist-seed 20260724 --brunel-seed 20260724 \
  --reuse-build-root copilot/tmp/gpu_baseline_20260907_3090_native_r1/builds \
  --work-dir copilot/tmp/gpu_baseline_20260907_3090_native_r2
```

#### Dynamics Checks

MNIST diagnostics used 100 accepted images, per-attempt finite-state and
runaway checks, and summaries every 25 accepted images. There were no guard
failures. Rates below average over the entire population and include the rest
phase and retried attempts; low population-average rates reflect competition
among 400 excitatory neurons, not a silent network.

| MNIST case | Retries | Input / E / I spikes | E / I mean Hz | E spikes / accepted stimulus | Active E neurons / accepted stimulus |
| --- | ---: | --- | --- | ---: | ---: |
| Dense three-trace | 3 | 240,978 / 1,647 / 4,132 | 0.080 / 0.201 | 16.35 | 5.67 |
| Dense one-trace | 2 | 239,701 / 1,477 / 3,650 | 0.072 / 0.179 | 14.70 | 5.56 |
| Bernoulli 12.5% one-trace | 0 | 236,750 / 1,680 / 4,509 | 0.084 / 0.225 | 16.80 | 4.84 |

Final mean adaptive thresholds were 26.05, 24.48 and 26.85 mV, respectively.
Maximum weights were 0.7361, 0.9991 and 6.2914, below their respective bounds
of 1, 1 and 8, with zero upper-bound saturation. Final mean feedforward column
sums were 78.0042, 77.9980 and 77.9985. Individual final sums can differ from 78
because learning follows normalization; normalization runs before each attempt.
Sampled post-rest voltages remained finite and plausible, and conductances
decayed to the printed zero precision. These summaries do not measure peak
conductance or voltage extrema inside a stimulus.

Brunel diagnostics recorded all 9,000 excitatory neurons over the measured
second and sampled 100,000 plastic weights every 100 ms. Both completed the
requested interval and remained below the 100 Hz abort threshold.

| Brunel rule | E / I spikes | E / I mean Hz | E rate range, 100 ms bins | Mean ISI CV | Final weight mean +/- SD |
| --- | --- | --- | --- | ---: | --- |
| Additive | 39,755 / 10,141 | 4.417 / 4.507 | 2.289-8.227 Hz | 0.564 | 45.4769 +/- 1.0154 |
| Morrison | 53,597 / 13,492 | 5.955 / 5.996 | 5.407-6.601 Hz | 0.713 | 45.4537 +/- 0.4482 |

Both sampled weight distributions were interior and unimodal, with zero
boundary mass. E/I totals exactly reproduce the historical current-default
control. The additive population is substantially burstier: the all-E 3 ms
population-count Fano factors are 1113.8 versus 40.25 for Morrison. These are
population temporal statistics, not single-neuron Fano factors, and their
absolute values cannot be compared with earlier reports recording only 1,000
neurons. This validates the active short-run behavior of the specified
configuration; it does not establish an asynchronous-irregular state or
long-run STDP equilibrium. No new classification-accuracy claim is made.

### Historical RTX 3090 Control, 2026-08-28

These existing FP32 results use GeNN 5.4.0, CUDA compiler 13.0.88, driver
596.49, an RTX 3090 24 GiB, and an AMD Ryzen 9 7950X host under WSL2. Three
sequential repetitions were collected. Run 1 built fresh; runs 2 and 3 reused
its generated models. These rows predate the expanded 23-case suite.

| Workload | Median us/step | Range us/step | Total spikes |
| --- | ---: | ---: | ---: |
| MNIST dense, three-trace | 34.884 | 34.814-36.118 | 246,685 |
| MNIST dense, one-trace | 28.875 | 28.546-29.736 | 244,155 |
| MNIST Bernoulli 12.5%, one-trace | 26.344 | 25.786-26.794 | 244,318 |
| Brunel additive, 5% indegree, zero delay | 42.204 | 42.119-42.634 | 49,896 |
| Brunel Morrison, 5% indegree, zero delay | 42.193 | 41.858-42.311 | 67,089 |

MNIST totals include input, E, and I spikes over all accepted/retried stimulus
and rest steps. Brunel totals include E and I spikes during the measured
interval only. They are not directly comparable firing-rate denominators.

Source commands, provenance, and raw repetitions:
[genn-sweep/README.md](genn-sweep/README.md) and
[sparse005-zero-results-20260828.txt](genn-sweep/sparse005-zero-results-20260828.txt).

## Launch And Synchronization Evidence

A100 profiling was collected separately with Nsight Systems 2024.6.2 and
Nsight Compute 2025.1.0. The [profile record](genn-sweep/profile-20260907-a100.json)
preserves commands, kernel statistics, metric units and raw trace hashes.
Full traces and NCU CSVs are under `copilot/tmp/a100_profiles_20260907`
locally and `/data/meow/stdp-baseline-20260907-v2/profiles` remotely.

Generated code and the traces identify these serial stages of a timestep:

| Order | Kernel | Work |
| ---: | --- | --- |
| 1 | `updatePresynapticKernel` | Deliver preceding spikes and apply presynaptic learning updates |
| 2 | `updatePostsynapticKernel` | Apply postsynaptic learning updates |
| 3, three-trace only | `neuronPrevSpikeTimeUpdateKernel` | Preserve previous spike times for the nearest-spike traces |
| 4 | `neuronSpikeQueueUpdateKernel` | Reset population spike counters for the next neuron update |
| 5 | `updateNeuronsKernel` | Advance inputs, neuron state and traces; detect and record new spikes |

One- and two-trace MNIST and both Brunel rules launch **four kernels per
step**; three-trace MNIST launches **five**. Two-trace's count is verified
from generated code; its own Nsight trace was not collected. The one-warp
spike-counter reset is a separate kernel even when little event work occurs.
The stream orders these stages globally, but that is distinct from making
the CPU wait for each stage.

There were **zero explicit Synchronize API calls inside the selected kernel
spans**. Synchronous copies wait for the device at MNIST presentation
boundaries and at final completion. There is no explicit CPU/GPU wait on
every simulation step in this timing path.

The following Nsight Systems values are means from three accepted MNIST
images (3,000 steps), in microseconds per step:

| Case | Pre | Post | Previous time | Counter reset | Neurons | GPU kernel sum | CPU launch API sum | First-to-last kernel span / steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense one-trace | 5.047 | 3.383 | 0 | 2.485 | 4.306 | 15.221 | 17.040 | 25.959 |
| Bernoulli 12.5% one-trace | 4.663 | 3.262 | 0 | 2.487 | 4.331 | 14.743 | 15.399 | 23.259 |
| Dense three-trace | 6.499 | 3.516 | 4.010 | 2.481 | 4.636 | 21.142 | 20.118 | 33.730 |

CPU API time overlaps GPU execution: **do not add it to GPU kernel time or
subtract it from native wall time**. Kernel duration itself includes dispatch,
memory stalls and idle lanes, not just useful arithmetic. The gaps within
the first-to-last span also include workload boundary activity and profiler
effects. Profiling changes timing, and these short runs are not the 100-image
native measurements. The data do not isolate a precise percentage of native
latency attributable solely to launch overhead.

NCU sampled ten pre, ten post and ten neuron launches for each of these
three cases after skipping 900 matching launches, without cache or clock
control. Mean SM throughput was only 0.03-0.29% of peak in the dense sampled
kernels; sampled active-warp occupancy was 1.56%. The sparse presynaptic
kernel reached 1.10% SM throughput and 7.91% active-warp occupancy.
Mean DRAM throughput was below 0.25% of peak in all sampled kernels.
These are brief kernel samples, not whole-training utilization averages.
The dense pre/post/neuron grids contain only 27/25/51 one-warp blocks, and
the counter reset is one block. This is direct evidence of poorly filled
GPU execution capacity in this small network.

The separate Brunel additive trace covers just the first 50 ms after the
100 ms warmup. Its pre/post means were 29.7/35.5 us but medians were only
4.10/3.58 us, with burst-associated maxima near 0.9 ms. Unlike the small
MNIST cases, recurrent event work can dominate portions of this workload.
That short burst window must not replace the full-second Brunel baseline.

## Interpretation

The evidence supports substantial fixed launch/dispatch cost and serial
small-kernel execution in the current GeNN MNIST path. It does **not** support
the narrower claim that a CPU/GPU synchronization call occurs every frame,
or the stronger claim that approximately 25 us is a fundamental GPU floor.
Four or five kernels per tick are consequential for a 1,584-neuron workload,
but their count alone does not prove which could be removed while preserving
the current event ordering.

Newer hardware measurably improves this implementation: H800 reduces MNIST
latency by 14-22% relative to A800 on hosts with the same CPU model, driver
and compiler. This is a modest improvement compared with expectations based
on peak throughput, but it rejects the prediction of no significant latency
reduction. No H100 or RTX 5090 measurement is included yet.

A100 and RTX 3090 belong to the Ampere generation, but have different SM
resource limits, caches, memory systems, clocks and host environments.
A100 (compute capability 8.0) supports 64 resident warps and 32 resident
blocks per SM, compared with 48 warps and 16 blocks for compute capability
8.6. These differences can matter for GeNN's one-warp blocks; they do not
predict complete workload latency. See NVIDIA's
[Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/).

A CUDA Graph replay or a correctness-checked fused implementation would be
a useful next experiment to quantify avoidable submission and kernel-boundary
cost. NVIDIA documents reduced graph launch overhead in its
[CUDA Graph launch measurements](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/).
Neither is measured here, so these results are a baseline of GeNN's current
implementation, not the best achievable GPU latency. Older RTX 3090
profiling and its WSL2 limitations remain in
[genn-profiling.md](genn-profiling.md).

ActiveN's reported roughly 4 us/frame is not independently remeasured in this
batch. A hardware speedup claim should use the matching row, timestep,
connectivity, learning coefficients, activity regime, and inclusion of
normalization/transfers/rest specified here; comparison against a generic
25 us GPU value would obscure the observed workload and machine differences.
