# GeNN CUDA profiling results

Measured 2026-08-05 and 2026-08-06 UTC. The original profiling instructions are
retained after the results under [Profiling plan](#genn-cuda-profiling-plan).

## FP32 and synapse-span follow-up

Measured 2026-08-06 UTC after the WSL2 restart. All GeNN workloads in the
original study were FP64 workloads: both wrappers constructed
`GeNNModel("double", ...)`, so GeNN's `scalar` values and simulation time were
double precision. Integer indices, counts, and spike queues remained integer.
The wrappers now expose `--precision {double,float}`, defaulting to `double` so
existing commands retain their previous behavior. The float setting changes
model scalars, state initialization arrays, transferred weights/rates, and time
to FP32.

### Native FP64 versus FP32

The following are matched, timing-disabled controls run after the restart, not
comparisons against the earlier profiling table. Every case used GeNN timing
off, aggregate accounting off, a prebuilt model, and five sequential
repetitions. MNIST resumed the same immutable 10k checkpoint and trained 100
accepted samples with its existing span setting. Brunel used seed 20260724,
arrival timing, 100 ms presimulation, and a 1,000 ms measured interval. GPU
clocks were not locked, so only the interleaved FP64/FP32 comparison in this
batch should be used to estimate the precision effect.

| Workload | FP64 median us/cycle (IQR) | FP32 median us/cycle (IQR) | FP32 throughput speedup | Runtime reduction |
|---|---:|---:|---:|---:|
| MNIST triplet dense | 32.950 (0.278) | 32.647 (0.600) | 1.009x | 0.9% |
| MNIST one-trace dense | 26.805 (0.407) | 26.494 (0.772) | 1.012x | 1.2% |
| MNIST one-trace sparse 12.5% | 24.403 (0.749) | 23.868 (0.277) | 1.022x | 2.2% |
| Brunel additive | 107.138 (1.161) | 93.776 (0.024) | 1.142x | 12.5% |
| Brunel Morrison | 128.296 (1.176) | 119.947 (0.456) | 1.070x | 6.5% |

FP32 therefore does not materially improve these MNIST graphs. Their small
grids and four or five serial launches per cycle still dominate; halving scalar
width or using the RTX 3090's much higher FP32 arithmetic throughput cannot
create more work or remove launches. Brunel benefits, especially additive, but
far less than the theoretical FP32:FP64 throughput ratio because FP64 was only
one part of the cycle and the reverse-remap post path remains latency limited.

These are performance-compatible runs, not numerically identical trajectories.
All FP32 smoke and control runs completed without non-finite state, runaway
activity, boundary-weight accumulation, or MNIST normalization-guard failure.
MNIST median weight means stayed within 0.001% across precision, while the
short stochastic activity windows overlapped. Brunel was deterministic within
each precision in these five runs, but changing precision changed the
trajectory: additive firing was 4.643 versus 4.867 Hz (-4.6%), and Morrison was
6.108 versus 5.824 Hz (+4.9%). Median sampled weight means changed from 45.043
to 44.914 for additive and 45.217 to 45.174 for Morrison. Consequently FP32 is
a useful performance option, but it is not established as an accuracy- or
dynamics-equivalent replacement; no long training/accuracy validation was run.

Generated CUDA and `cuobjdump` confirm that the FP32 builds contain `float`
state/time and no double arithmetic in the timestep kernels. Additive's
pre/post register counts remain 44/40; Morrison pre falls from 48 to 42; both
neuron kernels fall from 48 to 40. Shared-memory sizes and 32-thread blocks are
unchanged.

### FP32 hardware counters

Nsight Compute used the same nine full sections and 20 replay passes per launch
as the original study. The main captures contain five pre, post, and neuron
launches after 100 cycles. Morrison's five main pre launches were empty, so a
separate 100-launch window supplied 36 active pre samples. The table reports
medians. Profiler durations must not be compared directly with native cycle
time, and FP32/FP64 per-kernel durations are not controlled event-count pairs
because the short precision trajectories differ.

| Rule | Kernel | Active n | Duration us | Occupancy | SM throughput | DRAM throughput | Issue | FP32 FMA pipe (active) | FP64 pipe (active) | Long scoreboard cycles/issued instruction |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Additive | pre | 5 | 9.66 | 15.41% | 2.27% | 3.57% | 4.00% | 0.72% | 0.00% | 20.12 |
| Additive | post | 5 | 22.08 | 8.68% | 0.61% | 12.94% | 1.13% | 0.16% | 0.00% | 74.23 |
| Additive | neuron | 5 | 6.56 | 9.72% | 4.37% | 12.10% | 6.40% | 1.83% | 0.00% | 7.70 |
| Morrison | active pre | 36 | 25.98 | 12.46% | 3.11% | 6.16% | 4.09% | 0.97% | 0.00% | 21.65 |
| Morrison | post | 5 | 23.07 | 8.96% | 0.77% | 13.49% | 1.18% | 0.35% | 0.00% | 49.08 |
| Morrison | neuron | 5 | 6.69 | 9.36% | 4.42% | 11.87% | 7.51% | 2.29% | 0.00% | 6.26 |

The SASS samples contain `FFMA`, `FADD`, `FMUL`, and `MUFU`, but no `DFMA`,
`DADD`, or `DMUL`; measured FP64-pipe utilization is zero in every row. FP32
FMA utilization is also low, so the FP32 kernels are not compute-bound on the
FP32 pipe. The additive and active Morrison pre paths now show modest SM use
with dependency/memory stalls, while both post paths remain especially
latency-limited: only about 1.1-1.2% of issue slots are used and long-scoreboard
stalls remain 49-74 cycles per issued instruction. This explains why removing
FP64 produces only a 7-14% native Brunel throughput improvement rather than a
large architectural-ratio speedup.

### GeNN equivalent of `setSpanType`

GeNN 5.4 exposes the corresponding controls on each synapse group:

```python
synapses.parallelism_hint = ParallelismHint.POSTSYNAPTIC
# or:
synapses.parallelism_hint = ParallelismHint.PRESYNAPTIC
synapses.num_threads_per_spike = 32
```

The Brunel wrapper now exposes these for E-to-E as
`--ee-parallelism {postsynaptic,presynaptic}` and
`--ee-num-threads-per-spike N`; defaults are the original postsynaptic/x1.
This hint controls only the presynaptic-delivery strategy. The separate
postsynaptic plasticity kernel and its reverse remap are unchanged.

The alternatives were tested separately in FP64, with timing and aggregate
accounting disabled. PostSpan uses the five-run matched control. Viable PreSpan
x8/x32 settings used three 1,000 ms repetitions. PreSpan x1 was already
pathological in the smoke test, so only additive received one full 1,000 ms
confirmation; Morrison x1 is reported from its 20 ms smoke and is not a
steady-state comparison.

| Rule | E-E scheduling | Repetitions / window | Median us/cycle (IQR) | Slowdown vs PostSpan | Rate Hz |
|---|---|---:|---:|---:|---:|
| Additive | PostSpan | 5 x 1,000 ms | 107.138 (1.161) | 1.00x | 4.867 |
| Additive | PreSpan x1 | 1 x 1,000 ms | 10,538.435 | 98.36x | 4.867 |
| Additive | PreSpan x8 | 3 x 1,000 ms | 1,711.687 (4.571) | 15.98x | 4.867 |
| Additive | PreSpan x32 | 3 x 1,000 ms | 546.348 (0.318) | 5.10x | 4.867 |
| Morrison | PostSpan | 5 x 1,000 ms | 128.296 (1.176) | 1.00x | 5.824 |
| Morrison | PreSpan x1 | 1 x 20 ms smoke | 8,605.721 | 67.08x* | 8.000* |
| Morrison | PreSpan x8 | 3 x 1,000 ms | 1,141.092 (1.114) | 8.89x | 5.824 |
| Morrison | PreSpan x32 | 3 x 1,000 ms | 432.883 (0.381) | 3.37x | 5.824 |

`*` The Morrison x1 ratio and rate use a short startup transient and are shown
only to establish that x1 is grossly non-viable.

Generated code explains the result. PostSpan assigns roughly 9,568 E-E target
workers and has them loop over the small number of actual source spikes.
PreSpan x1 reserves one worker for each of 9,000 possible spike slots, but only
the few active slots run and each active worker serially traverses approximately
9,000 outgoing edges. PreSpan x8/x32 splits each active fan-out, but reserves
72,000/288,000 E-E worker slots every cycle; most are idle at 4.9-5.8 Hz. The
full presynaptic launch grows from `738x32` at x1 to `904x96` at x8 and
`3154x96` at x32, while the post launch stays `282x32`. Full-window firing
rates are identical to PostSpan and sampled weight means remain close, so the
slowdown is a scheduling effect. PostSpan is decisively the right setting for
this low-rate, very-high-fan-out Brunel graph.

Implementation, commands, immutable controls, profiler reports, and compact
summaries are under
[`copilot/tmp/genn_precision_span_20260806`](copilot/tmp/genn_precision_span_20260806/).
The source-of-record tables are `precision_summary.csv`, `span_summary.csv`,
and `ncu_float_summary.csv`. Failed toolchain attempts and the discarded
overlapping profiler attempt remain preserved but are excluded from summaries.

## Result summary

The strongest result is that the existing GeNN timing mode is not a
low-overhead measurement mode. It inserts six CUDA event records, one event
synchronization, and three elapsed-time queries into every cycle. Fresh A/B
controls show 36.7-153.2% cycle-time overhead. The low-overhead aggregate
accounting added for this study costs 0.18-1.47%, below the planned 3%
threshold for all five workloads.

With GeNN timing disabled, the median cycle costs are 37.277 us for MNIST
triplet dense, 29.706 us for MNIST one-trace dense, 27.554 us for MNIST
one-trace sparse, 107.957 us for Brunel additive, and 130.342 us for Brunel
Morrison. These are short, current-source profiling controls, not replacements
for the longer historical rows in `performance.md`.

The evidence supports these conclusions:

- MNIST native execution is primarily limited by fixed launch and small-kernel
  overhead at this graph size. One-trace dense launches only 27 pre blocks, 25
  post blocks, one queue block, and 51 neuron blocks per cycle, all as one-warp
  blocks. The sparse graph removes 87.3% of feedforward pre traversals and 85.3%
  of post traversals in this window, but is only 7.2% faster than fresh dense
  control.
- Brunel performs enough device work to dominate launch cost. In the
  event-timed diagnostic build, the E-E reverse-remap postsynaptic kernel takes
  about 70% of the three reported stage timers. Morrison is 20.7% slower than
  additive while also firing and traversing about 20% more E-E events, so this
  experiment cannot attribute the whole difference to the power-law rule.
- Every timestep kernel uses a 32-thread block. On compute capability 8.6, the
  16-block-per-SM limit caps theoretical resident-warp occupancy at 16/48, or
  33.3%, even though registers and shared memory are not tighter limits.
  MNIST's three main grids are also smaller than the RTX 3090's 82 SMs.
- Hardware counters collected after restarting WSL2 confirm severe MNIST
  underutilization. Dense pre/post/neuron kernels achieve only 2.4-2.7%
  occupancy and have no eligible warp in 96.7-97.6% of scheduler cycles. Sparse
  presynaptic x32 raises achieved occupancy to 22.2%, but reaches only 1.1% SM
  and 1.3% memory throughput because its work remains small and irregular.
- Reverse-remap post kernels have measured global-sector amplification of
  2.17-3.87x over the ideal transaction count, versus 1.01-1.59x for the pre
  paths. Active sparse MNIST post work uses a median 176 DRAM bytes per plastic
  traversal, versus 59-64 bytes for the dense variants. This supports a real
  scattered-access penalty, but none of the MNIST kernels approaches bandwidth
  saturation.
- The original FP64 Brunel additive pre kernel is FP64-compute dominated:
  median SM throughput is 46.0%,
  active-cycle FP64-pipe use is 55.9%, and DRAM throughput is 12.1%. Its post
  kernel is instead latency limited: only 0.84% of issue slots are used, long
  scoreboard stalls consume 100.1 warp cycles per issued instruction, and DRAM
  throughput is 17.7%. Morrison post is mixed FP64/latency work at 31.4% SM,
  34.8% FP64-pipe, and 13.0% DRAM throughput.
- Global atomics are not the leading limiter in these samples. Additive pre has
  the highest atomic-pipeline use at 3.7% of elapsed cycles; its custom traffic
  pass reports a median 172,589 L2 atomic requests with a 99.18% hit rate.
  Atomic set conflicts are measurable, but the stronger constraints are FP64
  execution on the pre path and dependency latency on the post path.

## Measurement state and protocol

All five jobs ran sequentially with no other observed GPU process. The host was
an AMD Ryzen 9 7950X (16 cores/32 threads) and an RTX 3090 24 GiB, compute
capability 8.6, driver 596.49, persistence enabled, and a 390 W power limit.
The final environment capture was idle at P8, 45 C, and 210 MHz SM clock; clocks
were not locked and per-run temperature/clock telemetry was not sampled. The
software was GeNN/PyGeNN 5.4.0 at `563c45c531e`, Python 3.13.14, NumPy 2.5.1,
CUDA toolkit 13.0.88, and GCC 13.4.0. Nsight Systems was 2025.1.3 and Nsight
Compute was 2025.2.1.

Three modes were measured five times per workload; the timing-on and timing-off
models were built independently, while native and accounting shared the same
timing-off generated model:

- **native**: GeNN event timing off and aggregate accounting off;
- **accounting**: GeNN event timing off and host/component/event accounting on;
- **timed**: GeNN event timing on and accounting on.

MNIST resumed the immutable 10,000-accepted-sample checkpoint and trained to
10,100 accepted samples. Each attempt retained the normal 700 stimulus and 300
rest ticks. Triplet and dense one-trace used postsynaptic scheduling; sparse
one-trace used presynaptic x32 scheduling. The sparse checkpoint precedes the
historical first normalization-guard violation at 17k. Brunel used seed
20260724, scale 1, arrival timing, a 100 ms presimulation, and a 1,000 ms/10,000
step measured interval with the normal firing-rate guard enabled.

Reported intervals are percentile bootstrap intervals for the median using
10,000 resamples and seed 20260806. With only five observations, every listed
95% interval equals the observed minimum-maximum range; the IQR is therefore a
more useful compact variability measure. MNIST attempt counts vary because the
GeNN trajectory is not fully repeatable across independent processes, so
us/cycle rather than total wall time is the primary A/B metric.

## Non-profiled controls

| Workload | Median us/cycle [bootstrap 95%] | IQR us | Median cycles | Median simulation s | Median E2E s | MNIST attempts/retries |
|---|---:|---:|---:|---:|---:|---:|
| MNIST triplet dense | 37.277 [36.750, 38.615] | 1.540 | 102,000 | 3.840 | 4.549 | 102 / 2 |
| MNIST one-trace dense | 29.706 [29.483, 31.165] | 0.976 | 101,000 | 3.013 | 3.753 | 101 / 1 |
| MNIST one-trace sparse 12.5% | 27.554 [27.236, 28.300] | 0.515 | 100,000 | 2.755 | 3.412 | 100 / 0 |
| Brunel additive | 107.957 [107.391, 108.273] | 0.267 | 10,000 | 1.080 | 8.868 | - |
| Brunel Morrison | 130.342 [129.020, 130.954] | 0.360 | 10,000 | 1.303 | 8.078 | - |

Brunel E2E includes construction, allocation, sampling, and serialization and
is not proportional to the one-second measured interval. MNIST native attempt
ranges were 101-103 for triplet, 101-102 for dense, and exactly 100 for sparse.

### Instrumentation A/B

| Workload | Accounting median us/cycle | Accounting vs native | Meets 3% | GeNN-timed median us/cycle | Timed vs native |
|---|---:|---:|---:|---:|---:|
| MNIST triplet dense | 37.670 | +1.055% | yes | 82.968 | +122.571% |
| MNIST one-trace dense | 29.963 | +0.863% | yes | 70.903 | +138.680% |
| MNIST one-trace sparse 12.5% | 27.960 | +1.474% | yes | 69.754 | +153.157% |
| Brunel additive | 108.155 | +0.183% | yes | 157.621 | +46.003% |
| Brunel Morrison | 130.771 | +0.329% | yes | 178.140 | +36.671% |

Low-overhead accounting did not produce a gross dynamics change. The
event-timed build likewise showed no gross change in these limited behavioral
diagnostics, but it was 36.7-153.2% slower than native execution. This
behavioral similarity does not imply performance similarity. Across all three
modes, median input rates differ by at most 0.5%, and E spike rates by at most
about 5% in these small stochastic MNIST windows. Final median weight means
agree to six decimal places; maxima remained 0.736 for triplet, roughly 1.00
for dense one-trace, and 6.27-6.32 for sparse, with the normalization guard
passing. Brunel spike and traversal counts were identical across repetitions
and modes.

A representative timing-disabled 100-image probe of each MNIST mode's `r3`
checkpoint gave 74-75% triplet, 92-97% dense one-trace, and 89-95% sparse
one-trace accuracy. This optimistic same-activity assignment/scoring probe is
small and the training trajectories are not paired, so it is only a collapse
check, not evidence of exact accuracy equivalence.

The old MNIST figures in `performance.md` are numerically close only to the
fresh **event-timed** controls. This agreement is evidence that the historical
runtimes include the same timing overhead; it is not evidence that timed and
native performance are close. Fresh native figures are 48-60% lower than those
historical MNIST values. The historical and current Brunel windows also differ
in duration and activity. Consequently the historical measurement rows in
`performance.md` were not replaced; the source-of-record controls for this
study are the rows above.

## Verified timestep map

`runner.cc` calls `updateSynapses(t)` and then `updateNeurons(t)`. All generated
launches omit a stream argument and therefore use stream 0. Their dependencies
are serial: synaptic outputs and weight state are consumed or advanced before
the queue/neuron kernels emit the next tick's spikes.

| Workload | Per-cycle launch sequence, each block 32 threads |
|---|---|
| MNIST triplet dense | pre `27x32`; post `25x32`; previous-spike-time `38x32`; queue `1x32`; neurons `51x32` |
| MNIST one-trace dense | pre `27x32`; post `25x32`; queue `1x32`; neurons `51x32` |
| MNIST one-trace sparse | pre `798x32`; post `25x32`; queue `1x32`; neurons `51x32` |
| Brunel additive | pre `755x32`; post `282x32`; queue `1x32`; neurons `353x32` |
| Brunel Morrison | pre `755x32`; post `282x32`; queue `1x32`; neurons `353x32` |

GeNN fuses groups inside kernels but does not fuse the four or five timestep
stages:

- MNIST `updatePresynapticKernel` merges input-to-E delivery with E-to-I
  one-to-one excitation and dense I-to-E inhibition. Triplet depression is
  applied on input arrival. One-trace pre events deliver conductance only.
  Dense postsynaptic scheduling loops over a source row; sparse presynaptic x32
  maps a warp to each possible source spike and uses row lengths.
- MNIST `updatePostsynapticKernel` contains only feedforward plasticity and uses
  the reverse remap for scattered incoming-weight access. Triplet potentiation
  samples pre/previous-post times. One-trace decays the input-neuron `x` trace
  in the neuron kernel and performs its signed fractional-power update here.
- The MNIST queue helper clears the merged input/E/I spike counts. Triplet adds
  a separate previous-spike-time kernel over the 784 input and 400 E neurons.
  `updateNeuronsKernel` merges branches for Poisson input, 400 I neurons, and
  400 E neurons; it performs CURAND input sampling, conductance integration,
  refractory logic, adaptive theta, and spike-list writes.
- Brunel pre propagation merges three static recurrent groups and the plastic
  E-E group. It reads the 16-slot spike queues for the 1.5 ms delay, traverses
  fixed-indegree rows, uses atomic delivery, and applies E-E pre-path trace and
  weight updates. The post kernel handles only E-E plasticity through the
  reverse remap. The queue kernel advances both E and I ring slots.
- Brunel's neuron kernel merges the 2,250 I and 9,000 E populations. It consumes
  alpha-current accumulators, advances voltage/refractory state, samples
  independent Poisson multiplicity with a CURAND product loop, and writes the
  next queue slot and E recording bitset. Additive and Morrison have identical
  topology and launch geometry; Morrison retains `pow(g, 0.4)` and extra trace
  arithmetic in the generated plastic path.

At MNIST attempt boundaries the host pulls, normalizes, and pushes feedforward
weights; pushes 784 rates; runs 700 ticks; pulls E counts; checks the retry
condition and state bounds; pushes zero rates; and runs 300 rest ticks. Final
diagnostics and checkpoints pull additional arrays. Native execution has no
per-tick host synchronization, but the count/state pulls synchronize at those
boundaries. Brunel similarly runs asynchronously inside a chunk and
synchronizes when recorded state and diagnostics are pulled.

## Kernel timing and static resources

The following CUDA-event values come from the deliberately perturbed `timed`
build. The timer starts immediately around the named main kernel; queue and
triplet previous-time helpers are excluded. `Unattributed` is timed wall minus
the three reported event durations and includes driver launches, helper kernels,
six event records, the per-cycle synchronization, and three elapsed-time API
queries.

| Workload | Pre us/cycle | Post us/cycle | Neuron us/cycle | Three-stage sum | Unattributed timed wall |
|---|---:|---:|---:|---:|---:|
| MNIST triplet dense | 11.825 | 6.991 | 8.614 | 27.264 | 55.758 |
| MNIST one-trace dense | 7.545 | 6.854 | 8.364 | 22.762 | 48.240 |
| MNIST one-trace sparse 12.5% | 7.506 | 7.199 | 8.387 | 23.088 | 46.917 |
| Brunel additive | 23.287 | 74.955 | 9.393 | 107.486 | 49.773 |
| Brunel Morrison | 26.767 | 91.677 | 9.961 | 128.699 | 49.441 |

The nearly constant 47-56 us residual is direct evidence that the GeNN timer
mode adds a large fixed per-cycle cost. Within the reported stage sum, Brunel
post work is about 70%; MNIST is split more evenly, with triplet pre work
largest. These event values must not be added to or substituted for native wall
time.

`cuobjdump --dump-resource-usage` provides the static compiler data below. All
timestep kernels have zero local memory and zero stack use.

| Workload | Pre registers/shared B | Post registers/shared B | Neuron registers/shared B | Helper registers |
|---|---:|---:|---:|---:|
| MNIST triplet dense | 37 / 256 | 42 / 256 | 42 / 136 | previous-time 11; queue 6 |
| MNIST one-trace dense | 20 / 256 | 42 / 256 | 42 / 136 | queue 6 |
| MNIST one-trace sparse | 20 / 256 | 42 / 256 | 42 / 136 | queue 6 |
| Brunel additive | 44 / 256 | 40 / 256 | 48 / 140 | queue 10 |
| Brunel Morrison | 48 / 256 | 40 / 256 | 48 / 140 | queue 10 |

The one-warp block shape, not registers or shared memory, is the static
occupancy limit. The hardware counters below confirm that the 33.3% theoretical
limit is not reached by most kernels. Grid size and readiness reduce achieved
occupancy further; registers, shared memory, local memory, and stack use are not
the active constraints.

## Nsight profiler results

Both timing-on and timing-off Nsight Systems captures are archived. MNIST
captures contain one 1,000-tick attempt. Brunel captures contain 100 presim and
200 measured ticks. The API trace verifies the generated launch count:

| Workload | Native/timed kernel launches | Timed event records | Timed event syncs | Median native-capture launch API us |
|---|---:|---:|---:|---:|
| MNIST triplet dense | 5,002 | 6,004 | 1,002 | 4.990 |
| MNIST one-trace dense | 4,002 | 6,004 | 1,002 | 5.461 |
| MNIST one-trace sparse | 4,002 | 6,004 | 1,002 | 5.400 |
| Brunel additive | 1,203 | 1,804 | 302 | 5.851 |
| Brunel Morrison | 1,203 | 1,804 | 302 | 5.510 |

The two or three extra launches are initialization launches. Timing-off traces
contain no `cudaEventRecord` or `cudaEventSynchronize` calls in the timestep
loop. A roughly 5.4-5.9 us profiled CPU launch call multiplied by four launches
already accounts for much of the 27-30 us native one-trace MNIST cycle, although
the API number itself is profiler-perturbed and is not a native kernel duration.

Nsight Systems overhead was material. A single short native capture was
8.8-18.8% slower than its longer native control median; timing-on capture
overhead was 22.9-71.5% relative to the already-slow timed controls. The capture
windows include startup/cold effects, so these are observed perturbations, not
stable correction factors.

On this WSL2 host, every raw `.nsys-rep` was created and CUDA API tables were
exported, but `cuda_gpu_kern_sum`, `cuda_gpu_mem_time_sum`, and related GPU
tables contained zero rows. Kernel duration distributions, launch gaps on the
GPU, GPU-time shares, transfer durations, and overlap therefore cannot be
reported. Generated source and the API trace establish stream-0 ordering, but
not measured overlap efficiency.

### Nsight Compute hardware counters

After host counter access was enabled and WSL2 restarted on 2026-08-06, Nsight
Compute 2025.2.1 successfully collected `LaunchStats`, `Occupancy`,
`SpeedOfLight`, `ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis`,
`SchedulerStats`, `WarpStateStats`, `InstructionStats`, and `SourceCounters`.
Each full-section launch required 20 replay passes. A separate seven-pass
traffic capture collected exact DRAM read/write bytes, global load/store and
atomic sectors, L2 atomic requests and hit/miss counts, and set-conflict cycles.

The main capture contains five consecutive pre, post, and neuron launches after
100 warmup cycles. Since an MNIST postsynaptic kernel is usually empty, separate
200-launch windows yielded 6 active triplet, 4 active dense one-trace, and 4
active sparse one-trace full-section samples. The corresponding traffic windows
yielded 4, 3, and 7 active samples. Morrison's initial five presynaptic samples
preceded delayed recurrent delivery, so a 100-launch window supplied 32 active
full-section and traffic samples. Morrison post has three active samples; other
rows use all five main samples.

The table reports medians. `Issue` is the percentage of scheduler cycles with
an issued warp. Kernel durations are profiler-observed diagnostic values; replay
wall time and the sum of these durations are not native cycle measurements.

| Workload | Kernel | Active n | Duration us | Achieved occupancy | SM throughput | Memory throughput | DRAM throughput | Issue |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST triplet | pre | 5 | 10.18 | 2.50% | 0.61% | 1.05% | 0.52% | 3.03% |
| MNIST triplet | post | 6 | 8.30 | 2.46% | 1.42% | 1.06% | 0.68% | 2.67% |
| MNIST triplet | neuron | 5 | 6.21 | 2.68% | 3.54% | 2.02% | 1.39% | 3.31% |
| MNIST one-trace dense | pre | 5 | 6.88 | 2.65% | 0.13% | 1.34% | 0.71% | 2.45% |
| MNIST one-trace dense | post | 4 | 9.22 | 2.39% | 3.83% | 1.01% | 0.56% | 2.53% |
| MNIST one-trace dense | neuron | 5 | 6.27 | 2.67% | 3.53% | 2.11% | 1.49% | 3.31% |
| MNIST one-trace sparse | pre | 5 | 5.22 | 22.18% | 1.12% | 1.28% | 0.41% | 3.81% |
| MNIST one-trace sparse | post | 4 | 8.82 | 2.42% | 0.54% | 0.64% | 0.21% | 2.18% |
| MNIST one-trace sparse | neuron | 5 | 5.95 | 2.42% | 3.53% | 2.22% | 1.57% | 3.36% |
| Brunel additive | pre | 5 | 77.02 | 8.41% | 46.05% | 13.79% | 12.14% | 3.87% |
| Brunel additive | post | 5 | 107.97 | 7.85% | 13.47% | 30.22% | 17.68% | 0.84% |
| Brunel additive | neuron | 5 | 9.70 | 9.28% | 23.14% | 12.18% | 12.18% | 4.16% |
| Brunel Morrison | active pre | 32 | 38.93 | 9.07% | 38.17% | 11.57% | 7.69% | 3.65% |
| Brunel Morrison | active post | 3 | 46.88 | 9.01% | 31.40% | 13.32% | 13.03% | 1.51% |
| Brunel Morrison | neuron | 5 | 10.40 | 9.55% | 27.59% | 11.38% | 11.38% | 5.07% |

The active Morrison pre samples span the delayed-activity ramp and are not an
equal-event comparison with additive pre. They establish how Morrison scales
once recurrent delivery begins, while the longer accounting controls remain the
source for average workload rates. Additive pre's 46.05% SM throughput is set
by its FP64 pipe, which reaches 55.92% of active-cycle capacity. Morrison post
is more mixed at 34.82% FP64-pipe use. In contrast, every MNIST kernel stays
below 16% active-cycle FP64-pipe use and below 4% elapsed SM throughput.

The next table uses same-invocation SASS thread counts to normalize total DRAM
bytes: global atomic instructions identify delivered pre events, global stores
identify post plastic traversals, and neuron rows use population size. `Sector
amp` is measured global sectors divided by the ideal coalesced sector count.

| Workload | Path | Sector amp | DRAM B/event | L1 hit | L2 hit | Long / short scoreboard cycles per issued instruction |
|---|---|---:|---:|---:|---:|---:|
| MNIST triplet | pre | 1.01x | 37.8 | 46.2% | 78.4% | 14.3 / 3.6 |
| MNIST triplet | active post | 2.17x | 64.3 | 35.9% | 72.5% | 15.2 / 6.6 |
| MNIST one-trace dense | pre | 1.03x | 23.8 | 6.4% | 76.9% | 21.9 / 1.3 |
| MNIST one-trace dense | active post | 2.55x | 59.4 | 40.5% | 76.3% | 9.1 / 13.4 |
| MNIST one-trace sparse | pre | 1.10x | 89.0 | 64.4% | 84.9% | 14.4 / 0.8 |
| MNIST one-trace sparse | active post | 2.27x | 176.4 | 26.6% | 84.7% | 14.7 / 5.2 |
| Brunel additive | pre | 1.59x | 45.1 | 32.2% | 72.9% | 9.8 / 15.2 |
| Brunel additive | post | 3.87x | 53.8 | 47.2% | 54.6% | 100.1 / 13.1 |
| Brunel Morrison | active pre | 1.56x | 44.8 | 32.6% | 72.2% | 10.9 / 15.1 |
| Brunel Morrison | active post | 3.84x | 51.7 | 51.4% | 55.4% | 37.9 / 20.6 |

The custom traffic pass confirms that the pre kernels use global atomics, not
global reductions. Median L2 atomic request counts and hit rates are 125 and
80.0% for triplet, 75 and 66.7% for dense one-trace, 98 and 74.5% for sparse
one-trace, 172,589 and 99.18% for Brunel additive, and 91,843 and 98.46% for
active Morrison.
Additive and Morrison pre have 53,576 and 28,447 median L1 atomic set-conflict
cycles respectively, but atomic-pipeline utilization remains only 3.70% and
3.56% of elapsed cycles. No sampled synaptic path has a local-memory spill.

Synaptic warps average 27-32 active threads except triplet post at 23.6; branch
divergence is therefore secondary to readiness and memory dependency. The
merged Brunel neuron kernel averages about 20 active threads because its E/I
branches differ, but it lasts only about 10 us. Generated CUDA was not compiled
with line information, so `SourceCounters` preserves hot program counters but
cannot resolve them to source lines. All detailed instruction categories,
opcodes, raw counters, and per-launch values remain in the machine-readable
exports.

## Completed event and transfer counters

MNIST values are medians of the five accounting runs. Counts cover 100 accepted
samples plus retries; rates divide by all executed cycles. A pre traversal is
enqueued logical feedforward work derived from each input spike and its exact
row degree. A post traversal is exact incoming plastic work derived from each E
spike and the feedforward column degree.

| MNIST workload | Structural input-E | Cycles | Input spikes | E spikes | I spikes | Input-E pre total / cycle | Input-E post total / cycle | Transfer bytes / attempt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Triplet dense | 313,600 | 103,000 | 240,998 | 1,649 | 4,158 | 96,399,200 / 942.151 | 1,292,816 / 12.552 | 5,103,200 |
| One-trace dense | 313,600 | 102,000 | 240,159 | 1,468 | 3,604 | 96,063,600 / 941.800 | 1,150,912 / 11.283 | 5,103,681 |
| One-trace sparse | 39,001 | 100,000 | 237,345 | 1,696 | 4,546 | 11,930,869 / 119.309 | 166,097 / 1.661 | 667,152 |

The same counters imply E-to-I enqueued traversals of 1,649, 1,468, and 1,696,
and I-to-E traversals of 1,659,042, 1,437,996, and 1,813,854 respectively. The
recurrent structural counts are 400 E-I and 159,600 I-E. The counters describe
events enqueued by emission; a separately measured delivered-event count was
not added, so the final transport slot at a window boundary is not distinguished.

Brunel structural counts are exactly 81,000,000 E-E, 20,250,000 E-I,
20,250,000 I-E, and 5,062,500 I-I synapses. Full-population counters are exact,
not scaled from the recorded 1,000-neuron subset:

| Workload | E / I spikes | E-E pre / cycle | E-I pre / cycle | I-E pre / cycle | I-I pre / cycle | E-E post / cycle |
|---|---:|---:|---:|---:|---:|---:|
| Additive | 43,635 / 12,793 | 39,272.841 | 9,817.200 | 11,514.295 | 2,878.850 | 39,271.500 |
| Morrison | 52,602 / 14,061 | 47,336.771 | 11,835.301 | 12,653.750 | 3,164.450 | 47,341.800 |

These are delay-queue enqueues. The 15-tick delayed deliveries pending at the
end of the measured interval were not counted separately. Pre/post totals are
close but need not match inside a finite delayed window.

### MNIST host phases

Median accounting time per attempt separates the 700-tick stimulus and
300-tick rest paths from boundary work:

| Workload | Normalize ms | Rate push ms | Stimulus + E-count pull ms | Validation ms | Zero push + rest ms | Final diagnostics/checkpoint ms |
|---|---:|---:|---:|---:|---:|---:|
| Triplet dense | 3.710 | 0.070 | 23.752 | 0.321 | 9.841 | 0.136 |
| One-trace dense | 3.594 | 0.066 | 18.124 | 0.313 | 7.922 | 0.146 |
| One-trace sparse | 1.370 | 0.051 | 18.088 | 0.320 | 8.078 | 0.119 |

Normalization and transfer are meaningful per-presentation costs, especially
for dense graphs, but they do not explain the per-cycle dense/sparse similarity:
stimulus and rest execution remain nearly flat after the sparse reduction.

## Supported bottlenecks and next experiments

In priority order:

1. Keep GeNN event timing disabled for throughput measurements. Replace the
   per-cycle timer with interval-level events or an explicitly diagnostic build.
   The current aggregate counters already meet the 3% threshold.
2. Amortize the four/five driver launches per tick. Test a compiled multi-step
   loop, CUDA Graph replay, or generator-level fusion of queue maintenance with
   neuron update. MNIST is the clearest target: dense kernels achieve only
   2.4-2.7% occupancy, use an issue slot in 2.4-3.4% of scheduler cycles, and
   remain below 4% SM and 2.3% memory throughput.
3. Rework Brunel E-E post traversal locality. The reverse-remap path uses
   3.84-3.87x the ideal global sectors and sees only 55% L2 hits. Additive post
   spends 100.1 warp cycles per issued instruction on long-scoreboard stalls
   while using only 17.7% DRAM throughput, so latency and coalescing are more
   credible targets than raw bandwidth. Test a column-oriented or transposed
   plastic representation against an equal-activity control.
4. Test controlled precision changes on the Brunel plastic paths. Additive pre
   uses 46.0% SM throughput and 55.9% of the active-cycle FP64 pipe; Morrison
   post also reaches 34.8% FP64-pipe use. Compare FP32 or deliberately mixed
   precision only with trajectory, weight, firing, and accuracy validation.
5. Test wider blocks or another GeNN parallelism schedule. One-warp blocks cap
   theoretical occupancy at 33.3%, and the dense MNIST grids do not expose one
   block per SM. Sparse pre reaches 22.2% occupancy with 798 blocks but only
   1.1% SM throughput, so its presynaptic x32 mapping also needs a controlled
   work-efficiency comparison. Validate dynamics because scheduling can change
   atomic order. Do not prioritize atomic replacement yet: measured atomic
   pipeline use peaks at only 3.7%.
6. Move or batch MNIST normalization on device only after launch amortization.
   It can improve E2E presentation cost, but it is not the leading steady-state
   cycle bottleneck shown here.

## Artifacts

The immutable profiling root is
[`copilot/tmp/genn_profiling_20260806_main/`](copilot/tmp/genn_profiling_20260806_main/).
It contains:

- [`artifact_manifest.json`](copilot/tmp/genn_profiling_20260806_main/artifact_manifest.json)
  and [`artifacts.sha256`](copilot/tmp/genn_profiling_20260806_main/artifacts.sha256)
  with environment, source/input/checkpoint/generated-code hashes and exact
  protocol;
- [`control_summary.json`](copilot/tmp/genn_profiling_20260806_main/control_summary.json),
  [`derived_report_summary.json`](copilot/tmp/genn_profiling_20260806_main/derived_report_summary.json),
  and [`controls.csv`](copilot/tmp/genn_profiling_20260806_main/controls.csv);
- 75 control output directories and logs, nine accuracy checks, and fresh
  timing-on/off generated builds for all five workloads;
- timing-on and timing-off raw `.nsys-rep`, SQLite, CUDA API CSV, and empty GPU
  summary exports under [`nsys/`](copilot/tmp/genn_profiling_20260806_main/nsys/)
  and [`nsys_native/`](copilot/tmp/genn_profiling_20260806_main/nsys_native/);
- the exact failed `ncu` command in
  [`run_ncu_permission_probe.sh`](copilot/tmp/genn_profiling_20260806_main/run_ncu_permission_probe.sh)
  and denial log in
  [`ncu_permission_probe2.log`](copilot/tmp/genn_profiling_20260806_main/logs/ncu_permission_probe2.log);
- static `cuobjdump` reports under
  [`static/`](copilot/tmp/genn_profiling_20260806_main/static/); and
- all build, control, profiler, accuracy, summarization, and manifest scripts at
  the profiling root. Each launcher fails if its output directory already
  exists.

The hardware-counter supplement is independently preserved under
[`copilot/tmp/genn_profiling_20260806_ncu/`](copilot/tmp/genn_profiling_20260806_ncu/).
It contains:

- [`artifact_manifest.json`](copilot/tmp/genn_profiling_20260806_ncu/artifact_manifest.json)
  and [`artifacts.sha256`](copilot/tmp/genn_profiling_20260806_ncu/artifacts.sha256)
  covering 149 files, exact source/checkpoint hashes, environment state, and
  capture protocols;
- [`run_ncu.sh`](copilot/tmp/genn_profiling_20260806_ncu/run_ncu.sh), the exact
  overwrite-safe full-section, activity, and traffic capture commands;
- [`ncu_summary.json`](copilot/tmp/genn_profiling_20260806_ncu/ncu_summary.json),
  [`kernel_summary.csv`](copilot/tmp/genn_profiling_20260806_ncu/kernel_summary.csv),
  and [`traffic_summary.csv`](copilot/tmp/genn_profiling_20260806_ncu/traffic_summary.csv)
  with per-launch data, medians, ranges, IQRs, instruction categories, stalls,
  and event normalization;
- 10 full-section `.ncu-rep` files with raw, instance-expanded, and
  human-readable exports, plus 10 custom-traffic `.ncu-rep` files with raw CSV
  exports; and
- the preserved initial missing-SciPy failure from the restarted disposable
  environment. It occurred before Brunel constructed a network and was not used
  in any result.

---

# GeNN CUDA profiling plan

## Purpose

This document is the handoff for a separate profiling session. Its goal is to
explain why the GeNN CUDA implementations take the measured time per simulation
cycle, identify their synchronization and memory-access costs, and complete the
missing MNIST performance counters without changing network behavior.

The profiling session should analyze these five existing workloads:

| Family | Workload | GeNN variant / timing |
|---|---|---|
| MNIST | Three-trace STDP, dense | `triplet-dense` |
| MNIST | One-trace STDP, dense | `one-trace-dense` |
| MNIST | One-trace STDP, Bernoulli 12.5% | `one-trace-bernoulli-0125` |
| Brunel | Additive STDP | `arrival` timing, scale 1 |
| Brunel | Morrison power-law STDP | `arrival` timing, scale 1 |

The sparse MNIST case is deliberately the existing **one-trace** variant. No
sparse three-trace implementation currently exists. The Brunel NEST-boundary
cases are deliberately excluded: only the stable `arrival` runs are in scope.

This is a profiling and measurement task, not an optimization task. Preserve a
baseline before modifying instrumentation, and do not mix performance numbers
from an instrumented build into [`performance.md`](performance.md) unless the
instrumentation overhead has been bounded as described below.

## Safety and reproducibility

- Follow `AGENTS.md`: inspect the live worktree and obtain confirmation before
  editing code. Do not commit, delete, overwrite, or reuse existing run
  directories.
- Other jobs may be active. Inspect process and GPU state first, do not kill
  them, and run the five profiling workloads sequentially. Previous concurrent
  GPU work caused the host container to be OOM-killed.
- Use a new immutable output directory for every run. Record the exact command,
  source hashes, generated-code hashes, checkpoint hashes, dataset, sample
  range/order, seed, environment, and profiler version.
- Record GPU model, persistence mode, clocks, power state, temperature, driver,
  CUDA toolkit, GeNN version/commit, compiler, and Python environment. Do not
  assume the temporary toolkit or virtual-environment paths used by earlier
  sessions still exist.
- Confirm CUDA with a minimal GeNN run before profiling. At the time this plan
  was written, the target was an RTX 3090 (24 GiB, compute capability 8.6), GeNN
  commit `563c45c` / version 5.4, and a patched CUDA 13.0 toolkit, but all of
  these must be rechecked.
- Check `ncu --version` and `nsys --version` during preflight. Neither executable
  was on `PATH` or found in the previously inspected temporary/Nix paths. Install
  or map compatible NVIDIA Nsight tools before starting. If hardware counters
  fail with `ERR_NVGPUCTRPERM`, arrange host-side profiling permission (or the
  required container capability); do not replace unavailable counters with
  estimates.

## Existing baselines

Use these artifacts as the initial comparison points. They are not substitutes
for fresh unprofiled control runs on the profiling host.

| Workload | Existing artifact | Cycles | Simulation wall | Time/cycle | Activity / result |
|---|---|---:|---:|---:|---|
| MNIST three-trace dense | [`genn_cuda_mnist_30k_20260725_a`](reimpl/runs/genn_cuda_mnist_30k_20260725_a/results/performance.json) | 30,746,000 | 2,245.391 s | 73.030 us | 89.2% at 30k; spike/update totals not recorded |
| MNIST one-trace dense | [`genn_cuda_onetrace_dense_train30000_post_20260805_a`](reimpl/runs/genn_cuda_onetrace_dense_train30000_post_20260805_a/results/performance.json) | 30,698,000 | 2,206.580 s | 71.880 us | 71,661,447 total spikes; 83.2% at 30k |
| MNIST one-trace sparse 12.5% | [`genn_cuda_onetrace_sparse0125_train30000_pre32_20260805_a`](reimpl/runs/genn_cuda_onetrace_sparse0125_train30000_pre32_20260805_a/results/performance.json) | 30,006,000 | 2,062.215 s | 68.727 us | 70,957,130 total spikes; 74.1% at 30k |
| Brunel additive arrival | [`port_genn_cuda_additive_arrival_scale1_10s_20260725_a`](brunel/runs/port_genn_cuda_additive_arrival_scale1_10s_20260725_a/results.json) | 100,000 | 10.285 s | 102.848 us | 28,455 E spikes; 23.049k E-E pre and post events/cycle |
| Brunel Morrison arrival | [`port_genn_cuda_morrison_scale1_2s_20260725_a`](brunel/runs/port_genn_cuda_morrison_scale1_2s_20260725_a/results.json) | 20,000 | 2.832 s | 141.589 us | 10,575 E spikes; 42.829k E-E pre and post events/cycle |

The Brunel table uses GeNN's reported simulation wall time. Existing end-to-end
wall times are 23.203 s for additive and 15.283 s for Morrison. The MNIST
one-trace end-to-end wall times are 2,212.438 s (dense) and 2,067.943 s (sparse).
See [`performance.md`](performance.md) for definitions and the complete tables.

Important baseline caveats:

- The old three-trace artifact lacks input/E/I spike totals and logical pre/post
  update counts. Filling those fields is part of this task.
- The old sparse 30k training run predates the current 2% post-normalization
  `wmax` guard and first violated that invariant in the block ending at 17k.
  Use a valid checkpoint/window before that violation, or perform a fresh run
  with the current guard. Label the choice.
- Dense one-trace used postsynaptic span x1. Sparse one-trace used presynaptic
  span x32. Preserve those settings for the baseline; alternative scheduling is
  a separate experiment.
- Existing GeNN one-trace accuracy does not match Brian 1 exactly: dense is
  83.2% versus 86.2%, and sparse is 74.1% versus 77.9%, under the recorded 30k
  protocol. Profiling must preserve each GeNN trajectory, not silently tune it.

## Phase 1: map a GeNN timestep

Start with source inspection and then verify the result against an actual CUDA
trace. Do not infer the launch sequence from GeNN API calls alone.

For each workload, inspect and archive the generated sources, especially:

- `runner.cc` and the model entry points;
- `neuronUpdate.cc`;
- `synapseUpdate.cc`;
- `customUpdate.cc`, if generated; and
- initialization, connectivity, and host transfer code.

The current MNIST wrapper is [`reimpl/backends/genn_backend.py`](reimpl/backends/genn_backend.py).
It calls `model.step_time()` once per 0.5 ms tick. At every presentation attempt
it also normalizes feedforward weights on the host (pull, NumPy normalization,
push), pushes the input-rate array, performs 700 stimulus ticks, pulls E spike
counts, pushes zero rates, and performs 300 rest ticks. Periodic diagnostics pull
additional state. Account for these attempt-boundary operations separately from
steady-state per-tick CUDA work.

The current Brunel wrapper is [`brunel/ports/genn_port.py`](brunel/ports/genn_port.py).
It also calls `step_time()` from a host loop. Its scale-1 graph has 9,000 E and
2,250 I neurons, 126,562,500 recurrent synapses in total, and 81,000,000 plastic
E-E synapses. E-E connectivity is sparse fixed-indegree connectivity with a
reverse remap for postsynaptic updates.

For every workload, produce a per-cycle sequence showing:

1. Kernel launch order, names, grid/block dimensions, stream, and dependencies.
2. The job performed by every kernel and the state arrays it reads/writes.
3. Which neuron populations or synapse groups GeNN merges inside a kernel.
4. Which stages remain separate launches. Generated models seen so far use
   `updateNeuronsKernel`, `updatePresynapticKernel`, and
   `updatePostsynapticKernel`, but verify the exact generated model.
5. Spike-queue and delay handling, random-number generation, conductance
   delivery, trace decay/update, weight update, bounds, and reverse-remap use.
6. Device-to-device dependencies, event records, host/device copies, explicit
   synchronizations, and implicit synchronization caused by host pulls.
7. Work performed only at stimulus/rest boundaries, presentation retries,
   normalization, statistics intervals, and checkpoint/evaluation boundaries.

Answer explicitly whether GeNN fuses work. Population/synapse groups may be
merged as branches within a generated kernel, while neuron, presynaptic, and
postsynaptic stages can still be distinct launches. Report both forms rather
than calling the whole timestep either "fused" or "unfused."

GeNN timing is already enabled in the wrappers. Generated runners expose CUDA
event timers such as `neuronUpdateTime`, `presynapticUpdateTime`, and
`postsynapticUpdateTime`, although the Python runners currently do not report
all of them. Prefer exposing these aggregate timers with one read at the end of
a measurement interval. First verify whether timing events themselves have
measurable overhead.

## Phase 2: establish non-profiled controls

Build once, warm up the GPU, and run at least five short, independent unprofiled
measurements per workload. Use a fixed checkpoint and a fixed sample/time window
for comparisons. Report median, minimum, maximum, and interquartile range for:

- simulation wall time and end-to-end wall time;
- cycles and microseconds per cycle;
- accepted samples, attempts, and retries for MNIST;
- firing and plastic-event counts; and
- time spent in normalization, state transfers, evaluation, and other host work.

Use a representative steady-state window rather than measuring initialization
alone. Separate MNIST stimulus and rest intervals because their spike traffic is
very different. Run Brunel presimulation/warmup before capturing a stable
measurement interval and keep the existing firing-rate safety guard active.

Compare an uninstrumented build with the proposed low-overhead counter/timer
build using the same protocol. Treat a median microseconds/cycle change of at
most 3% as the default "not observably changed" criterion, provided firing,
retry, weight, and accuracy diagnostics also remain compatible. Record noise and
confidence intervals; do not claim equivalence merely because one pair of runs
is close. If overhead exceeds 3%, keep a separate diagnostic build and retain
the uninstrumented build as the performance source of record.

CUDA profilers necessarily perturb execution. In particular, Nsight Compute can
replay and serialize kernels. Its wall time must never be used to assert native
throughput or compliance with the 3% criterion. Report Nsight Systems overhead
separately as well.

## Phase 3: Nsight Systems timeline

Use Nsight Systems first to establish the real launch and synchronization
sequence. Start with CUDA and NVTX tracing and minimal CPU sampling. Capture a
short range after warmup rather than the entire training run. If needed, add a
small gated NVTX or CUDA profiler range around a chosen stimulus/rest or Brunel
window, subject to the confirmation rule in `AGENTS.md`.

For each workload, report:

- kernel names, invocation counts, duration distribution, launch spacing, and
  percentage of captured GPU time;
- CPU launch/API time, gaps between kernels, explicit and implicit sync calls;
- transfer direction, size, frequency, duration, and whether transfers overlap
  kernels;
- whether a single stream serializes all work;
- attempt-boundary normalization and diagnostic pull/push costs for MNIST; and
- evidence for fixed launch/host overhead versus activity-dependent work.

Export both the raw `.nsys-rep` and machine-readable summary tables. Preserve
the exact capture command and capture-range definition.

## Phase 4: targeted Nsight Compute measurements

Do not run `ncu --set full` over a complete MNIST or Brunel run. It can require
many replay passes, produce huge outputs, and trigger another memory failure.
Select a few representative post-warmup invocations of each distinct kernel,
using kernel-name filters plus launch skip/count or profiler ranges supported by
the installed Nsight Compute version. Profile kernels sequentially and archive
the raw `.ncu-rep` files and CSV exports.

Collect metrics in staged passes, using the installed tool's section and metric
names. Useful standard sections include `LaunchStats`, `Occupancy`,
`SpeedOfLight`, `MemoryWorkloadAnalysis`, `SchedulerStats`, `WarpStateStats`,
`InstructionStats`, and `SourceCounters`. At minimum, obtain:

- duration, invocation count, grid/block size, registers/thread, shared/local
  memory, and theoretical/achieved occupancy;
- active, eligible, and issued warps per scheduler per cycle, issue-slot use,
  branch divergence, instruction mix, and FP64 use;
- SM compute utilization and relevant execution-pipe utilization;
- DRAM read/write bytes, transactions, throughput, and utilization;
- L1/TEX and L2 requests, sectors/transactions, hit rates, and throughput;
- requested versus actual global load/store traffic (or the current equivalent
  of global load/store efficiency), coalescing, and replay indicators;
- atomic instruction/transaction counts and contention indicators;
- warp stall reasons, especially long scoreboard, memory dependency, barrier,
  not selected, math-pipe throttle, LG throttle, and MIO throttle; and
- source-correlated hot instructions when line information is available.

Normalize byte, request, sector, transaction, and instruction counts both per
kernel invocation and per relevant event. State clearly whether an event means
a source spike, an outgoing synapse traversal, a postsynaptic spike, or an
incoming plastic synapse traversal.

## Counters to complete

Collect the following outside `ncu` whenever low-overhead aggregate accounting
is possible:

- input, E, I, and total emitted spike counts;
- structural synapse counts by group;
- logical presynaptic delivery/traversal counts by synapse group;
- logical postsynaptic plastic traversal counts by synapse group;
- queued/delivered delay events, where distinct from source spikes;
- simulation cycles, accepted samples, attempts, and retries;
- simulation wall, end-to-end wall, and component times; and
- host/device transfer bytes and time, especially MNIST normalization and state
  diagnostics.

Use spike counts and known row/column degrees to derive aggregate logical
traversals where exact. Avoid adding an atomic operation to every synaptic event
in the performance build. If detailed plastic outcomes are needed, measure
positive, negative, zero, and clamped updates plus total absolute weight change
in a separate diagnostic build and quantify its overhead.

Terminology matters:

- Three-trace MNIST performs depression on presynaptic arrivals and potentiation
  on postsynaptic spikes.
- One-trace MNIST does no plastic weight write on presynaptic delivery; its
  signed weight updates occur on postsynaptic events. Presynaptic events still
  traverse weights to deliver conductance.
- In Brunel, every E spike ultimately participates as a presynaptic event on its
  outgoing E-E synapses and a postsynaptic event on its incoming E-E synapses.
  With uniform fixed indegree/outdegree totals, aggregate logical E-E pre and
  post traversals are therefore approximately 1:1, although delays mean they
  need not occur in the same cycle.

The dense MNIST presynaptic traversal is contiguous by source row. A
postsynaptic traversal uses the reverse remap and can access synaptic weights in
a scattered order. The same reverse-remap issue applies to Brunel E-E post
updates. Do not infer a "cache-line traffic explosion" from this layout alone:
measure sectors, transactions, hit rates, and bytes per event on the actual
Ampere GPU.

## Comparisons and hypotheses

Use the five workloads to separate, as far as the data permits:

- fixed host/launch/synchronization cost per cycle;
- cost proportional to source spikes and presynaptic traversals;
- cost proportional to postsynaptic spikes and reverse-remap traversals;
- extra trace arithmetic/state traffic in three-trace versus one-trace STDP;
- contiguous dense traversal versus sparse/reverse-remapped access;
- register/shared-memory occupancy limits;
- memory bandwidth, cache latency/coalescing, atomic contention, and scheduler
  starvation; and
- MNIST attempt-boundary host normalization/copies versus Brunel steady-state
  device work.

The sparse one-trace graph reduces logical synaptic work by roughly 87% but the
recorded GeNN cycle time improves by only about 4.4% over dense one-trace. This
strongly suggests a large fixed or non-feedforward cost, but it is not yet proof
of a particular bottleneck. The profile should test that hypothesis.

A useful exploratory model is:

```text
time/cycle = fixed_cost
           + pre_traversals/cycle * pre_cost
           + post_traversals/cycle * post_cost
           + transfers/cycle * transfer_cost
```

Fit or bound this only after separating stimulus/rest and host-boundary work.
Five heterogeneous workloads are insufficient for a strong causal regression;
use controlled zero/low/high-spike windows or small fixed-rate microcases if
more points are needed, and label them as diagnostic rather than trained-network
performance.

## Artifacts and reporting

Create new uniquely tagged profiling directories without modifying the existing
run artifacts. Each profiling case should contain:

- a manifest with exact command, environment, source state, input/checkpoint
  hashes, profiler settings, warmup, capture range, and acceptance thresholds;
- stdout/stderr and unprofiled timing repetitions;
- generated CUDA source and hashes;
- raw `.nsys-rep` / `.ncu-rep` outputs and exported tables;
- activity/update counters and a machine-readable summary; and
- a short interpretation naming the limiting evidence and uncertainties.

Update [`performance.md`](performance.md) with missing MNIST counters and fresh
performance figures only from unprofiled runs or a low-overhead build that meets
the stated validation criterion. Put profiler-observed kernel times and
profiler overhead in a clearly separate profiling section or report. Do not
replace existing historical measurements.

The final handoff should provide:

1. A verified kernel/control-flow map for every workload, including fusion and
   synchronization points.
2. A per-kernel table of timeline share and the principal compute, scheduler,
   occupancy, and memory metrics.
3. Completed MNIST firing, pre/post traversal, timing, retry, and transfer data.
4. An explicit A/B overhead check showing whether added counters preserve
   native microseconds/cycle and dynamics.
5. A comparison of the five workloads that identifies supported bottlenecks,
   rejects unsupported hypotheses, and ranks the next optimization experiments.
