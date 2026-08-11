# Performance baseline

**WARNING: Every preserved GeNN CUDA runner used for this report had GeNN
per-timestep timing enabled, including a CUDA synchronization every cycle. These
are instrumented runtimes, not native-throughput baselines. Matched controls
found 2.23-2.53x runtime inflation for MNIST and 1.37-1.46x for Brunel. Do not
rescale historical rows mechanically; use the timing-disabled controls in
`genn-profiling.md`. NEST-GPU did not use this profiler path, although its runs
include spike-recording and chunk-boundary overhead.**

Generated 2026-08-04 and expanded 2026-08-05. The original inventory was built
from existing artifacts; the one-trace GPU extension records new sequential
runs on the same AMD Ryzen 9 7950X / RTX 3090 host. They are operational
baseline measurements, not isolated peak throughput.

## Scope and interpretation

The report covers the primary zero-delay three-trace MNIST training runs and
all 83 Brunel directories that contain `results.json`. Twenty Brunel directories
have a manifest but no result and are listed at the end. The Brunel inventory
includes smoke tests, parameter sweeps, early-stopped unstable runs, and
superseded timing experiments so that no completed measurement is silently
dropped. Use the canonical tables for architectural comparisons.

Three timing scopes must not be mixed:

- **E2E** is `whole_program_wall_seconds` for Brunel and includes construction,
  compilation/build where applicable, simulation, sampling, and serialization.
- **Simulation wall/step** is the measured simulation phase divided by its
  executed steps. It excludes presimulation, construction, and result writing.
- MNIST's **workload wall** covers normalization, image setup, all stimulus/rest
  attempts, retry decisions, and simulator execution. Most MNIST runners did
  not record whole-process E2E time, so that field is reported as unavailable.

The existing artifacts are not profiler traces. In particular, they do not
contain actual synapse-kernel invocation counters or time split by neuron,
presynaptic, and postsynaptic work. The event-derived Brunel columns are logical
work estimates:

```text
recorded E spikes = round(rate_hz * N_record * measured_ms / 1000)
estimated all-E spikes = recorded E spikes * N_E / N_record
E spikes/step = estimated all-E spikes / measured_steps
E->E pre updates/step = E spikes/step * C_E
E->E post updates/step = E spikes/step * C_E          (plastic rules only)
ns/pre update = simulation_wall_s * 1e9 / estimated E->E pre updates
ns/post update = simulation_wall_s * 1e9 / estimated E->E post updates
```

`C_E` is both the fixed E-to-E indegree and the mean E-to-E outdegree. Therefore
the logical pre and post counts are equal for these plastic graphs. A leading
`~` marks extrapolation from the first 1,000 excitatory neurons; it is exact
when all E neurons were recorded. Counts cover the measured phase only, not
presimulation. Inhibitory spikes and E-to-I, I-to-E, I-to-I, and external-input
events were not recorded and are excluded.

The `ns/update` values are **amortized whole-simulation wall per logical E-to-E
event**, not measured synapse service time. They include neuron integration,
external input, other synapse groups, recording, synchronization, and framework
overheads. NEST archives post spikes and processes history from pre events, so
its "post update" is logical comparable work rather than an actual post callback.
NEST-GPU uses the documented nearest-pair approximation and is not an exact
all-to-all-trace implementation.

## MNIST training baseline

Every MNIST presentation attempt has 1,000 cycles of 0.5 ms. "Accepted E
spikes" sums periodic diagnostics for accepted stimulus windows only; it omits
retry-attempt and rest spikes. Consequently the post columns are lower bounds:
each recorded E spike addresses 784 dense input synapses. Input spikes were not
counted, so pre-update counts and pre-update cost cannot be reconstructed.

| Backend / artifact | Accepted / attempts | Cycles | Workload wall s | E2E s | us/cycle | Accepted E spikes | E spikes/cycle | Min post updates/cycle | Wall/min-post ns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Brian2 CPU [`pilot1000`](reimpl/runs/brian2_cpu_pilot1000_counterfix_20260723_a/manifest.json) | 1,000 / 1,086 | 1,086,000 | 460.676 | n/r | 424.195 | 9,942 | 0.00915 | 7.177 | 59,102 |
| GeNN CPU [`pilot1000`](reimpl/runs/genn_cpu_pilot1000_counterfix_20260723_a/manifest.json) | 1,000 / 1,084 | 1,084,000 | 156.245 | n/r | 144.138 | 10,477 | 0.00967 | 7.577 | 19,022 |
| GeNN CUDA [`pilot1000`](reimpl/runs/genn_cuda_mnist_pilot1000_20260725_a/results/performance.json) | 1,000 / 1,096 | 1,096,000 | 84.474 | n/r | 77.074 | n/r | n/r | n/r | n/r |
| NEST CPU fixed-step [`pilot1000`](reimpl/runs/nest_cpu_fixed_step_pilot1000_20260724_a/manifest.json) | 1,000 / 1,133 | 1,133,000 | 339.970 | n/r | 300.062 | 9,699 | 0.00856 | 6.711 | 44,709 |
| Brian2 CPU [`train30000`](reimpl/runs/brian2_cpu_train30000_counterfix_20260723_a/manifest.json) | 30,000 / 30,329 | 30,329,000 | 18,408.285 | n/r | 606.953 | 476,679 | 0.01572 | 12.322 | 49,257 |
| GeNN CPU [`train30000`](reimpl/runs/genn_cpu_train30000_counterfix_20260723_a/manifest.json) | 30,000 / 30,748 | 30,748,000 | 5,455.013 | n/r | 177.410 | 473,589 | 0.01540 | 12.075 | 14,692 |
| GeNN CUDA [`train30000`](reimpl/runs/genn_cuda_mnist_30k_20260725_a/results/performance.json) | 30,000 / 30,746 | 30,746,000 | 2,245.391 | n/r | 73.030 | n/r | n/r | n/r | n/r |
| NEST CPU [`train30000`](reimpl/runs/nest_cpu_train30000_ordering_lazy_20260723_a/manifest.json) | 30,000 / 35,544 | 35,544,000 | 17,181.876 | n/r | 483.397 | 287,988 | 0.00810 | 6.352 | 76,099 |
| Brian2CUDA [`one_attempt`](reimpl/runs/brian2cuda_mnist_one_attempt_20260725_a/results.json) | 1 / 1 | 1,000 | 0.309 | 29.099 | 308.925 | 10 | 0.01000 | 7.840 | 39,404 |

`n/r` means the quantity was not recorded. GeNN build time was 0.587 s for the
CPU pilot, 0.594 s for the CPU 30k run, 4.828 s for the CUDA pilot, and 4.777 s
for the CUDA 30k run; these are excluded from workload wall. The Brian2CUDA E2E
number includes 27.890 s of compilation and is a one-attempt runtime validation,
not end-to-end training throughput.

The main MNIST throughput baseline is therefore 606.953 us/cycle for Brian2
CPU, 177.410 us/cycle for GeNN CPU, 73.030 us/cycle for GeNN CUDA, and 483.397
us/cycle for NEST CPU. These are not dynamics-matched: the final probe
accuracies were 89.2%, 87.3%, 89.2%, and 48.4%, respectively. NEST is an
implementation-mismatch result, not a performance peer for the convergent
zero-delay workload.

## One-trace MNIST GPU extension

These measurements add the dense one-trace network and its seeded 12.5%
Bernoulli derivative. The dense graph has 313,600 structural feedforward
synapses. The sparse mask exactly matches the Brian 1 mask generated by NumPy
`RandomState(20260723)` and has 39,001 structural synapses. Both use the
zero-delay midpoint model, 0.5 ms cycles, 700-cycle stimuli, 300-cycle rests,
and the optimistic fixed 1,000-image checkpoint probe documented in
`derived-networks.md` and `sparse-networks.md`.

Unlike the older MNIST artifacts, the new runners count input, excitatory, and
inhibitory spikes plus logical feedforward pre- and post-event work. A logical
pre update is one structural synapse reached by an input spike; a logical post
update is one structural incoming synapse visited for an excitatory spike.
The ns/update columns are still amortized whole-simulation costs, not isolated
kernel timings.

### Complete GeNN CUDA training

| Variant / artifact | Accepted / attempts | Ticks | Simulation s | E2E s | us/cycle | Cycles/s | Total firing | Firing/cycle | Pre/cycle | Post/cycle | ns/pre | ns/post |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense, postsynaptic x1 [`30k`](reimpl/runs/genn_cuda_onetrace_dense_train30000_post_20260805_a/results/performance.json) | 30,000 / 30,698 | 30,698,000 | 2,206.580 | 2,212.438 | 71.880 | 13,912.0 | 71,661,447 | 2.3344 | 913.775 | 11.3055 | 78.663 | 6,357.973 |
| Bernoulli 12.5%, presynaptic x32 [`30k`](reimpl/runs/genn_cuda_onetrace_sparse0125_train30000_pre32_20260805_a/results/performance.json) | 30,000 / 30,006 | 30,006,000 | 2,062.215 | 2,067.943 | 68.727 | 14,550.4 | 70,957,130 | 2.3648 | 115.788 | 1.6510 | 593.558 | 41,627.641 |

Sparse connectivity reduced logical pre work by 87.3% and logical post work by
85.4%, but reduced cycle time by only 4.4%. This workload is therefore not
dominated by the counted feedforward work at this graph size. The corresponding
amortized ns/update values increase sharply when the denominator becomes sparse;
they do not mean an individual sparse update literally takes 7.5 times longer.

The sparse run completed as a diagnostic but violated the Brian 1 experiment's
normalization safety criterion. In the block ending at 17k, a weight was first
observed above the allowed `8 * 1.02 = 8.16`; transient maxima later reached 33.37 and
the final maximum was 25.02. Only a few weights were involved and column sums
remained close to 78, but the fractional-power update is not valid above
`wmax`. The reusable runners now enforce the same 2% guard and would stop this
trajectory during that block. Its 20k and 30k performance and accuracy are retained as
pre-guard diagnostic evidence, not as a valid Brian-equivalent run.

### GeNN scheduling sweep

Each row is a fresh 100-sample CUDA training run. `post x1` is the GeNN default
postsynaptic parallelism. `pre xN` sets `ParallelismHint.PRESYNAPTIC` and
`num_threads_per_spike=N` on the feedforward synapse group.

| Topology / scheduling | Attempts | us/cycle | E2E s | Relative to topology default |
|---|---:|---:|---:|---:|
| Dense post x1 [`run`](reimpl/runs/genn_cuda_onetrace_dense_sched_post_100_20260805_a/results/performance.json) | 116 | 70.143 | 12.965 | baseline |
| Dense pre x1 [`run`](reimpl/runs/genn_cuda_onetrace_dense_sched_pre1_100_20260805_a/results/performance.json) | 116 | 79.355 | 13.977 | 13.1% slower |
| Dense pre x8 [`run`](reimpl/runs/genn_cuda_onetrace_dense_sched_pre8_100_20260805_a/results/performance.json) | 115 | 75.608 | 13.469 | 7.8% slower |
| Dense pre x32 [`run`](reimpl/runs/genn_cuda_onetrace_dense_sched_pre32_100_20260805_a/results/performance.json) | 115 | 72.552 | 13.026 | 3.4% slower |
| Dense pre x64 [`run`](reimpl/runs/genn_cuda_onetrace_dense_sched_pre64_100_20260805_a/results/performance.json) | 113 | 77.482 | 13.576 | 10.5% slower |
| Sparse post x1 [`run`](reimpl/runs/genn_cuda_onetrace_sparse0125_sched_post_100_20260805_a/results/performance.json) | 100 | 65.397 | 11.276 | baseline |
| Sparse pre x32 [`run`](reimpl/runs/genn_cuda_onetrace_sparse0125_sched_pre32_100_20260805_a/results/performance.json) | 100 | 64.574 | 11.068 | 1.3% faster |

Presynaptic x32 was selected for the sparse 30k run, but the default remained
best for dense connectivity. More threads per spike do not guarantee better
throughput here because the 784-input graph is too small to amortize the extra
delivery scheduling.

### Brian2CUDA one-attempt validation

CUDA standalone does not expose the presentation-level normalization/retry/
checkpoint controller between attempts while preserving hidden device state.
These rows therefore cover one accepted 1,000-cycle attempt, not 30k training.
`Build+run` is dominated by a conservative single-job CUDA 13 compilation.

| Variant / artifact | Ticks | Simulation s | Build+run s | us/cycle | Cycles/s | Total firing | Firing/cycle | Pre/cycle | Post/cycle | ns/pre | ns/post |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense [`attempt`](reimpl/runs/brian2cuda_onetrace_dense_one_attempt_20260805_f/results.json) | 1,000 | 0.448015 | 433.769 | 448.015 | 2,232.1 | 2,427 | 2.427 | 957.200 | 7.840 | 468.047 | 57,144.770 |
| Bernoulli 12.5% [`attempt`](reimpl/runs/brian2cuda_onetrace_sparse0125_one_attempt_20260805_a/results.json) | 1,000 | 0.498991 | 432.620 | 498.991 | 2,004.0 | 2,431 | 2.431 | 119.989 | 0.980 | 4,158.640 | 509,174.490 |

At this one-attempt scale, the sparse graph is 11.4% slower, so launch and
neuron/controller overhead dominate. These rows have no checkpoint accuracy.

### NEST-GPU checkpoint validation

NEST-GPU cannot train the exact rule with the current synapse ABI: its callback
lacks the accumulated source trace required by the one-trace update. The rows
below are static inference from Brian 1 30k checkpoints using a documented
`aeif_cond_beta` approximation and one 0.5 ms transport step. Only E and I
spikes are recordable, so total firing and firing/cycle exclude Poisson input;
pre-update metrics are unavailable and post updates are zero because plasticity
is disabled.

| Variant / artifact | Attempts | Ticks | Simulation s | E2E s | us/cycle | Cycles/s | Recorded E+I firing | E+I/cycle | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense [`probe`](reimpl/runs/nestgpu_onetrace_dense_ref30k_eval1000_20260805_a/results/performance.json) | 1,005 | 1,005,000 | 371.879 | 374.148 | 370.029 | 2,702.5 | 54,020 | 0.05375 | 85.4% |
| Bernoulli 12.5% [`probe`](reimpl/runs/nestgpu_onetrace_sparse0125_ref30k_eval1000_20260805_a/results/performance.json) | 1,000 | 1,000,000 | 369.503 | 371.695 | 369.503 | 2,706.3 | 59,542 | 0.05954 | 76.7% |

These timings are not training peers. They establish that the NEST-GPU runtime
can reconstruct and execute both checkpoint topologies, but its 370 us/cycle is
about 5.5 times GeNN CUDA checkpoint inference and its neuron dynamics are only
an approximation.

### Accuracy and dynamics

| Variant / evaluator | 10k | 20k | 30k | Assigned at 30k | Spikes/image at 30k | Active E/image at 30k |
|---|---:|---:|---:|---:|---:|---:|
| Dense Brian 1 reference | 84.3% | 88.4% | 86.2% | 400 | 22.551 | 8.284 |
| Dense GeNN CUDA fresh | 74.4% | 82.7% | 83.2% | 387 | 15.313 | 4.420 |
| Dense NEST-GPU, Brian checkpoint | - | - | 85.4% | 400 | 26.940 | 11.977 |
| Sparse Brian 1 reference | 76.5% | 76.9% | 77.9% | 400 | 22.156 | 14.692 |
| Sparse GeNN CUDA fresh, pre-guard | 68.5% | 73.7% | 74.1% | 399 | 19.470 | 9.927 |
| Sparse NEST-GPU, Brian checkpoint | - | - | 76.7% | 400 | 29.665 | 19.676 |

The GeNN CUDA one-trace runs do not match Brian 1 dynamics. Dense training is
3.0 points lower at 30k and its inference is substantially quieter; sparse is
3.8 points lower and also narrower. NEST-GPU is close in accuracy only because
it starts from Brian-trained weights and theta. Its higher firing and active
counts show that this is not a dynamics match. Checkpoint paths, exact commands,
and SHA-256 values are recorded in `reimpl/RESULTS.md`.

## Canonical full-scale Brunel runs

All rows use `N_E = C_E = 9,000`, `N_I = C_I = 2,250`, 0.1 ms cycles, and
126,562,500 recurrent synapses, of which 81,000,000 are E-to-E. Recorded firing
is from 1,000 E neurons and all-E counts are estimates.

| Framework / case | Status | Steps | E2E s | us/step | Recorded E spikes | Est. all-E spikes | E spikes/step | E-E pre/step | E-E post/step | ns/pre | ns/post |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NEST CPU additive, native [`10s`](brunel/runs/equilibrium_additive_scale1_10s_20260724_a/results.json) | done | 100,000 | 372.460 | 3,370.657 | 416,452 | ~3,748,068 | 37.481 | 337.326k | 337.326k | 9.992 | 9.992 |
| Brian2 CPU additive, arrival [`10s`](brunel/runs/port_brian2_additive_scale1_10s_20260725_a/results.json) | done | 100,000 | 552.350 | 5,193.312 | 28,567 | ~257,103 | 2.571 | 23.139k | 23.139k | 224.437 | 224.437 |
| GeNN CPU additive, arrival [`10s`](brunel/runs/port_genn_additive_scale1_10s_20260725_a/results.json) | done | 100,000 | 901.916 | 8,784.119 | 29,512 | ~265,608 | 2.656 | 23.905k | 23.905k | 367.464 | 367.464 |
| GeNN CUDA additive, arrival [`10s`](brunel/runs/port_genn_cuda_additive_arrival_scale1_10s_20260725_a/results.json) | done | 100,000 | 23.203 | 102.848 | 28,455 | ~256,095 | 2.561 | 23.049k | 23.049k | 4.462 | 4.462 |
| NEST-GPU additive, nearest [`10s`](brunel/runs/port_nestgpu_additive_scale1_10s_20260725_a/results.json) | done | 100,000 | 44.679 | 434.593 | 29,287 | ~263,583 | 2.636 | 23.722k | 23.722k | 18.320 | 18.320 |
| GeNN CUDA additive, NEST timing [`cutoff`](brunel/runs/port_genn_cuda_additive_nesttiming_nestboundary_scale1_10s_seed20260724_a/results.json) | cut at 1.7 s | 17,000 | 23.773 | 539.465 | 44,216 | ~397,944 | 23.408 | 210.676k | 210.676k | 2.561 | 2.561 |
| NEST CPU Morrison, native tuned [`10s`](brunel/runs/stability_morrison_g8_eta3200_alpha102600_10s_seed20260724_a/results.json) | done | 100,000 | 121.265 | 621.834 | 58,587 | ~527,283 | 5.273 | 47.455k | 47.455k | 13.104 | 13.104 |
| Brian2 CPU Morrison, arrival [`2s`](brunel/runs/port_brian2_morrison_scale1_2s_20260725_a/results.json) | done | 20,000 | 185.196 | 7,918.705 | 10,296 | ~92,664 | 4.633 | 41.699k | 41.699k | 189.902 | 189.902 |
| GeNN CPU Morrison, arrival [`2s`](brunel/runs/port_genn_morrison_scale1_2s_20260725_a/results.json) | done | 20,000 | 322.502 | 14,582.691 | 10,918 | ~98,262 | 4.913 | 44.218k | 44.218k | 329.792 | 329.792 |
| GeNN CUDA Morrison, arrival [`2s`](brunel/runs/port_genn_cuda_morrison_scale1_2s_20260725_a/results.json) | done | 20,000 | 15.283 | 141.589 | 10,575 | ~95,175 | 4.759 | 42.829k | 42.829k | 3.306 | 3.306 |
| Brian2CUDA Morrison, NEST boundary [`4s`](brunel/runs/port_brian2cuda_morrison_nesttiming_nestboundary_scale1_4s_seed20260725_a/results.json) | done | 40,000 | 89.015 | 674.680 | 22,955 | ~206,595 | 5.165 | 46.484k | 46.484k | 14.514 | 14.514 |
| GeNN CUDA Morrison, NEST boundary [`cutoff`](brunel/runs/port_genn_cuda_morrison_nesttiming_nestboundary_scale1_10s_seed20260725_a/results.json) | cut at 3.1 s | 31,000 | 28.402 | 392.015 | 53,041 | ~477,369 | 15.399 | 138.591k | 138.591k | 2.829 | 2.829 |
| NEST-GPU Morrison, nearest [`2s`](brunel/runs/port_nestgpu_morrison_scale1_2s_20260725_a/results.json) | done | 20,000 | 7.687 | 324.674 | 10,459 | ~94,131 | 4.707 | 42.359k | 42.359k | 7.665 | 7.665 |

### Baseline conclusions

The faithful additive NEST-timing ports do not reproduce NEST CPU dynamics.
GeNN CUDA crosses the 100 Hz block-rate guard at 1.7 s, and the corresponding
Brian2CUDA diagnostics also become very active. Their low amortized ns/update
is partly a high-event-rate effect and must not be interpreted as a matched
speedup. The arrival-timing additive ports agree with each other near 2.6 all-E
spikes/step but differ from NEST's late synchronized 37.5 spikes/step.

The best dynamics-aligned Morrison comparison is NEST CPU tuned at 5.27 E
spikes/step versus Brian2CUDA at 5.16. Their simulation costs are 621.8 and
674.7 us/step. GeNN CUDA can match the low-rate regime for selected state seeds
(see the inventory), but the NEST-boundary seed shown above runs away. NEST-GPU
is fast but uses different nearest-pair learning semantics.

For future profiling, add device/CPU counters for E and I spikes, every synapse
group's delivered events, actual pre/post callbacks, delay-queue activity,
neuron-update time, plastic update time, Poisson/input time, recorder transfer,
host control, and synchronization. Without those counters this report can
normalize wall time by logical work, but it cannot locate a runtime bottleneck.

## Complete Brunel artifact inventory

`native` denotes NEST's native rule, `arrival` the original Brian2/GeNN port
timing, `nest_dendritic` the later NEST-like timing experiments, and `nearest`
the NEST-GPU approximation. Static rows have no plastic post update. A zero-
spike smoke test has no meaningful event-normalized time.

### Brian2 CPU

| Run | Rule/timing | Status | `N_E/C_E/N_rec` | Steps | E2E s | us/step | Recorded E spikes | All-E spikes | E spikes/step | E-E pre/step | E-E post/step | ns/pre | ns/post |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`port_brian2_additive_scale1_10s_20260725_a`](brunel/runs/port_brian2_additive_scale1_10s_20260725_a/results.json) | additive/arrival | done | 9000/9000/1000 | 100,000 | 552.350 | 5193.312 | 28,567 | ~257,103 | 2.571 | 23.139k | 23.139k | 224.437 | 224.437 |
| [`port_brian2_additive_scale1_250ms_20260725_a`](brunel/runs/port_brian2_additive_scale1_250ms_20260725_a/results.json) | additive/arrival | done | 9000/9000/1000 | 2,500 | 109.840 | 12619.031 | 1,340 | ~12,060 | 4.824 | 43.416k | 43.416k | 290.654 | 290.654 |
| [`port_brian2_additive_smoke_20260725_b`](brunel/runs/port_brian2_additive_smoke_20260725_b/results.json) | additive/arrival | done | 18/18/18 | 200 | 0.766 | 1088.638 | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - |
| [`port_brian2_additive_smoke_20260725_c`](brunel/runs/port_brian2_additive_smoke_20260725_c/results.json) | additive/arrival | cut@0.02s | 18/18/18 | 200 | 0.578 | 545.956 | 36 | 36 | 0.180 | 3.240 | 3.240 | 168504.820 | 168504.820 |
| [`port_brian2_morrison_scale1_250ms_20260725_a`](brunel/runs/port_brian2_morrison_scale1_250ms_20260725_a/results.json) | morrison/arrival | done | 9000/9000/1000 | 2,500 | 79.707 | 13552.336 | 1,313 | ~11,817 | 4.727 | 42.541k | 42.541k | 318.570 | 318.570 |
| [`port_brian2_morrison_scale1_2s_20260725_a`](brunel/runs/port_brian2_morrison_scale1_2s_20260725_a/results.json) | morrison/arrival | done | 9000/9000/1000 | 20,000 | 185.196 | 7918.705 | 10,296 | ~92,664 | 4.633 | 41.699k | 41.699k | 189.902 | 189.902 |
| [`smoke_brian2_morrison_nestexclude_20260725_b`](brunel/runs/smoke_brian2_morrison_nestexclude_20260725_b/results.json) | morrison/nest_dendritic | done | 9/9/9 | 10 | 86.403 | 44464.051 | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - |

### Brian2CUDA

| Run | Rule/timing | Status | `N_E/C_E/N_rec` | Steps | E2E s | us/step | Recorded E spikes | All-E spikes | E spikes/step | E-E pre/step | E-E post/step | ns/pre | ns/post |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`port_brian2cuda_additive_nesttiming_nestexclude_scale1_10s_seed20260724_a`](brunel/runs/port_brian2cuda_additive_nesttiming_nestexclude_scale1_10s_seed20260724_a/results.json) | additive/nest_dendritic | done | 9000/9000/1000 | 100,000 | 226.237 | 1643.133 | 689,630 | ~6,206,670 | 62.067 | 558.600k | 558.600k | 2.942 | 2.942 |
| [`port_brian2cuda_additive_nesttiming_scale1_2s_20260725_a`](brunel/runs/port_brian2cuda_additive_nesttiming_scale1_2s_20260725_a/results.json) | additive/nest_dendritic | done | 9000/9000/1000 | 20,000 | 132.817 | 3019.916 | 390,111 | ~3,510,999 | 175.550 | 1.580M | 1.580M | 1.911 | 1.911 |
| [`port_brian2cuda_additive_scale1_250ms_20260725_a`](brunel/runs/port_brian2cuda_additive_scale1_250ms_20260725_a/results.json) | additive/arrival | done | 9000/9000/1000 | 2,500 | 452.548 | 824.500 | 1,521 | ~13,689 | 5.476 | 49.280k | 49.280k | 16.731 | 16.731 |
| [`port_brian2cuda_additive_smoke_20260725_c`](brunel/runs/port_brian2cuda_additive_smoke_20260725_c/results.json) | additive/arrival | done | 18/18/18 | 20 | 35.910 | 479.050 | 1 | 1 | 0.050 | 0.900 | 0.900 | 532277.778 | 532277.778 |
| [`port_brian2cuda_morrison_nesttiming_nestboundary_scale1_4s_seed20260725_a`](brunel/runs/port_brian2cuda_morrison_nesttiming_nestboundary_scale1_4s_seed20260725_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 40,000 | 89.015 | 674.680 | 22,955 | ~206,595 | 5.165 | 46.484k | 46.484k | 14.514 | 14.514 |
| [`port_brian2cuda_morrison_nesttiming_nesttie_scale1_10s_seed20260725_a`](brunel/runs/port_brian2cuda_morrison_nesttiming_nesttie_scale1_10s_seed20260725_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 100,000 | 540.534 | 773.532 | 55,544 | ~499,896 | 4.999 | 44.991k | 44.991k | 17.193 | 17.193 |
| [`port_brian2cuda_morrison_nesttiming_scale1_10s_seed20260725_a`](brunel/runs/port_brian2cuda_morrison_nesttiming_scale1_10s_seed20260725_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 100,000 | 248.189 | 1769.905 | 772,093 | ~6,948,837 | 69.488 | 625.395k | 625.395k | 2.830 | 2.830 |
| [`port_brian2cuda_morrison_nesttiming_scale1_2s_20260725_a`](brunel/runs/port_brian2cuda_morrison_nesttiming_scale1_2s_20260725_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 20,000 | 86.550 | 741.000 | 11,502 | ~103,518 | 5.176 | 46.583k | 46.583k | 15.907 | 15.907 |
| [`port_brian2cuda_morrison_nesttiming_scale1_2s_seed20260725_a`](brunel/runs/port_brian2cuda_morrison_nesttiming_scale1_2s_seed20260725_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 20,000 | 85.267 | 695.745 | 10,995 | ~98,955 | 4.948 | 44.530k | 44.530k | 15.624 | 15.624 |
| [`port_brian2cuda_morrison_nesttiming_scale1_2s_seed20260726_a`](brunel/runs/port_brian2cuda_morrison_nesttiming_scale1_2s_seed20260726_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 20,000 | 86.816 | 729.095 | 12,074 | ~108,666 | 5.433 | 48.900k | 48.900k | 14.910 | 14.910 |
| [`smoke_brian2cuda_additive_nestboundary_ns001_is001_10ms_20260726_e`](brunel/runs/smoke_brian2cuda_additive_nestboundary_ns001_is001_10ms_20260726_e/results.json) | additive/nest_dendritic | done | 90/90/90 | 100 | 322.501 | 436.230 | 59 | 59 | 0.590 | 53.100 | 53.100 | 8215.254 | 8215.254 |
| [`smoke_brian2cuda_morrison_nestboundary_ns001_is001_10ms_20260726_a`](brunel/runs/smoke_brian2cuda_morrison_nestboundary_ns001_is001_10ms_20260726_a/results.json) | morrison/nest_dendritic | done | 90/90/90 | 100 | 324.185 | 549.540 | 134 | 134 | 1.340 | 120.600 | 120.600 | 4556.716 | 4556.716 |

### GeNN CUDA

| Run | Rule/timing | Status | `N_E/C_E/N_rec` | Steps | E2E s | us/step | Recorded E spikes | All-E spikes | E spikes/step | E-E pre/step | E-E post/step | ns/pre | ns/post |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`port_genn_cuda_additive_arrival_scale1_10s_20260725_a`](brunel/runs/port_genn_cuda_additive_arrival_scale1_10s_20260725_a/results.json) | additive/arrival | done | 9000/9000/1000 | 100,000 | 23.203 | 102.848 | 28,455 | ~256,095 | 2.561 | 23.049k | 23.049k | 4.462 | 4.462 |
| [`port_genn_cuda_additive_nesttiming_nestboundary_scale1_10s_seed20260724_a`](brunel/runs/port_genn_cuda_additive_nesttiming_nestboundary_scale1_10s_seed20260724_a/results.json) | additive/nest_dendritic | cut@1.7s | 9000/9000/1000 | 17,000 | 23.773 | 539.465 | 44,216 | ~397,944 | 23.408 | 210.676k | 210.676k | 2.561 | 2.561 |
| [`port_genn_cuda_additive_nesttiming_nestexclude_scale1_10s_seed20260724_a`](brunel/runs/port_genn_cuda_additive_nesttiming_nestexclude_scale1_10s_seed20260724_a/results.json) | additive/nest_dendritic | cut@0.8s | 9000/9000/1000 | 8,000 | 25.937 | 1477.348 | 36,441 | ~327,969 | 40.996 | 368.965k | 368.965k | 4.004 | 4.004 |
| [`port_genn_cuda_additive_nesttiming_nestexclude_scale1_10s_seed20260725_a`](brunel/runs/port_genn_cuda_additive_nesttiming_nestexclude_scale1_10s_seed20260725_a/results.json) | additive/nest_dendritic | cut@0.6s | 9000/9000/1000 | 6,000 | 26.120 | 1805.340 | 36,602 | ~329,418 | 54.903 | 494.127k | 494.127k | 3.654 | 3.654 |
| [`port_genn_cuda_additive_nesttiming_scale1_10s_20260725_a`](brunel/runs/port_genn_cuda_additive_nesttiming_scale1_10s_20260725_a/results.json) | additive/nest_dendritic | cut@2s | 9000/9000/1000 | 20,000 | 22.362 | 538.142 | 69,129 | ~622,161 | 31.108 | 279.972k | 279.972k | 1.922 | 1.922 |
| [`port_genn_cuda_additive_scale1_250ms_20260725_a`](brunel/runs/port_genn_cuda_additive_scale1_250ms_20260725_a/results.json) | additive/arrival | done | 9000/9000/1000 | 2,500 | 12.564 | 303.081 | 1,407 | ~12,663 | 5.065 | 45.587k | 45.587k | 6.648 | 6.648 |
| [`port_genn_cuda_morrison_nesttiming_nestboundary_scale1_10s_seed20260725_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_nestboundary_scale1_10s_seed20260725_a/results.json) | morrison/nest_dendritic | cut@3.1s | 9000/9000/1000 | 31,000 | 28.402 | 392.015 | 53,041 | ~477,369 | 15.399 | 138.591k | 138.591k | 2.829 | 2.829 |
| [`port_genn_cuda_morrison_nesttiming_nestexclude_scale1_10s_seed20260725_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_nestexclude_scale1_10s_seed20260725_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 100,000 | 49.029 | 224.855 | 67,365 | ~606,285 | 6.063 | 54.566k | 54.566k | 4.121 | 4.121 |
| [`port_genn_cuda_morrison_nesttiming_nesttie_scale1_10s_seed20260725_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_nesttie_scale1_10s_seed20260725_a/results.json) | morrison/nest_dendritic | cut@6.7s | 9000/9000/1000 | 67,000 | 44.345 | 364.932 | 103,223 | ~929,007 | 13.866 | 124.792k | 124.792k | 2.924 | 2.924 |
| [`port_genn_cuda_morrison_nesttiming_scale1_10s_seed20260725_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_scale1_10s_seed20260725_a/results.json) | morrison/nest_dendritic | cut@4.1s | 9000/9000/1000 | 41,000 | 29.001 | 329.089 | 75,109 | ~675,981 | 16.487 | 148.386k | 148.386k | 2.218 | 2.218 |
| [`port_genn_cuda_morrison_nesttiming_scale1_10s_seed20260726_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_scale1_10s_seed20260726_a/results.json) | morrison/nest_dendritic | cut@6.4s | 9000/9000/1000 | 64,000 | 34.742 | 258.973 | 86,221 | ~775,989 | 12.125 | 109.123k | 109.123k | 2.373 | 2.373 |
| [`port_genn_cuda_morrison_nesttiming_scale1_2s_20260725_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_scale1_2s_20260725_a/results.json) | morrison/nest_dendritic | cut@2s | 9000/9000/1000 | 20,000 | 19.991 | 351.649 | 39,518 | ~355,662 | 17.783 | 160.048k | 160.048k | 2.197 | 2.197 |
| [`port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260724_b`](brunel/runs/port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260724_b/results.json) | morrison/nest_dendritic | cut@2s | 9000/9000/1000 | 20,000 | 20.018 | 349.940 | 39,518 | ~355,662 | 17.783 | 160.048k | 160.048k | 2.186 | 2.186 |
| [`port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260724_state20260725_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260724_state20260725_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 20,000 | 15.980 | 168.370 | 13,618 | ~122,562 | 6.128 | 55.153k | 55.153k | 3.053 | 3.053 |
| [`port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260724_state20260726_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260724_state20260726_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 20,000 | 16.013 | 158.403 | 12,672 | ~114,048 | 5.702 | 51.322k | 51.322k | 3.086 | 3.086 |
| [`port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260725_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260725_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 20,000 | 15.763 | 152.401 | 12,227 | ~110,043 | 5.502 | 49.519k | 49.519k | 3.078 | 3.078 |
| [`port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260725_state20260724_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260725_state20260724_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 20,000 | 16.086 | 160.914 | 13,602 | ~122,418 | 6.121 | 55.088k | 55.088k | 2.921 | 2.921 |
| [`port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260726_a`](brunel/runs/port_genn_cuda_morrison_nesttiming_scale1_2s_seed20260726_a/results.json) | morrison/nest_dendritic | done | 9000/9000/1000 | 20,000 | 15.689 | 157.059 | 12,124 | ~109,116 | 5.456 | 49.102k | 49.102k | 3.199 | 3.199 |
| [`port_genn_cuda_morrison_scale1_2s_20260725_a`](brunel/runs/port_genn_cuda_morrison_scale1_2s_20260725_a/results.json) | morrison/arrival | done | 9000/9000/1000 | 20,000 | 15.283 | 141.589 | 10,575 | ~95,175 | 4.759 | 42.829k | 42.829k | 3.306 | 3.306 |
| [`smoke_genn_cuda_morrison_nesttie_20260725_a`](brunel/runs/smoke_genn_cuda_morrison_nesttie_20260725_a/results.json) | morrison/nest_dendritic | done | 18/18/18 | 10 | 5.484 | 63.804 | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - |

### GeNN single-threaded CPU

| Run | Rule/timing | Status | `N_E/C_E/N_rec` | Steps | E2E s | us/step | Recorded E spikes | All-E spikes | E spikes/step | E-E pre/step | E-E post/step | ns/pre | ns/post |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`port_genn_additive_scale1_10s_20260725_a`](brunel/runs/port_genn_additive_scale1_10s_20260725_a/results.json) | additive/arrival | done | 9000/9000/1000 | 100,000 | 901.916 | 8784.119 | 29,512 | ~265,608 | 2.656 | 23.905k | 23.905k | 367.464 | 367.464 |
| [`port_genn_additive_scale1_250ms_20260725_a`](brunel/runs/port_genn_additive_scale1_250ms_20260725_a/results.json) | additive/arrival | done | 9000/9000/1000 | 2,500 | 63.313 | 17308.029 | 1,790 | ~16,110 | 6.444 | 57.996k | 57.996k | 298.435 | 298.435 |
| [`port_genn_additive_smoke_20260725_e`](brunel/runs/port_genn_additive_smoke_20260725_e/results.json) | additive/arrival | cut@0.02s | 18/18/18 | 200 | 1.927 | 11.348 | 37 | 37 | 0.185 | 3.330 | 3.330 | 3407.676 | 3407.676 |
| [`port_genn_cpu_morrison_nesttiming_scale1_2s_20260725_a`](brunel/runs/port_genn_cpu_morrison_nesttiming_scale1_2s_20260725_a/results.json) | morrison/nest_dendritic | cut@1.9s | 9000/9000/1000 | 19,000 | 682.661 | 34530.379 | 58,541 | ~526,869 | 27.730 | 249.570k | 249.570k | 138.360 | 138.360 |
| [`port_genn_morrison_scale1_250ms_20260725_a`](brunel/runs/port_genn_morrison_scale1_250ms_20260725_a/results.json) | morrison/arrival | done | 9000/9000/1000 | 2,500 | 84.571 | 19325.036 | 1,289 | ~11,601 | 4.640 | 41.764k | 41.764k | 462.724 | 462.724 |
| [`port_genn_morrison_scale1_2s_20260725_a`](brunel/runs/port_genn_morrison_scale1_2s_20260725_a/results.json) | morrison/arrival | done | 9000/9000/1000 | 20,000 | 322.502 | 14582.691 | 10,918 | ~98,262 | 4.913 | 44.218k | 44.218k | 329.792 | 329.792 |
| [`port_genn_morrison_smoke_20260725_a`](brunel/runs/port_genn_morrison_smoke_20260725_a/results.json) | morrison/arrival | done | 18/18/18 | 1,000 | 1.795 | 21.005 | 374 | 374 | 0.374 | 6.732 | 6.732 | 3120.164 | 3120.164 |

### NEST CPU

| Run | Rule/timing | Status | `N_E/C_E/N_rec` | Steps | E2E s | us/step | Recorded E spikes | All-E spikes | E spikes/step | E-E pre/step | E-E post/step | ns/pre | ns/post |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`desktop_additive_scale1_250ms_20260724_a`](brunel/runs/desktop_additive_scale1_250ms_20260724_a/results.json) | additive/native | done | 9000/9000/1000 | 2,500 | 28.242 | 640.683 | 1,939 | ~17,451 | 6.980 | 62.824k | 62.824k | 10.198 | 10.198 |
| [`desktop_morrison_scale1_250ms_20260724_a`](brunel/runs/desktop_morrison_scale1_250ms_20260724_a/results.json) | morrison/native | done | 9000/9000/1000 | 2,500 | 31.942 | 2072.164 | 6,884 | ~61,956 | 24.782 | 223.042k | 223.042k | 9.290 | 9.290 |
| [`equilibrium_additive_scale1_10s_20260724_a`](brunel/runs/equilibrium_additive_scale1_10s_20260724_a/results.json) | additive/native | done | 9000/9000/1000 | 100,000 | 372.460 | 3370.657 | 416,452 | ~3,748,068 | 37.481 | 337.326k | 337.326k | 9.992 | 9.992 |
| [`pilot_additive_ns002_is002_10s_20260724_a`](brunel/runs/pilot_additive_ns002_is002_10s_20260724_a/results.json) | additive/native | done | 180/180/180 | 100,000 | 4.545 | 2.628 | 44,848 | 44,848 | 0.448 | 80.726 | 80.726 | 32.555 | 32.555 |
| [`pilot_additive_ns002_is1_10s_20260724_a`](brunel/runs/pilot_additive_ns002_is1_10s_20260724_a/results.json) | additive/native | done | 180/9000/180 | 100,000 | 22.925 | 138.777 | 164,801 | 164,801 | 1.648 | 14.832k | 14.832k | 9.357 | 9.357 |
| [`pilot_morrison_ns002_is002_10s_20260724_a`](brunel/runs/pilot_morrison_ns002_is002_10s_20260724_a/results.json) | morrison/native | done | 180/180/180 | 100,000 | 6.443 | 12.367 | 670,169 | 670,169 | 6.702 | 1.206k | 1.206k | 10.252 | 10.252 |
| [`pilot_morrison_ns002_is1_10s_20260724_a`](brunel/runs/pilot_morrison_ns002_is1_10s_20260724_a/results.json) | morrison/native | done | 180/9000/180 | 100,000 | 34.499 | 256.244 | 322,211 | 322,211 | 3.222 | 28.999k | 28.999k | 8.836 | 8.836 |
| [`smoke_additive_20260724_b`](brunel/runs/smoke_additive_20260724_b/results.json) | additive/native | done | 18/9/18 | 50 | 0.030 | 2.556 | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - |
| [`smoke_early_stop_20260725_a`](brunel/runs/smoke_early_stop_20260725_a/results.json) | static/native | cut@0.005s | 18/18/18 | 50 | 0.033 | 1.482 | 113 | 113 | 2.260 | 40.680 | - | 36.435 | - |
| [`smoke_morrison_tuning_20260724_a`](brunel/runs/smoke_morrison_tuning_20260724_a/results.json) | morrison/native | done | 18/18/18 | 20 | 0.028 | 1.512 | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - |
| [`smoke_static_tuning_20260724_a`](brunel/runs/smoke_static_tuning_20260724_a/results.json) | static/native | done | 18/18/18 | 20 | 0.028 | 1.495 | 0 | 0 | 0.000 | 0.000 | - | - | - |
| [`stability_morrison_g5_eta1685_alpha053865_1s_seed20260724_a`](brunel/runs/stability_morrison_g5_eta1685_alpha053865_1s_seed20260724_a/results.json) | morrison/native | done | 9000/9000/1000 | 10,000 | 216.789 | 18750.215 | 267,974 | ~2,411,766 | 241.177 | 2.171M | 2.171M | 8.638 | 8.638 |
| [`stability_morrison_g5_eta1685_alpha056430_1s_seed20260724_a`](brunel/runs/stability_morrison_g5_eta1685_alpha056430_1s_seed20260724_a/results.json) | morrison/native | cut@0.4s | 9000/9000/1000 | 4,000 | 57.760 | 7839.095 | 45,051 | ~405,459 | 101.365 | 912.283k | 912.283k | 8.593 | 8.593 |
| [`stability_morrison_g5_eta1685_alpha061560_1s_seed20260724_a`](brunel/runs/stability_morrison_g5_eta1685_alpha061560_1s_seed20260724_a/results.json) | morrison/native | cut@0.6s | 9000/9000/1000 | 6,000 | 87.580 | 9976.621 | 80,769 | ~726,921 | 121.153 | 1.090M | 1.090M | 9.150 | 9.150 |
| [`stability_morrison_g8_eta3200_alpha061560_10s_seed20260724_a`](brunel/runs/stability_morrison_g8_eta3200_alpha061560_10s_seed20260724_a/results.json) | morrison/native | cut@2.2s | 9000/9000/1000 | 22,000 | 113.679 | 3620.906 | 103,002 | ~927,018 | 42.137 | 379.235k | 379.235k | 9.548 | 9.548 |
| [`stability_morrison_g8_eta3200_alpha061560_1s_seed20260724_a`](brunel/runs/stability_morrison_g8_eta3200_alpha061560_1s_seed20260724_a/results.json) | morrison/native | done | 9000/9000/1000 | 10,000 | 35.287 | 594.400 | 5,949 | ~53,541 | 5.354 | 48.187k | 48.187k | 12.335 | 12.335 |
| [`stability_morrison_g8_eta3200_alpha076950_2s_seed20260724_a`](brunel/runs/stability_morrison_g8_eta3200_alpha076950_2s_seed20260724_a/results.json) | morrison/native | cut@1.5s | 9000/9000/1000 | 15,000 | 59.274 | 1850.759 | 34,689 | ~312,201 | 20.813 | 187.321k | 187.321k | 9.880 | 9.880 |
| [`stability_morrison_g8_eta3200_alpha102600_10s_seed20260724_a`](brunel/runs/stability_morrison_g8_eta3200_alpha102600_10s_seed20260724_a/results.json) | morrison/native | done | 9000/9000/1000 | 100,000 | 121.265 | 621.834 | 58,587 | ~527,283 | 5.273 | 47.455k | 47.455k | 13.104 | 13.104 |
| [`stability_morrison_g8_eta3200_alpha102600_2s_seed20260724_a`](brunel/runs/stability_morrison_g8_eta3200_alpha102600_2s_seed20260724_a/results.json) | morrison/native | done | 9000/9000/1000 | 20,000 | 44.665 | 597.735 | 11,876 | ~106,884 | 5.344 | 48.098k | 48.098k | 12.427 | 12.427 |
| [`stability_morrison_g8_eta3200_alpha102600_2s_seed20260725_a`](brunel/runs/stability_morrison_g8_eta3200_alpha102600_2s_seed20260725_a/results.json) | morrison/native | done | 9000/9000/1000 | 20,000 | 69.683 | 756.646 | 11,131 | ~100,179 | 5.009 | 45.081k | 45.081k | 16.784 | 16.784 |
| [`stability_morrison_g8_eta3200_alpha102600_2s_seed20260726_a`](brunel/runs/stability_morrison_g8_eta3200_alpha102600_2s_seed20260726_a/results.json) | morrison/native | done | 9000/9000/1000 | 20,000 | 69.984 | 818.300 | 12,125 | ~109,125 | 5.456 | 49.106k | 49.106k | 16.664 | 16.664 |
| [`stability_static_g55_eta1850_1s_seed20260724_a`](brunel/runs/stability_static_g55_eta1850_1s_seed20260724_a/results.json) | static/native | done | 9000/9000/1000 | 10,000 | 34.385 | 232.290 | 8,115 | ~73,035 | 7.303 | 65.731k | - | 3.534 | - |
| [`stability_static_g5_eta1685_1s_seed20260724_a`](brunel/runs/stability_static_g5_eta1685_1s_seed20260724_a/results.json) | static/native | done | 9000/9000/1000 | 10,000 | 35.644 | 271.014 | 9,927 | ~89,343 | 8.934 | 80.409k | - | 3.370 | - |
| [`stability_static_g6_eta1685_1s_seed20260724_a`](brunel/runs/stability_static_g6_eta1685_1s_seed20260724_a/results.json) | static/native | done | 9000/9000/1000 | 10,000 | 34.090 | 186.709 | 5,785 | ~52,065 | 5.207 | 46.858k | - | 3.985 | - |
| [`stability_static_g6_eta1850_1s_seed20260724_a`](brunel/runs/stability_static_g6_eta1850_1s_seed20260724_a/results.json) | static/native | done | 9000/9000/1000 | 10,000 | 33.975 | 193.433 | 5,670 | ~51,030 | 5.103 | 45.927k | - | 4.212 | - |
| [`stability_static_g6_eta2100_1s_seed20260724_a`](brunel/runs/stability_static_g6_eta2100_1s_seed20260724_a/results.json) | static/native | done | 9000/9000/1000 | 10,000 | 34.565 | 220.524 | 7,038 | ~63,342 | 6.334 | 57.008k | - | 3.868 | - |
| [`stability_static_g8_eta3200_1s_seed20260724_a`](brunel/runs/stability_static_g8_eta3200_1s_seed20260724_a/results.json) | static/native | done | 9000/9000/1000 | 10,000 | 34.732 | 214.532 | 6,014 | ~54,126 | 5.413 | 48.713k | - | 4.404 | - |
| [`stability_static_g8_eta4000_1s_seed20260724_a`](brunel/runs/stability_static_g8_eta4000_1s_seed20260724_a/results.json) | static/native | done | 9000/9000/1000 | 10,000 | 35.427 | 303.207 | 10,319 | ~92,871 | 9.287 | 83.584k | - | 3.628 | - |
| [`tune_morrison_ns002_is1_2s_alpha006_20260724_a`](brunel/runs/tune_morrison_ns002_is1_2s_alpha006_20260724_a/results.json) | morrison/native | done | 180/9000/180 | 20,000 | 0.761 | 3.686 | 24 | 24 | 0.001 | 10.800 | 10.800 | 341.269 | 341.269 |
| [`tune_morrison_ns002_is1_2s_alpha008_20260724_a`](brunel/runs/tune_morrison_ns002_is1_2s_alpha008_20260724_a/results.json) | morrison/native | done | 180/9000/180 | 20,000 | 0.735 | 3.205 | 24 | 24 | 0.001 | 10.800 | 10.800 | 296.756 | 296.756 |
| [`tune_morrison_ns002_is1_2s_alpha010_20260724_a`](brunel/runs/tune_morrison_ns002_is1_2s_alpha010_20260724_a/results.json) | morrison/native | done | 180/9000/180 | 20,000 | 0.753 | 3.402 | 24 | 24 | 0.001 | 10.800 | 10.800 | 314.964 | 314.964 |
| [`tune_morrison_ns002_is1_2s_alpha015_20260724_a`](brunel/runs/tune_morrison_ns002_is1_2s_alpha015_20260724_a/results.json) | morrison/native | done | 180/9000/180 | 20,000 | 0.769 | 3.505 | 24 | 24 | 0.001 | 10.800 | 10.800 | 324.574 | 324.574 |

### NEST-GPU

| Run | Rule/timing | Status | `N_E/C_E/N_rec` | Steps | E2E s | us/step | Recorded E spikes | All-E spikes | E spikes/step | E-E pre/step | E-E post/step | ns/pre | ns/post |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| [`port_nestgpu_additive_cuda_smoke_20260725_d`](brunel/runs/port_nestgpu_additive_cuda_smoke_20260725_d/results.json) | additive/nearest | done | 18/18/18 | 10 | 0.059 | 220.335 | 0 | 0 | 0.000 | 0.000 | 0.000 | - | - |
| [`port_nestgpu_additive_scale1_10s_20260725_a`](brunel/runs/port_nestgpu_additive_scale1_10s_20260725_a/results.json) | additive/nearest | done | 9000/9000/1000 | 100,000 | 44.679 | 434.593 | 29,287 | ~263,583 | 2.636 | 23.722k | 23.722k | 18.320 | 18.320 |
| [`port_nestgpu_additive_scale1_250ms_20260725_b`](brunel/runs/port_nestgpu_additive_scale1_250ms_20260725_b/results.json) | additive/nearest | done | 9000/9000/1000 | 2,500 | 2.112 | 550.915 | 0 | ~0 | 0.000 | 0.000 | 0.000 | - | - |
| [`port_nestgpu_additive_scale1_250ms_20260725_c`](brunel/runs/port_nestgpu_additive_scale1_250ms_20260725_c/results.json) | additive/nearest | done | 9000/9000/1000 | 2,500 | 2.131 | 553.271 | 1,943 | ~17,487 | 6.995 | 62.953k | 62.953k | 8.789 | 8.789 |
| [`port_nestgpu_morrison_scale1_2s_20260725_a`](brunel/runs/port_nestgpu_morrison_scale1_2s_20260725_a/results.json) | morrison/nearest | done | 9000/9000/1000 | 20,000 | 7.687 | 324.674 | 10,459 | ~94,131 | 4.707 | 42.359k | 42.359k | 7.665 | 7.665 |

## Incomplete Brunel directories

These directories have manifests but no `results.json`; they are excluded from
all metrics because a build directory or partial binary is not a completed run:

- `port_brian2_additive_smoke_20260725_a`
- `port_brian2cuda_additive_codegen_20260725_a`
- `port_brian2cuda_additive_nesttiming_nestboundary_scale1_2s_seed20260724_a`
- `port_brian2cuda_additive_smoke_20260725_a`
- `port_brian2cuda_additive_smoke_20260725_b`
- `port_brian2cuda_morrison_codegen_20260725_a`
- `port_genn_additive_smoke_20260725_a`
- `port_genn_additive_smoke_20260725_b`
- `port_genn_additive_smoke_20260725_d`
- `port_nestgpu_additive_cuda_smoke_20260725_a`
- `port_nestgpu_additive_cuda_smoke_20260725_b`
- `port_nestgpu_additive_cuda_smoke_20260725_c`
- `port_nestgpu_additive_scale1_250ms_20260725_a`
- `port_nestgpu_additive_validation_20260725_a`
- `port_nestgpu_morrison_validation_20260725_a`
- `smoke_brian2_morrison_nestexclude_20260725_a`
- `smoke_brian2cuda_additive_nestboundary_ns001_is001_10ms_20260726_a`
- `smoke_brian2cuda_additive_nestboundary_ns001_is001_10ms_20260726_b`
- `smoke_brian2cuda_additive_nestboundary_ns001_is001_10ms_20260726_c`
- `smoke_brian2cuda_additive_nestboundary_ns001_is001_10ms_20260726_d`

`smoke_additive_20260724_a` contains neither manifest nor result and is also
excluded.
