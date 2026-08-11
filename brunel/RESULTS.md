# Brunel STDP implementation and run report

Run dates: 2026-07-24 to 2026-07-26 UTC. Some artifact tags use the host-local
date `20260725` (UTC+08:00).

## Protocol

The paired desktop runs used NEST `3.9.0-post0.dev14` from vendored revision
`182eba4`, eight local threads, seed `20260724`, and the workload in
`nest_brunel_stdp.py` (SHA-256
`4fe5676602dbe62ef7e6a707021ffcca50182cf19f3245c00353b90f1b1ce27c`).
The graph has 9,000 excitatory and 2,250 inhibitory neurons, fixed indegrees
of 9,000 excitatory and 2,250 inhibitory inputs, 81,000,000 plastic E-to-E
synapses, 0.1 ms resolution, and 1.5 ms delays. This matches the documented
scale-1 desktop workload. The only intended paired-case change is the E-to-E
plasticity rule and its rule-specific parameters.

The subsequent stability study used the same NEST revision, topology,
resolution, delay, thread count, and initial seed with an extended runner
(SHA-256
`7fe2589d02addcf1cfd8cb6689351f2e68a9b3c2102b8e1884b2d58d49872305`).
It added a static E-to-E control, independent inhibitory-weight ratio `g` and
external-drive `eta`, and a 100 Hz per-100-ms-block early-stop condition.
Those runs used a 100 ms presimulation and 100 ms reporting chunks. Two final
seed checks used seeds `20260725` and `20260726`.

The host was also running two single-threaded Brian 1 training jobs and one
unrelated single-threaded simulator. Timings are operational wall-clock
measurements under that load, not isolated peak-performance results.

## Desktop workload results

| Rule and duration | Measured steps | `Run` wall | Wall/step | Mean rate | Final E-E weights | Boundary mass |
|---|---:|---:|---:|---:|---:|---:|
| Morrison, 250 ms | 2,500 | 5.1804 s | 2.07216 ms | 27.536 Hz | 46.570 +/- 0.672 pA | 0% |
| Additive, 250 ms | 2,500 | 1.6017 s | 0.640683 ms | 7.756 Hz | 45.488 +/- 0.534 pA | 0% |
| Additive, 10 s | 100,000 | 337.0657 s | 3.37066 ms | 41.645 Hz | 56.989 +/- 38.922 pA | 79.182% |

The 50 ms presimulation adds 500 steps. Including it, the Morrison desktop
run took 5.4026 s for 3,000 steps (1.80086 ms/step), the short additive run
took 1.9315 s (0.643830 ms/step), and the long additive run took 337.3864 s
for 100,500 steps (3.35708 ms/step). Construction took 23.42 s, 23.30 s, and
25.64 s respectively. The complete process wall times, including graph
construction, sampling, and result serialization, were 31.94 s, 28.24 s, and
372.46 s.

## Behavior assessment

The two rules recover their distinct weight-selection mechanisms:

- Morrison remained a single interior mode. At 250 ms its sampled weights
  occupied one histogram peak, with no mass near a boundary.
- Additive STDP initially broadened slowly, then separated sharply after about
  5.5 s. At 10 s, 27.669% of sampled weights were below `0.1 Wmax` and 51.513%
  were above `0.9 Wmax`; the histogram has dominant peaks at zero and `Wmax`.

This is not evidence that either scale-1 run reached the intended balanced
equilibrium. Morrison's rate rose from 7.46 Hz in the first measured 50 ms to
43.26 Hz in the last 50 ms, while its weight mean and variance were still
moving. The additive run became highly synchronous as its boundary modes
formed: its final 500 ms rate was 82.382 Hz, population-count Fano factor was
1,543, and mean single-neuron ISI CV was 2.48. The increased spike traffic also
explains why the long additive wall time per step is much greater than the
short-run value.

NEST's `scale` parameter multiplies population size; it is not average fan-out.
The excitatory and inhibitory indegrees remain 9,000 and 2,250 as scale grows,
so connection probability and shared-input correlations decrease. The NEST
example warns that scale 1 can synchronize and says meaningful validation
requires scale greater than 10 and a final rate below 10 spikes/s. This is a
documented lower guideline, not an experimentally sharp minimum.
Morrison et al.'s target is a low-rate asynchronous-irregular state coupled to
an approximately Gaussian, interior distribution near `45.65 +/- 3.99 pA`.
That complete equilibrium was not recovered here. A scale greater than 10
would require over 1.26 billion recurrent connections; integer scale 11 has
about 1.392 billion. This does not fit the container's 30 GiB memory budget
with the present build.

Earlier reduced-population diagnostics are retained under `brunel/runs/`, but
are not equilibrium controls. Reducing indegree removes recurrent balance;
keeping full indegree with a tiny population creates many multapses and strong
seed/thread-dependent correlations.

## Scale-1 stability tuning

Static E-to-E controls showed that inhibition and external drive can tune the
mean rate, but cannot recover the large-network correlation structure at
scale 1:

| `g` | `eta` | Mean rate | Maximum 100 ms rate | Fano, 3 ms | Mean ISI CV |
|---:|---:|---:|---:|---:|---:|
| 5.0 | 1.685 | 9.927 Hz | 16.93 Hz | 90.92 | 0.706 |
| 6.0 | 1.850 | 5.670 Hz | 6.86 Hz | 51.24 | 0.651 |
| 8.0 | 3.200 | 6.014 Hz | 6.78 Hz | 50.43 | 0.671 |
| 8.0 | 4.000 | 10.319 Hz | 17.69 Hz | 187.13 | 0.744 |

The stock Morrison depression coefficient is `alpha = 0.0513`. Increasing it
without changing the operating point did not prevent runaway. Jointly
increasing inhibition, drive, and depression produced a stable short-run
candidate:

| `g` | `eta` | `alpha` | Requested / actual | Result | Mean rate | Maximum block rate | Final weight mean +/- std | Fano |
|---:|---:|---:|---:|:---|---:|---:|---:|---:|
| 5 | 1.685 | 0.053865 | 1.0 / 1.0 s | runaway | 267.974 Hz | 936.47 Hz | 88.095 +/- 83.796 pA | 2997.83 |
| 5 | 1.685 | 0.061560 | 1.0 / 0.6 s | stopped | 134.615 Hz | 722.07 Hz | 56.285 +/- 43.340 pA | 3280.31 |
| 8 | 3.200 | 0.061560 | 10.0 / 2.2 s | stopped | 46.819 Hz | 688.55 Hz | 62.600 +/- 38.322 pA | 2655.10 |
| 8 | 3.200 | 0.076950 | 2.0 / 1.5 s | stopped | 23.126 Hz | 120.08 Hz | 64.743 +/- 1.615 pA | 1639.11 |
| 8 | 3.200 | 0.102600 | 10.0 / 10.0 s | completed | 5.859 Hz | 8.58 Hz | 45.519 +/- 1.260 pA | 50.39 |

The `g = 8`, `eta = 3.2`, `alpha = 0.06156` case looked controlled at 1 s
(5.949 Hz) but ran away after 2 s. A quiet 250 ms or 1 s pilot therefore does
not establish stability. For this seed and operating point, 1.5 times the
stock depression failed while twice the stock value survived 10 s. This
brackets a practical short-run threshold, not a universal critical value.

### Candidate seed check and timing

| Seed | Duration | Mean rate | Maximum block rate | Fano | Weight mean +/- std | Simulation wall/step |
|---:|---:|---:|---:|---:|---:|---:|
| 20260724 | 2 s | 5.938 Hz | 8.10 Hz | 56.92 | 45.562 +/- 0.585 pA | 0.598 ms |
| 20260725 | 2 s | 5.566 Hz | 6.72 Hz | 38.89 | 45.502 +/- 0.553 pA | 0.757 ms |
| 20260726 | 2 s | 6.063 Hz | 9.91 Hz | 64.33 | 45.587 +/- 0.597 pA | 0.818 ms |
| 20260724 | 10 s | 5.859 Hz | 8.58 Hz | 50.39 | 45.519 +/- 1.260 pA | 0.622 ms |

All candidate samples remained a single interior histogram mode with zero
sampled boundary mass. The 10 s NEST simulation calls took 62.183 s for
100,000 measured steps. Including the presimulation, all `Run` calls took
62.563 s for 101,000 steps; construction took 23.556 s and complete process
wall time was 121.265 s.

This tuning prevents immediate scale-1 runaway, but it changes the benchmark
operating point: inhibition is 60% stronger, external drive is about 90%
stronger, and depression is doubled. It also does not recover Morrison's
large-network state. The candidate's Fano factor remains around 50 rather than
8.5, and its weight variance was still growing at 10 s. Establishing a
stationary equilibrium would require a substantially longer run, several
seeds, and plateaus in rate, Fano factor, weight mean, and weight variance.

## Commands and artifacts

```sh
bash brunel/run_with_nest_env.sh --rule morrison \
  --output brunel/runs/desktop_morrison_scale1_250ms_20260724_a \
  --threads 8 --seed 20260724 --network-scale 1 --indegree-scale 1 \
  --record-neurons 1000 --weight-sample-size 100000 \
  --presim-ms 50 --sim-ms 250 --chunk-ms 50

bash brunel/run_with_nest_env.sh --rule additive \
  --output brunel/runs/desktop_additive_scale1_250ms_20260724_a \
  --threads 8 --seed 20260724 --network-scale 1 --indegree-scale 1 \
  --record-neurons 1000 --weight-sample-size 100000 \
  --presim-ms 50 --sim-ms 250 --chunk-ms 50

bash brunel/run_with_nest_env.sh --rule additive \
  --output brunel/runs/equilibrium_additive_scale1_10s_20260724_a \
  --threads 8 --seed 20260724 --network-scale 1 --indegree-scale 1 \
  --record-neurons 1000 --weight-sample-size 100000 \
  --presim-ms 50 --sim-ms 10000 --chunk-ms 500

bash brunel/run_with_nest_env.sh --rule morrison \
  --output brunel/runs/stability_morrison_g8_eta3200_alpha102600_10s_seed20260724_a \
  --threads 8 --seed 20260724 --network-scale 1 --indegree-scale 1 \
  --inhibitory-weight-ratio 8 --external-drive-eta 3.2 \
  --morrison-alpha 0.1026 --abort-rate-hz 100 \
  --record-neurons 1000 --weight-sample-size 100000 \
  --presim-ms 100 --sim-ms 10000 --chunk-ms 100
```

Each run directory contains its manifest, complete periodic and final JSON,
and initial/final sampled weights. Matching stdout logs are in `brunel/logs/`.

The complete static-control and plastic-sweep tables, termination details, and
artifact naming are retained in `STABILITY_TUNING.md`.

## Brian2, GeNN, and NEST-GPU ports

### Implementation

The ports share the measured scale-1 model constants in `ports/common.py`:
0.1 ms integration steps, 1.5 ms recurrent delay, 9,000 excitatory and 2,250
inhibitory neurons, fixed indegrees of 9,000 and 2,250, and 126,562,500
recurrent synapses. Of these, 81,000,000 E-to-E synapses are plastic. The
measured NEST configurations use `tau_plus = 20 ms` (the NEST model default,
not the canonical example's 15 ms) and `tau_minus = 30 ms`.

The common presets are:

| Rule | `g` | `eta` | `lambda` | `alpha` | `mu_plus` | `mu_minus` | Bound |
|---|---:|---:|---:|---:|---:|---:|---:|
| Additive | 5 | 1.685 | 0.01 | 1.05 | 0 | 0 | `Wmax = 2 JE` |
| Tuned Morrison | 8 | 3.2 | 0.1 | 0.1026 | 0.4 | 1 | none |

`ports/brian2_port.py` uses Brian2 event-driven per-synapse pre/post traces,
fixed-indegree connectivity with replacement, and the exact NEST
`IAFPropagatorAlpha` state transition. `run_brian2.py` is the CPU runner and
`run_brian2cuda.py` compiles the same model with Brian2CUDA standalone.

`ports/genn_port.py` defines a custom GeNN neuron with the same exact alpha-PSC
transition and a custom all-to-all-trace plastic synapse. The same model
definition was run with both `single_threaded_cpu` and `cuda` backends.

The runs used Python 3.13.12, vendored Brian2 revision `1bfa1a9`,
Brian2CUDA revision `825c0c5`, and GeNN revision `563c45c` (PyGeNN 5.4.0).
The final implementation hashes are `15e7ed8b` for `ports/common.py`,
`94e93131` for `ports/brian2_port.py`, `f613fa73` for
`ports/genn_port.py`, `18d828ec` for `ports/nestgpu_port.py`, and `eab7e463`
for `run_brian2cuda.py`. Runs used the seed recorded in each manifest, but
framework-specific RNGs and graph generators mean equal numeric seeds are not
a shared random realization.

Both ports sample external Poisson multiplicity directly at each target. The
stationary external input has no explicit 1.5 ms delay, unlike recurrent
connections. A constant delay only shifts a stationary Poisson stream after
startup, and all measurements follow 50 or 100 ms of presimulation.

`ports/nestgpu_port.py` is an executable NEST-GPU driver. The vendored core at
revision `830b15b` contains `src/stdp_pl.{h,cu}`, which implements the Morrison
weight dependence. It was built against CUDA 12.8 and run on the RTX 3090.

NEST-GPU has two material semantic limitations:

- its plastic callback supplies only the nearest signed pre/post interval and
  the weight, so it cannot implement NEST's accumulated all-to-all traces
  without changing connection storage; and
- its fixed-indegree Python API cannot disable recurrent autapses.

The NEST-GPU Morrison callback is therefore explicitly a nearest-pair
approximation:

```text
Delta t >= 0: w <- w + lambda * w^mu * exp(-Delta t / tau_plus)
Delta t <  0: w <- max(0, w - lambda * alpha * w * exp(Delta t / tau_minus))
```

These limitations are written into every NEST-GPU manifest. Its measured
results below must not be treated as equivalent to the Brian2, GeNN, or NEST
all-to-all rules until the callback/storage issue is resolved.

### Full-scale pilot comparison

The original short additive cases used 50 ms presimulation and 250 ms
measurement. The tuned Morrison ports used 100 ms presimulation and 2 s
measurement; the NEST row is the matching seed-`20260724` candidate check.
These Brian2 and GeNN rows used the original `arrival` learning timing. Rates
and Fano factors use the first 1,000 excitatory neurons. Timings cover
simulation calls only.

| Rule / framework | Duration | Rate | Fano, 3 ms | Final E-E weights | Modes | Wall/step |
|---|---:|---:|---:|---:|---:|---:|
| Additive / NEST | 250 ms | 7.756 Hz | 71.28 | 45.488 +/- 0.534 pA | 1 | 0.641 ms |
| Additive / Brian2 | 250 ms | 5.360 Hz | 39.08 | 45.348 +/- 0.557 pA | 1 | 12.619 ms |
| Additive / GeNN CPU | 250 ms | 7.160 Hz | 79.70 | 45.307 +/- 0.622 pA | 1 | 17.308 ms |
| Morrison / NEST | 2 s | 5.938 Hz | 56.92 | 45.562 +/- 0.585 pA | 1 | 0.598 ms |
| Morrison / Brian2 | 2 s | 5.148 Hz | 46.62 | 45.082 +/- 0.562 pA | 1 | 7.919 ms |
| Morrison / GeNN CPU | 2 s | 5.459 Hz | 68.17 | 44.879 +/- 0.613 pA | 1 | 14.583 ms |

The ports recover the tuned Morrison behavior at 2 s: low rates, no boundary
mass, and one interior weight mode. Their weight means drift down more than
NEST's, so this is qualitative agreement rather than numerical identity. A
longer multi-seed check is still needed before claiming a stationary match.

### Original additive 10 s comparison (`arrival` timing)

| Framework | Rate | Fano, 3 ms | Final E-E weights | Boundary mass | Modes | Simulation wall | Wall/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| NEST | 41.645 Hz | 1,543 | 56.989 +/- 38.922 pA | 79.182% | boundary-dominated | 337.066 s | 3.371 ms |
| Brian2 | 2.857 Hz | 29.27 | 43.712 +/- 1.524 pA | 0% | 1 | 519.331 s | 5.193 ms |
| GeNN CPU | 2.951 Hz | 35.39 | 43.582 +/- 1.568 pA | 0% | 1 | 878.412 s | 8.784 ms |

Neither original port reproduced the NEST additive run's transition near 5.5 s
to synchronous firing and a boundary-dominated distribution. Instead, the two
independent ports agreed closely: firing gradually declined and the weights
remained a broadened interior mode. The CUDA timing audit below identifies a
systematic pair-timing mismatch in these original runs and demonstrates the
transition after correcting it.

The timing comparison is also not a controlled framework benchmark. NEST used
eight threads, GeNN used its single-threaded CPU backend, and Brian2 used Cython
runtime code. Other unrelated CPU and memory-heavy jobs remained active, as
requested.

### CUDA runtime validation and event-boundary audit

CUDA 12.8 kernels were compiled and executed on an NVIDIA GeForce RTX 3090
(24 GiB, compute capability 8.6; driver 596.49). Brian2CUDA, GeNN CUDA, and the
vendored NEST-GPU runtime all executed the full scale-1 Brunel graph. The graph
contains 126,562,500 recurrent synapses, including 81,000,000 plastic E-to-E
synapses.

The audit found two independent event-time details. First, NEST evaluates
postsynaptic history for a presynaptic source spike at `t_pre - d`, then uses
the dendritic delay `d` again in the causal pair interval. Brian2 and GeNN
originally learned at presynaptic arrival `t_pre + d` but used undelayed
postsynaptic events, shifting every pair by `2d = 3 ms`. The
`nest_dendritic` mode retains the 1.5 ms transmission delay and delays the
postsynaptic learning path by 3 ms.

Second, an event exactly on that boundary is asymmetric in NEST. The history
interval is `(t_last_pre - d, t_pre - d]`, so the boundary post contributes to
potentiation. `get_K_value(t_pre - d)` searches for a post strictly before the
boundary, so the same post does not contribute to depression. The faithful
`nest_causal_boundary` mode therefore computes LTP from the old presynaptic
trace and LTD from the old postsynaptic trace, then retains both trace
increments. This is neither ordinary pre-first nor ordinary post-first event
ordering. The source locations are
`3rdparty/nest-simulator/models/stdp_pl_synapse_hom.h:269-287`,
`3rdparty/nest-simulator/nestkernel/archiving_node.cpp:96-112`,
`ports/brian2_port.py`, and `ports/genn_port.py`.

The earlier paired GeNN CUDA additive control isolated only the 3 ms timing
change and used framework-pre-first ties:

| Timing | Requested / actual | Rate | Final E-E weights | Boundary mass | Wall/step |
|---|---:|---:|---:|---:|---:|
| `arrival` | 10 / 10 s | 2.846 Hz | 43.707 +/- 1.510 pA | 0% | 0.103 ms |
| `nest_dendritic` | 10 / 2 s, stopped | 34.565 Hz | 63.136 +/- 28.933 pA | 49.851% | 0.538 ms |

The corrected run crossed the 100 Hz block cutoff during 1.5-2.0 s. A matching
Brian2CUDA diagnostic run was already synchronous in its first 500 ms and ended
at 195.056 Hz with 90.198% boundary mass after 2 s. These demonstrate that the
3 ms shift is behaviorally important, but they are not exact NEST-boundary
comparisons.

Faithful-boundary full-scale results are:

| Rule / framework | Requested / actual | Rate | Final E-E weights | Boundary mass | Wall/step | Outcome |
|---|---:|---:|---:|---:|---:|:---|
| Additive / NEST | 10 / 10 s | 41.645 Hz | 56.989 +/- 38.922 pA | 79.182% | 3.371 ms | high/bimodal branch |
| Additive / GeNN CUDA | 10 / 1.7 s | 26.009 Hz | 58.241 +/- 34.286 pA | 55.514% | 0.539 ms | 257.97 Hz block cutoff |
| Additive / Brian2CUDA | 2 / incomplete | unavailable | unavailable | unavailable | unavailable | host OOM after compile |
| Morrison / NEST | 10 / 10 s | 5.859 Hz | 45.519 +/- 1.260 pA | 0% | 0.622 ms | controlled, unimodal |
| Morrison / Brian2CUDA | 4 / 4 s | 5.739 Hz | 45.569 +/- 0.800 pA | 0% | 0.675 ms | controlled, unimodal |
| Morrison / GeNN CUDA | 10 / 3.1 s | 17.110 Hz | 65.173 +/- 2.083 pA | 0% | 0.392 ms | 146.99 Hz block cutoff |

The GeNN additive run recovers the expected boundary-selected branch, but its
earlier transition is not a numerical match to NEST. The faithful full-scale
Brian2CUDA additive run generated and compiled its 81-million-synapse model,
then the host was OOM-killed before it wrote `results.json`. Its immutable
incomplete artifact is
`port_brian2cuda_additive_nesttiming_nestboundary_scale1_2s_seed20260724_a`.
It must not be quoted as a completed result.

Tie-order sensitivity is large enough to change branch selection. With GeNN
CUDA seed `20260725`, framework-pre-first, post-first, and faithful-boundary
Morrison runs hit the cutoff after 4.1, 6.7, and 3.1 s respectively, while the
zero-lag-exclusion diagnostic completed 10 s at 6.737 Hz. With Brian2CUDA seed
`20260725`, framework-pre-first became synchronous around 6.6-7.0 s,
post-first completed 10 s at 5.554 Hz, and faithful-boundary completed the
memory-limited 4 s check at 5.739 Hz. The diagnostic modes are useful
sensitivity controls, not substitutes for `nest_causal_boundary`.

Short CUDA runtime results are:

| Rule / runtime / timing | Duration | Rate | Fano, 3 ms | Final E-E weights | Boundary mass | Wall/step |
|---|---:|---:|---:|---:|---:|---:|
| Additive / GeNN CUDA / `arrival` | 250 ms | 5.628 Hz | 44.09 | 45.255 +/- 0.624 pA | 0% | 0.303 ms |
| Additive / Brian2CUDA / `arrival` | 250 ms | 6.084 Hz | 55.35 | 45.399 +/- 0.532 pA | 0% | 0.824 ms |
| Additive / NEST-GPU approximation | 250 ms | 7.772 Hz | 109.44 | 45.243 +/- 0.631 pA | 0% | 0.553 ms |
| Morrison / GeNN CUDA / `arrival` | 2 s | 5.288 Hz | 57.57 | 45.006 +/- 0.574 pA | 0% | 0.142 ms |
| Morrison / Brian2CUDA / `nest_dendritic` | 2 s | 5.751 Hz | 43.65 | 45.588 +/- 0.577 pA | 0% | 0.741 ms |
| Morrison / NEST-GPU approximation | 2 s | 5.230 Hz | 40.33 | 45.143 +/- 0.569 pA | 0% | 0.325 ms |

Wall/step covers measured simulator execution and excludes construction and
compilation. Runaway spike traffic makes corrected additive timings much
slower and prevents treating cutoff rows as steady-state throughput
measurements.

The remaining mismatch is realization-sensitive rather than a demonstrated
formula difference. Repeating a GeNN case reproduced its result exactly, while
hybrid state/graph seed checks did not isolate initial voltage or graph seed as
a single cause. Brian2CUDA can occupy either branch under different tie
diagnostics, and GeNN CPU followed the same runaway tendency as GeNN CUDA. The
frameworks do not share connectivity, Poisson streams, or initial states, and
the scale-1 graph is explicitly correlation-sensitive. A definitive numerical
cross-framework test therefore needs an exported common graph, common initial
state, and externally supplied spike trains. The NEST-GPU rows additionally
use nearest-pair plasticity and include autapses, so their agreement is not an
equivalence result.

### OOM recovery and low-memory validation

The faithful scale-1 Brian2CUDA additive attempt was not repeated after its
host OOM. Instead, both rule kernels were compiled and executed sequentially
with `network_scale = indegree_scale = 0.01`, 90 excitatory neurons, 22
inhibitory neurons, 8,100 plastic E-to-E synapses, and 10 ms of measurement.
Both used `nest_dendritic`, `nest_causal_boundary`, and the new
`--compile-jobs 1` limit:

| Rule | Rate | Final E-E weights | Boundary mass | Simulation wall/step | Compile wall |
|---|---:|---:|---:|---:|---:|
| Additive | 65.556 Hz | 45.696 +/- 0.445 pA | 0% | 0.436 ms | 321.095 s |
| Morrison | 148.889 Hz | 45.716 +/- 0.394 pA | 0% | 0.550 ms | 322.917 s |

These 10 ms reduced-graph results validate CUDA compilation, execution, both
learning formulas, the faithful tie path, timing instrumentation, and artifact
serialization. Their rates are not scale-1 behavioral evidence. Serial
compilation kept available host memory near 20 GiB throughout; Brian2's default
one-job-per-visible-CPU compilation had transiently consumed about 4 GiB more
even for this small graph. Future full-scale attempts should retain
`--compile-jobs 1` and add an explicit host-memory budget before allocating the
81-million-synapse trace arrays.

Completed recovery artifacts are
`smoke_brian2cuda_additive_nestboundary_ns001_is001_10ms_20260726_e` and
`smoke_brian2cuda_morrison_nestboundary_ns001_is001_10ms_20260726_a`.
Earlier `_a` through `_d` additive smoke directories contain failed environment
or toolkit-discovery attempts and were preserved rather than reused.

### Port commands and artifacts

```sh
bash brunel/run_with_reimpl_env.sh brunel/run_brian2.py \
  --rule additive --output brunel/runs/port_brian2_additive_scale1_10s_20260725_a \
  --network-scale 1 --indegree-scale 1 --presim-ms 50 \
  --sim-ms 10000 --chunk-ms 500

bash brunel/run_with_reimpl_env.sh brunel/run_genn.py \
  --rule additive --output brunel/runs/port_genn_additive_scale1_10s_20260725_a \
  --network-scale 1 --indegree-scale 1 --presim-ms 50 \
  --sim-ms 10000 --chunk-ms 500

bash brunel/run_with_reimpl_env.sh brunel/run_brian2.py \
  --rule morrison --output brunel/runs/port_brian2_morrison_scale1_2s_20260725_a \
  --network-scale 1 --indegree-scale 1 --presim-ms 100 \
  --sim-ms 2000 --chunk-ms 100

bash brunel/run_with_reimpl_env.sh brunel/run_genn.py \
  --rule morrison --output brunel/runs/port_genn_morrison_scale1_2s_20260725_a \
  --network-scale 1 --indegree-scale 1 --presim-ms 100 \
  --sim-ms 2000 --chunk-ms 100

python brunel/run_brian2cuda.py --rule additive \
  --stdp-timing nest_dendritic --stdp-tie-order nest_causal_boundary \
  --compile-jobs 1 --output brunel/runs/brian2cuda_additive_example

python brunel/run_genn.py --rule additive --backend cuda \
  --stdp-timing nest_dendritic --stdp-tie-order nest_causal_boundary \
  --output brunel/runs/genn_cuda_additive_example

python brunel/run_nestgpu.py --rule morrison \
  --output brunel/runs/nestgpu_morrison_example
```

Each completed runtime directory contains `manifest.json`, `results.json`, and
`weight_samples.npz`. Failed smoke directories were retained rather than
reused. The CUDA artifact tags quoted in the tables all begin with
`port_{genn_cuda,brian2cuda,nestgpu}_`; corrected controls contain
`nesttiming` and original controls contain `arrival` where needed.
