# GeNN CPU single-network latency, 2026-09-10

All 23 cases completed with one network, one simulation thread, and 5 timed repetitions per case. The Ryzen 9 7950X guest affinity was [9]. Other host jobs continued running; this is a contended-host measurement, not an isolated peak.

**Correction:** the earlier 32-replica aggregate figures measure independent-network throughput. They do not measure timestep latency and cannot support a CPU speedup over a single GPU network. This report uses only the elapsed timer of one network divided by that network's executed steps. Repetitions run sequentially. No division by a replica or core count is performed.

## GeNN parallelization assessment

The installed GeNN 5.4 CPU backend executes one simulation thread. The upstream experimental ISPC backend and PR #710 were also inspected: their generated foreach loops use SIMD, but no task launch, OpenMP region or CPU thread pool distributes this workload across cores. No usable multicore backend was enabled. We used the requested single-core fallback. This was a backend capability investigation, not a completed implementation of a new parallel backend.

A new backend would need explicit ownership of shared RNG state, spike lists and synaptic current accumulators, plus barriers that preserve synapse/neuron and STDP ordering. Its complete timestep timer would have to include those barriers. Merely enabling a compiler OpenMP flag does not provide this.

[ISPC development report](https://genn-team.github.io/posts/developing-an-ispc-backend-for-genn-bridging-gpu-and-cpu-performance-for-neural-network-simulations.html); [open ISPC pull request](https://github.com/genn-team/genn/pull/710). Audited upstream ISPC revision: 8aef8f129b5847631cd104b40bc85786efd66b93; PR head: a70e7382a777dcde40352ebee24b5b3cca90b270. Source copies and hashes: copilot/tmp/genn_cpu_parallel_audit_20260910_a/.

## Elapsed microseconds per timestep

CPU values are the median and full range of five sequential runs. GPU columns are the existing FP32 single-network baseline. CPU and GPU use the same model definitions and input artifacts but different random generators; diagnostic activity is retained separately and is not bitwise identical.

| Case | CPU, 1 thread | CPU range | RTX 3090 | A100 | A800 | H800 |
|---|---:|---:|---:|---:|---:|---:|
| brunel_additive | 383.838 | 383.092–385.613 | 40.831 | 45.234 | 40.771 | 32.717 |
| brunel_morrison | 535.369 | 531.168–539.355 | 41.247 | 53.687 | 52.253 | 42.238 |
| mnist_1trace_bernoulli_0125 | 9.879 | 9.771–11.448 | 24.109 | 14.731 | 14.016 | 10.991 |
| mnist_1trace_bernoulli_0250 | 10.305 | 9.912–11.179 | 24.710 | 15.290 | 14.621 | 11.425 |
| mnist_1trace_bernoulli_0500 | 11.007 | 10.595–11.774 | 25.600 | 16.718 | 15.808 | 12.625 |
| mnist_1trace_dense | 13.678 | 12.532–14.866 | 26.473 | 18.848 | 16.680 | 14.343 |
| mnist_1trace_fixed-fanout_0125 | 10.056 | 9.463–10.242 | 24.284 | 14.702 | 13.990 | 11.007 |
| mnist_1trace_fixed-fanout_0250 | 10.295 | 9.730–10.520 | 24.873 | 15.308 | 14.533 | 11.473 |
| mnist_1trace_fixed-fanout_0500 | 10.559 | 10.399–11.356 | 25.673 | 16.682 | 15.908 | 12.674 |
| mnist_2trace_bernoulli_0125 | 11.899 | 11.421–12.404 | 25.082 | 15.312 | 14.539 | 11.852 |
| mnist_2trace_bernoulli_0250 | 13.890 | 12.984–13.932 | 25.430 | 16.504 | 15.753 | 12.820 |
| mnist_2trace_bernoulli_0500 | 18.061 | 17.461–18.609 | 26.698 | 18.829 | 17.802 | 14.424 |
| mnist_2trace_dense | 29.496 | 28.910–30.739 | 27.300 | 19.598 | 17.627 | 14.370 |
| mnist_2trace_fixed-fanout_0125 | 12.051 | 11.336–12.515 | 25.112 | 15.278 | 14.466 | 11.812 |
| mnist_2trace_fixed-fanout_0250 | 14.551 | 13.586–14.617 | 25.686 | 16.586 | 15.584 | 12.816 |
| mnist_2trace_fixed-fanout_0500 | 19.092 | 18.171–19.427 | 27.257 | 19.117 | 17.751 | 14.541 |
| mnist_3trace_bernoulli_0125 | 10.526 | 10.016–10.793 | 30.326 | 18.017 | 17.149 | 13.766 |
| mnist_3trace_bernoulli_0250 | 11.568 | 10.853–11.730 | 31.064 | 18.934 | 17.947 | 14.588 |
| mnist_3trace_bernoulli_0500 | 14.696 | 13.837–14.765 | 32.143 | 20.807 | 19.662 | 16.146 |
| mnist_3trace_dense | 19.914 | 18.600–21.180 | 32.674 | 21.770 | 20.212 | 16.835 |
| mnist_3trace_fixed-fanout_0125 | 10.883 | 10.105–11.047 | 30.405 | 17.990 | 17.179 | 13.752 |
| mnist_3trace_fixed-fanout_0250 | 11.285 | 10.934–11.818 | 31.054 | 18.961 | 17.953 | 14.616 |
| mnist_3trace_fixed-fanout_0500 | 12.714 | 12.475–13.225 | 32.109 | 20.812 | 19.648 | 16.302 |

## NEST: one network with real internal threading

These earlier runs already measure single-network latency, including NEST synchronization. They use the custom current-default Brunel port. The NEST engine uses -O3; its initial Nix build stripped -march=native, while the custom module retained it. Neuron/plastic state is FP32 and event-buffer sums are FP64. Native NEST RNG and connectivity differ from GeNN and depend on thread count. Background contention changed during the matrix, so these are not clean SMT scaling results.

| Rule | Threads | Median µs/step | Range |
|---|---:|---:|---:|
| additive | 16 | 765.493 | 697.181–1039.361 |
| morrison | 16 | 1349.767 | 1290.546–1412.230 |
| additive | 32 | 9721.236 | 9270.818–9918.504 |
| morrison | 32 | 10226.252 | 9998.889–10410.727 |

## Protocol and validation

GeNN/PyGeNN 5.4.0, GCC optimized with -O3 -march=native and NIX_ENFORCE_NO_NATIVE=0; no optional GeNN fast-math. Simulation state is FP32 and host normalization FP64. BLAS and OpenMP environment thread counts are one. Affinity pins the process to one guest logical CPU; the host SMT sibling is not reserved or disabled.

Real MNIST uses data/mnist and the frozen baseline_inputs_20260907_v2 bundle. Each repetition restores the 10,000-image weights/theta checkpoint with other runtime state reset, then trains 100 accepted images from index 10000, seed 20260724. This is branched continuation. The timed region includes normalization, the host control loop, all retries, 700 stimulus ticks and 300 rest ticks per attempt. The denominator includes every executed tick. No accuracy or convergence comparison is made.

Brunel uses 9000 E + 2250 I neurons, fixed indegrees 450/112 with replacement and without recurrent autapses, 6,322,500 connections, dt=0.1 ms, next-tick delivery, arrival-timed causal-boundary STDP, sqrt(20) delivery scaling and external-rate scales 0.47/0.32. The timer covers 1000 ms / 10000 steps after 100 ms presimulation.

Construction, compilation, loading, presimulation and result serialization are excluded, matching the GPU baseline boundaries. Parent process startup/rendezvous is excluded; this is harness setup, not simulation synchronization. The GeNN CPU step function completes synchronously, so no unmeasured device work remains after its loop returns.

Every timing repetition exactly reproduced its CPU diagnostic step count, total spikes and event counters. Diagnostics guard finite state, firing, retries, weights, thresholds and normalization. The report generator rejects replica runs, incomplete repetitions, unpinned workers and mismatched time/step denominators. The deterministic NEST checks cover current integration, refractory behavior, next-step delivery and simultaneous STDP events.

Power estimation is deferred. This WSL2 container exposes no readable package energy counter; the raw energy records are null. No per-watt claim or power estimate is made.

## Artifacts

- Complete raw run, source copies, command and hashes: `copilot/tmp/cpu_genn_latency_20260910_a`
- Input manifest SHA-256: `e94363abf5e97f786c7217c509ce79ce96e395329ff2d5dff0b1c26639d57e80`
- GPU baseline: `genn-sweep/baseline-20260907-matrix.json`
- [Full latency table](timings.csv)
- [Activity and state diagnostics](diagnostics.csv)
- [Machine-readable report and provenance](summary.json)
- [NEST timings](nest-timings.csv)
- [Single-network hardware cycles and work accounting](cycles.md)

Validation run: 48 tests passed (baseline, reimplementation, Brunel, energy, NEST mechanics and latency-report checks). An independent artifact audit verified all 115 timings, CPU affinity, exact diagnostic counters and source hashes.
