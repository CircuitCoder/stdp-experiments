# CPU measurements, initial batch 20260910

**Superseded for latency comparisons:** this is an independent-replica throughput
experiment. Its aggregate figures are not the speed of one network. Use the
[corrected single-network latency results](../results-latency-20260910/report.md).
The original measurements below are retained as historical evidence.

The 23-case GeNN CPU suite completed with 32 independent single-threaded replicas per case and 5 timed repetitions. The initial host is a Ryzen 9 7950X with 32 guest CPUs. This measures aggregate independent-network throughput; one network does not gain 32-way GeNN CPU threading. NEST rows measure one multithreaded Brunel network.

The host was contended by unrelated jobs. NEST compilation and an aborted NEST pilot also overlapped part of the GeNN run. These are initial operational measurements, not isolated peak results.

**A complete CPU energy result is unavailable:** one or more cases lack readable package energy. Unavailable measurements are null, and no TDP or assumed power is substituted. This initial WSL2 container exposes no readable package energy counter.

## GeNN results

All durations below are microseconds per simulation step. Aggregate time is the measured batch wall time divided by the sum of executed steps across replicas. The range spans all complete batches. Replica latency is the median worker latency within each batch, then the median across batches. GPU columns retain the single-network FP32 baseline and therefore use a different concurrency policy; they do not establish a saturated CPU-versus-GPU throughput ranking.

| Case | CPU aggregate | Batch range | CPU replica latency | RTX 3090 latency | A100 latency |
|---|---:|---:|---:|---:|---:|
| brunel_additive | 45.629 | 43.832–46.644 | 1242.142 | 40.831 | 45.234 |
| brunel_morrison | 86.837 | 83.176–90.923 | 2257.971 | 41.247 | 53.687 |
| mnist_1trace_bernoulli_0125 | 1.134 | 1.118–1.180 | 28.195 | 24.109 | 14.731 |
| mnist_1trace_bernoulli_0250 | 1.311 | 1.153–1.395 | 31.808 | 24.710 | 15.290 |
| mnist_1trace_bernoulli_0500 | 1.583 | 1.505–1.605 | 36.815 | 25.600 | 16.718 |
| mnist_1trace_dense | 1.963 | 1.735–2.120 | 47.805 | 26.473 | 18.848 |
| mnist_1trace_fixed-fanout_0125 | 1.055 | 1.031–1.085 | 27.524 | 24.284 | 14.702 |
| mnist_1trace_fixed-fanout_0250 | 1.065 | 1.056–1.109 | 27.223 | 24.873 | 15.308 |
| mnist_1trace_fixed-fanout_0500 | 1.239 | 1.169–1.281 | 31.586 | 25.673 | 16.682 |
| mnist_2trace_bernoulli_0125 | 1.607 | 1.454–1.656 | 37.307 | 25.082 | 15.312 |
| mnist_2trace_bernoulli_0250 | 1.738 | 1.668–1.850 | 42.635 | 25.430 | 16.504 |
| mnist_2trace_bernoulli_0500 | 2.082 | 1.805–2.712 | 51.851 | 26.698 | 18.829 |
| mnist_2trace_dense | 2.942 | 2.864–3.464 | 75.456 | 27.300 | 19.598 |
| mnist_2trace_fixed-fanout_0125 | 1.231 | 1.217–1.307 | 32.872 | 25.112 | 15.278 |
| mnist_2trace_fixed-fanout_0250 | 1.509 | 1.420–1.547 | 37.659 | 25.686 | 16.586 |
| mnist_2trace_fixed-fanout_0500 | 1.981 | 1.821–2.551 | 49.926 | 27.257 | 19.117 |
| mnist_3trace_bernoulli_0125 | 1.125 | 1.087–1.132 | 28.598 | 30.326 | 18.017 |
| mnist_3trace_bernoulli_0250 | 1.250 | 1.228–1.359 | 32.430 | 31.064 | 18.934 |
| mnist_3trace_bernoulli_0500 | 1.569 | 1.465–1.645 | 39.995 | 32.143 | 20.807 |
| mnist_3trace_dense | 2.076 | 1.955–2.243 | 55.317 | 32.674 | 21.770 |
| mnist_3trace_fixed-fanout_0125 | 1.134 | 1.085–1.197 | 28.244 | 30.405 | 17.990 |
| mnist_3trace_fixed-fanout_0250 | 1.201 | 1.171–1.294 | 30.895 | 31.054 | 18.961 |
| mnist_3trace_fixed-fanout_0500 | 1.415 | 1.331–1.522 | 37.233 | 32.109 | 20.812 |

## NEST results

These use the custom current-default Brunel port, not the historical full-indegree NEST runner. Neuron and plastic state use FP32; NEST event-buffer sums remain FP64. The NEST engine was built with GCC 14.3 and -O3 (its Nix wrapper stripped -march=native); the extension uses -O3 -march=native. NEST random streams and generated connectivity differ from GeNN and change with thread count.

The 32-thread placement was observed at runtime: all 32 workers were pinned to distinct guest logical CPUs. The test used passive OpenMP waiting. Other host workloads increased during the NEST matrix, so the 16-versus-32-thread rows are operational observations rather than an isolated SMT scaling experiment.

| Rule | Threads | Median µs/step | Range | E rate, Hz | I rate, Hz |
|---|---:|---:|---:|---:|---:|
| additive | 16 | 765.493 | 697.181–1039.361 | 4.382 | 4.416 |
| morrison | 16 | 1349.767 | 1290.546–1412.230 | 5.762 | 5.886 |
| additive | 32 | 9721.236 | 9270.818–9918.504 | 4.382 | 4.437 |
| morrison | 32 | 10226.252 | 9998.889–10410.727 | 5.996 | 6.028 |

## Workload and validation

GeNN/PyGeNN 5.4.0; FP32 simulation state, FP64 host normalization. The initial build used GCC 14.3 and -O3 -march=native, without -ffast-math, with NIX_ENFORCE_NO_NATIVE=0. Source snapshots, source and checkpoint hashes, compiler identity, exact command, environment, and process/load observations are in the run artifacts.

The result directories preserve the exact measured source. Subsequent harness changes only extend provenance and handle failed energy reads.

Real MNIST uses data/mnist, the frozen 23-case input bundle, seed 20260724, and the sequential training range starting at index 10000. Each replica begins from the same weights/theta checkpoint with runtime state reset (branched continuation), trains 100 accepted images, and includes all retry, 700-stimulus-tick and 300-rest-tick work in both the wall timer and denominator. No accuracy or convergence claim is made.

Brunel uses 9000 E + 2250 I neurons, 450/112 fixed indegrees with replacement and without recurrent autapses, 6,322,500 recurrent connections, dt=0.1 ms, next-tick delivery, arrival-timed causal-boundary STDP, sqrt(20) recurrent delivery scaling and external-rate scales 0.47/0.32. Each measurement follows 100 ms presimulation and covers 1000 ms / 10000 steps.

Construction, loading, compilation, presimulation and result serialization are excluded. GeNN workers rendezvous after setup and remain parked after timing until the entire batch finishes, so serialization and diagnostic reads do not contaminate the shared batch interval. The batch includes start/wakeup and tail imbalance. Package energy is collected once around the batch when supported.

Every GeNN replica in every timing repetition reproduced its CPU diagnostic step count, spike count, and event counters exactly. CPU and CUDA use different random generators; their spike totals need not be bitwise identical. Separate diagnostics check retries, finite state, weights, thresholds, column sums, activity concentration, and Brunel firing and weight distributions. NEST deterministic micro-tests check the voltage/current step, refractory behavior, next-cycle delivery, and the additive/Morrison simultaneous-event rule.

Brunel weight summaries sample approximately 100,000 plastic connections; the range is the sample range. CPU population Fano factors use all 9,000 excitatory neurons and complete 3 ms bins (the final 1 ms is omitted). Do not compare them directly to GPU statistics computed from a smaller neuron sample.

The combined baseline, reimplementation, Brunel and CPU checks passed: 41 tests. These establish the tested short-run mechanics and diagnostics; the custom NEST port has not been validated as a long-training replacement for all GeNN models.

For simulator selection, concurrency support and native-Linux energy commands, see [the CPU assessment](../assessment.md).

## Artifacts

- GeNN run: `copilot/tmp/cpu_genn_20260910_full32_native_c`
- Input manifest SHA-256: `e94363abf5e97f786c7217c509ce79ce96e395329ff2d5dff0b1c26639d57e80`
- GPU reference: `genn-sweep/baseline-20260907-matrix.json`
- Machine-readable CPU timings: [timings.csv](timings.csv)
- Per-case firing and final-state checks: [diagnostics.csv](diagnostics.csv)
- Compact provenance and results: [summary.json](summary.json)
- NEST timings and diagnostics: [nest-timings.csv](nest-timings.csv)
- NEST run: `copilot/tmp/cpu_nest_matrix_20260910_b/additive-t16`
- NEST run: `copilot/tmp/cpu_nest_matrix_20260910_b/morrison-t16`
- NEST run: `copilot/tmp/cpu_nest_matrix_20260910_b/additive-t32`
- NEST run: `copilot/tmp/cpu_nest_matrix_20260910_b/morrison-t32`
