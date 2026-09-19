# Single-network cycle and work accounting

This is a separate one-run PMU pass for each dense case, after the five-run latency matrix. One network ran at a time on guest logical CPU 9, using the same native binaries and 100 accepted MNIST images. Counters covered the simulation loop plus a small callback overhead; setup was excluded. Every result reproduced the baseline diagnostic step count and event counters. Each counter had time_running/time_enabled = 1.0, so no multiplexing scale factor was required.

| Dense MNIST | Active cycles/step | Instructions/step | Instructions/cycle |
|---|---:|---:|---:|
| mnist_1trace_dense | 64,331 | 210,927 | 3.28 |
| mnist_2trace_dense | 156,967 | 391,426 | 2.49 |
| mnist_3trace_dense | 110,806 | 282,948 | 2.55 |

These are main-thread user-mode hardware cycles measured with perf_event_open, not elapsed time multiplied by an assumed frequency. They include active memory stalls and exclude descheduled time. The earlier aggregate replica times cannot be converted into the cycles needed by one network. No single-network 10k-cycle timestep was observed here.

## Work per element

The network updates 784 input + 400 excitatory + 400 inhibitory neurons every tick. Synaptic work is event-driven: approximately 2.35–2.36 input spikes per tick trigger about 939–946 forward feedforward visits, followed by about 10–13 reverse STDP visits. It does not traverse the full 313,600-element feedforward matrix on every tick. Normalization does traverse that matrix once per 1,000-tick attempt and is included in the timer.

| Dense MNIST | FF pre visits/tick | FF post visits/tick | Recurrent visits/tick | Total mixed visits/tick | Amortized cycles/visit |
|---|---:|---:|---:|---:|---:|
| mnist_1trace_dense | 945.56 | 12.08 | 14.82 | 2556.46 | 25.16 |
| mnist_2trace_dense | 938.53 | 10.29 | 12.77 | 2545.59 | 61.66 |
| mnist_3trace_dense | 942.75 | 13.21 | 16.62 | 2556.58 | 43.34 |

Here a mixed visit means either a neuron-loop iteration or a synaptic visit. The last column divides total measured cycles by the combined count. It charges RNG, exponential/power functions, normalization and host control to those visits as well; it is not an isolated synapse-update instruction cost and should not be used as one. The recurrent count is 400 visits per inhibitory spike (including the zero-weight self entry) plus one per excitatory spike.

## Cache interpretation

The FP32 feedforward weights alone occupy 784 × 400 × 4 = 1,254,400 bytes (1.196 MiB), already larger than the 1 MiB per-core L2 reported by this guest. Connectivity indices, reverse maps, recurrent weights and neuron state increase the allocation further. Thus the complete network is not resident in one core's L2. Active rows can be hot and accesses can be served from other cache levels; allocation size does not determine the hit rate.

The generic perf cache-reference/cache-miss events in the raw output do not identify an L2 hit rate. The guest also reports a virtualized L3 topology, so it should not be treated as a direct physical CCD cache map. No claim of complete L2 residency or measured L2 miss latency is made.

For the 512-PU comparison, match event counts, learning rule, numerical functions and timestep boundaries. Peak FP32 FLOPs/cycle and full-matrix transfer bandwidth do not describe the work of this event-driven scalar implementation. The CPU can still beat a GPU on some small MNIST cases because the existing GPU profile shows several small kernel launches per tick; the corrected single-core Brunel results, in contrast, are much slower than the GPU baseline.

Artifacts: `copilot/tmp/cpu_single_counter_20260910_a/profile.py`, `manifest.json`, `summary.json` and `work-accounting.json`. The profiling script identifies the reused build directories through the origin manifest, checks live source hashes against it, and stores raw values plus enabled/running durations. A subsequent binary-sha256.json records hashes of the three generated libraries.
