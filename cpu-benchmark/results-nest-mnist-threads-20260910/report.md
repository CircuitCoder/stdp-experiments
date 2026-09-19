# Quick NEST MNIST: one versus 32 threads

One network at a time on the Ryzen 9 7950X host. Dense MNIST only, all three trace rules. Each case used 20 accepted images and three sequential timing repetitions after its diagnostic. Thread counts were tested sequentially (one, then 32). Other host processes remained active.

| Rule | 1 thread, median µs/tick | 32 threads, median µs/tick | Speedup |
|---|---:|---:|---:|
| mnist_1trace_dense | 123.013 | 104.044 | 1.18× |
| mnist_2trace_dense | 137.968 | 98.430 | 1.40× |
| mnist_3trace_dense | 126.239 | 101.173 | 1.25× |

| Rule | 1 thread range, µs/tick | 32 threads range, µs/tick |
|---|---:|---:|
| mnist_1trace_dense | 118.880–126.838 | 98.766–105.615 |
| mnist_2trace_dense | 137.257–139.687 | 95.119–99.435 |
| mnist_3trace_dense | 126.162–126.591 | 100.751–1439.506 |

The first 32-thread three-trace repetition took 1439.506 µs/tick; the other two took 100.751 and 101.173 µs/tick. The slow repetition is retained in the raw results and range. Its cause was not isolated; these measurements share the host with other processes. The median does not characterize that tail latency.

The NEST runner and compiled native module are unchanged from the completed 21-case experiment. Each rule starts from the same frozen 10,000-image weights/theta checkpoint in baseline_inputs_20260907_v2, resetting other runtime state, then uses real MNIST training images starting at index 10000. This is a branched continuation timing test, not an accuracy evaluation. The complete source snapshots and hashes are recorded per thread count.

Each biological tick is 0.5 ms. Each attempt includes 700 stimulus and 300 rest ticks, plus intensity retries when needed. Elapsed timing includes native NEST synchronization, host control, normalization and final spike-count reads; construction and setup are excluded. No replica or thread count divides elapsed time. Native NEST random streams depend on thread count: the one-thread cases each took 20 attempts; the 32-thread one-, two- and three-trace cases took 20, 21 and 21 attempts. The denominator uses the actual executed ticks. This is a short operational concurrency comparison, not fixed-spike-train scaling.

Both configurations use OMP_WAIT_POLICY=ACTIVE, GOMP_SPINCOUNT=300000, OMP_PROC_BIND=spread and OMP_PLACES=cores. Observed thread counts were exactly one and 32. The one-thread worker was affined to logical CPUs 0–1 (one reported physical core); the 32-thread workers were affined two per physical core, spanning logical CPUs 0–31. WSL2 affinity does not reserve host CPU resources.

All 18 timing repetitions matched their own diagnostic final FP32 weights and thresholds, population/event counts and executed synaptic visits. Source/module identity between the two configurations and time/step denominators were checked. No runtime code changes or rebuild were needed. The earlier 16-thread full matrix used 100 images and five repetitions, so it is not used here to calculate a scaling ratio.

Normalization cap overshoots remain nonfatal observations. Power was not measured.

- Raw runs and exact launch command: `copilot/tmp/cpu_nest_mnist_t1_t32_20260910_a`
- [Timings and activity counts](timings.csv)
- [Results, validation, manifests and thread placement](summary.json)
- [Earlier 16-thread full matrix](../results-nest-mnist-20260910/report.md)
