# CPU single-network latency

Run from the repository root. This benchmark uses the current 23-case FP32
GeNN-CUDA workload definitions in `BASELINE.md`.

The [corrected 7950X latency results](results-latency-20260910/report.md)
contain all 23 cases and the earlier NEST Brunel timings at 16 and 32 threads.
The [NEST MNIST matrix](results-nest-mnist-20260910/report.md) adds all 21
MNIST cases using one network with 16 native NEST threads.
The [CPU assessment](assessment.md) covers simulator selection and energy
measurement options. Power estimation is deferred for the present experiment.

GeNN's installed CPU backend runs one simulation thread. The experimental
ISPC backend was inspected too: it uses SIMD loops, without a multicore task
launcher. No usable multicore GeNN backend was enabled for this suite.
`run.py` therefore runs exactly one network at a time, pinned to one logical
CPU, and reports its elapsed wall time divided by its executed timesteps.
Five repetitions run sequentially. The runner rejects `--workers` above one.

The [earlier replica experiment](results-20260910/report.md) measures aggregate
independent-network throughput. Its aggregate values are **not timestep
latency** and do not support a CPU speed comparison with one GPU network.

```sh
nix-shell cpu-benchmark/shell.nix --run 'export NIX_ENFORCE_NO_NATIVE=0; taskset -c 9 python cpu-benchmark/run.py --output copilot/tmp/cpu-fresh-run --repetitions 5 --samples 100'
```

Every output directory must be fresh. Each case first builds and runs a
diagnostic; timing repetitions reuse that generated library without rebuilding.
Diagnostic and timing population/event counts must agree. All builds and
artifacts are preserved. The Nix shell sets the generated-code compiler to
`g++ -O3 -march=native`, without GeNN's optional `-ffast-math`.
`NIX_ENFORCE_NO_NATIVE=0` is essential because Nix otherwise strips the native
architecture option. Other Python/BLAS thread pools use one thread. CPU 9 is
the logical CPU selected for the September 10 run; choose an appropriate CPU
for a new host. Pinning does not reserve the physical core or its SMT sibling.

Timing excludes construction, compilation, loading, presimulation and result
serialization. It includes the simulation loop, host control, normalization
and retry work specified by the GPU baseline. Parent process setup/rendezvous
is outside this timer. Any internal simulation synchronization belongs inside
the timer; the current GeNN CPU loop is synchronous and has no worker barriers.

The standard input bundle is `copilot/tmp/baseline_inputs_20260907_v2`.
If unavailable, prepare a fresh bundle with
`python genn-sweep/baseline.py prepare --output NEW_DIRECTORY`, then pass it
using `--inputs`. The parent checkpoints and real MNIST data must be present.

## NEST MNIST

`run_nest_mnist.py` implements the complete 21-case MNIST matrix using the
separate `mnist_module/` extension. It runs one network with native NEST
threading. The fixed-degree masks use **fan-out**: 200, 100 or 50 E targets per
input. Random masks are Bernoulli at the corresponding densities. Both use
the same frozen masks and 10,000-image weights/theta checkpoints as GeNN.

```sh
nix-shell cpu-benchmark/shell.nix
export NIX_ENFORCE_NO_NATIVE=0
CXX=g++ cmake -S cpu-benchmark/mnist_module -B /absolute/fresh/mnist-module-build \
  -DNEST_PREFIX=/absolute/nest-build/install
cmake --build /absolute/fresh/mnist-module-build --parallel 2

OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=300000 OMP_PROC_BIND=spread OMP_PLACES=cores \
python cpu-benchmark/run_nest_mnist.py \
  --nest-prefix /absolute/nest-build/install \
  --module /absolute/fresh/mnist-module-build/cpumnistmodule.so \
  --output copilot/tmp/nest-mnist-fresh-run --threads 16 --repetitions 5 --samples 100
```

Each case has a diagnostic followed by sequential timing repetitions. Timing
includes NEST synchronization, normalization, host control, stimulus/rest
ticks and retries. Weights belong to their target neuron so postsynaptic
learning occurs at the correct tick, even for silent inputs. Native static
connections route input identities; their transport weights are not the
learned weights. Shared input traces use alternating buffers separated by
NEST's per-tick barrier. This extension supports one network in one MPI
process; it is not a general distributed NEST synapse model.

Normalization uses an FP64 sum and scale, then stores FP32 weights, inside
the target neurons. It can exceed the STDP cap. Per the benchmark policy,
cap overshoots are recorded without aborting; STDP clipping is unchanged.
Finite state, nonnegative weights, structural masks, normalization sums,
runaway firing and retry limits are still checked. Each timing repetition
must exactly reproduce its diagnostic spikes, executed synaptic visits,
final weights and thresholds. Native NEST random streams differ from GeNN.

The dense thread pilot uses five images and two timing repetitions for each
of 1, 2, 4, 8, 16 and 32 threads:

```sh
bash cpu-benchmark/run_nest_mnist_threads.sh copilot/tmp/nest-mnist-fresh-threads \
  /absolute/nest-build/install /absolute/mnist-module-build/cpumnistmodule.so

CPU_BENCH_NEST_PREFIX=/absolute/nest-build/install \
CPU_BENCH_MNIST_MODULE=/absolute/mnist-module-build/cpumnistmodule.so \
python -m pytest --import-mode=importlib -q cpu-benchmark/test_nest_mnist.py

python cpu-benchmark/summarize_nest_mnist.py --nest COMPLETE_NEST_MNIST_RUN \
  --genn COMPLETE_GENN_CPU_LATENCY_RUN --thread-pilot COMPLETE_THREAD_PILOT \
  --output NEW_REPORT_DIRECTORY
```

The [NEST MNIST results](results-nest-mnist-20260910/report.md) retain the full
matrix, GeNN comparisons, thread pilot and normalization-bound observations.
The [full-run validation record](../copilot/tmp/cpu_nest_mnist_full16_20260910_a/validation/validation.json)
records 61 passed tests, 105 exact repetition checks and source/module hashes.
The earlier `reimpl/run_nest.py` training port remains a historical experiment.

## NEST Brunel

`run_nest.py` runs one multithreaded network. The separate `nest_module/`
implements the current Brunel arrival timing, next-step delivery, current
integration order and STDP tie rule, preserving raw plastic weight units.
It does not change the historical `brunel/nest_brunel_stdp.py` runner or the
MNIST NEST port. The extension stores post histories for this bounded
benchmark; it is not intended for unbounded training runs.

Build the vendored NEST with Python and OpenMP, then the extension:

```sh
nix-shell cpu-benchmark/shell.nix
export NIX_ENFORCE_NO_NATIVE=0
bash cpu-benchmark/build_nest.sh /absolute/fresh/nest-build
CXX=g++ cmake -S cpu-benchmark/nest_module -B /absolute/fresh/module-build \
  -DNEST_PREFIX=/absolute/fresh/nest-build/install
cmake --build /absolute/fresh/module-build --parallel 2

OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 OMP_PROC_BIND=spread OMP_PLACES=threads \
python cpu-benchmark/run_nest.py \
  --nest-prefix /absolute/fresh/nest-build/install \
  --module /absolute/fresh/module-build/cpubrunelmodule.so \
  --output copilot/tmp/nest-fresh-run --threads 32 --repetitions 5 --rule additive
```

Use `--rule morrison` for the other rule. Set NEST's thread count through
`--threads`; NEST ignores `OMP_NUM_THREADS`. Compare physical-core concurrency
and SMT concurrency rather than assuming more threads reduce latency.
The native NEST RNG and connectivity generation differ from GeNN and depend
on NEST's thread count. Repeatability is checked within each configuration.

To run both rules at both thread counts in separate fresh case directories:

```sh
bash cpu-benchmark/run_nest_matrix.sh copilot/tmp/nest-fresh-matrix \
  /absolute/fresh/nest-build/install \
  /absolute/fresh/module-build/cpubrunelmodule.so
```

## Energy and measurement boundaries

`energy.py` samples readable Linux powercap **package** energy counters at
250 ms intervals and at measurement boundaries, handles counter wraparound, and
deduplicates sysfs aliases. Nested core domains are excluded to avoid double
counting. Package energy includes other processes and idle package overhead;
it excludes external DRAM. It is recorded once around the run, including the
small parent wakeup/completion overhead, while the latency timer is inside
the simulation worker. No per-core attribution or idle subtraction is performed.

Absent or failed energy counters produce null watts/joules. This is the
expected result in the current WSL2 container; CPU timing remains available.
Do not replace those nulls with TDP. GPU board energy includes a different
hardware boundary, and the existing GPU measurements run one network, so
label the power scope when comparing results. No power result is claimed here.

## Checks and reporting

```sh
nix-shell cpu-benchmark/shell.nix --run 'python -m pytest --import-mode=importlib -q genn-sweep/test_baseline.py reimpl/tests brunel/tests cpu-benchmark/test_energy.py cpu-benchmark/test_latency_report.py'
CPU_BENCH_NEST_PREFIX=/absolute/nest-build/install \
CPU_BENCH_NEST_MODULE=/absolute/module-build/cpubrunelmodule.so \
python -m pytest --import-mode=importlib -q cpu-benchmark/test_nest.py

python cpu-benchmark/summarize.py --genn COMPLETE_GENN_RUN \
  --nest COMPLETE_NEST_RUNS --output NEW_REPORT_DIRECTORY
```

The script rejects historical replica runs, incomplete runs, unpinned
measurements and inconsistent time/step denominators. All displayed CPU,
NEST and GPU timings are the elapsed time per timestep of one network.
The complete matrix and diagnostics are retained in CSV.
