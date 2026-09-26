# Image and Morrison workload extensions

These are initial implementations and pilots derived from the current FP32
baseline in `BASELINE.md`. They do not replace the frozen benchmark cases.
See [PILOTS.md](PILOTS.md) for the initial results and limitations.
The zero-delay Morrison experiment is now complete. Its
[final report](MORRISON-FINAL.md) consolidates the settled result, network and
two-trace STDP definition, spike/update counts per timestep, H800 runtime,
Brunel comparison, and preserved evidence.
The current [integer STDP timestamps](MORRISON-INTEGER-TICKS-20260920.md)
replace the earlier FP32 tolerance workaround. The report explains the floating-
point arithmetic, generated-kernel tests, and updated Morrison controls.
The [longer image runs](FULL-IMAGE-THETA10-20260920.md) record the completed
four-intervention comparison and the selected threshold-adaptation runs on
full and two-class Fashion-MNIST/CIFAR-10. `long_images.py` adds uninterrupted
training with isolated periodic evaluations and integer image STDP timestamps.
The [sequential countermeasure experiments](SEQUENTIAL-COUNTERMEASURES-20260922.md)
test stronger CIFAR interventions before running the three remaining Fashion
interventions on both class sets. `queue_images.py` runs one case at a time;
`long_images.py` records independent training and inference presentation times.

## Image workloads

`images.py` supports Fashion-MNIST (native 28×28 grayscale) and CIFAR-10
(native 32×32 RGB, channel-major). Both retain 400 E and 400 I neurons and
the baseline midpoint conductance dynamics, zero configured delay, 0.5 ms
timestep, normalization to 78, and 350 ms stimulus / 150 ms rest. Input
rates remain `pixel / 8 * intensity`, initially intensity 2. A retry raises
intensity by one when fewer than five E spikes occur. Dense plastic/total
effective counts are 313,600/473,600 for Fashion and 1,228,800/1,388,800 for
CIFAR. GeNN and NEST also store 400 zero inhibitory diagonal entries.

`--top-classes 2` selects labels by frequency across the entire training
split, breaking ties by ascending label ID, then filters both splits. It
does not merge labels or change E/I population sizes. Both official datasets
are balanced, so the selected labels are 0 and 1: T-shirt/top and trouser;
airplane and automobile. Their training/test sizes are 12,000/2,000 and
10,000/2,000 respectively, versus 60,000/10,000 and 50,000/10,000 in full.

Fresh initialization uses NumPy RandomState seed 20260724, positive uniform
weights, the selected structural mask, column normalization, and theta 20 mV.
All three rules share initialization within each dataset. Existing Bernoulli
and fixed-fan-out topologies remain available at 50%, 25%, and 12.5%; initial
pilots focus on the three dense rules. Each class contributes 20 held-out
training examples for neuron assignment. STDP trains on a seeded permutation
of the remaining training images, wrapping across epochs. Labels are used
only for filtering, split construction, and readout.

With `--evaluate`, a fresh, plasticity-off network loads the trained weights
and thresholds, uses inference inhibition 17, assigns neurons on the held-out
training images, then scores 20 test images per retained class. Thus full
pilots score 200 images and two-class pilots score 40. Neither set is used
for STDP. Weight and threshold immutability is checked after inference; the
training network is closed before inference so its runtime cannot be changed.
Class-normalized population scores use only retained classes, with ties
resolved by ascending label ID. These small probes are not final accuracy
measurements or the stock reference's same-activity evaluator.

The source extensions pass explicit model dimensions to GeNN, masks, and
portable checkpoint I/O. Checkpoint reuse verifies data hashes, class
selection, model constants, learning rule, mask, assignment selection and
training order. It restores only weights/theta and accepted sample count;
membranes, conductances, traces and RNG restart. It is a branched continuation.

Use the uncompressed Fashion IDX files already in `data/`. CIFAR uses the
[authors' binary archive](https://www.cs.toronto.edu/~kriz/cifar.html), with
archive MD5 `c32a1d4ab5d03f1284b67883e8d87530`. Its five training batches must
all be present in the supplied directory. There is no resize, whitening,
color conversion, augmentation, or ON/OFF expansion.

```sh
python workloads/images.py --dataset fashion-mnist --data-path data \
  --rule 3 --train-samples 1000 --evaluate --output FRESH_DIRECTORY

python workloads/images.py --dataset cifar10 --data-path data/cifar-10-batches-bin \
  --top-classes 2 --rule 2 --train-samples 1000 --evaluate --output FRESH_DIRECTORY

# All four datasets and all three dense rules, in separate processes:
python workloads/image_matrix.py --fashion-path data \
  --cifar-path data/cifar-10-batches-bin --train-samples 1000 --evaluate \
  --output FRESH_TRAINING_ROOT

# Short continuation from exactly the same twelve checkpoints on another host:
python workloads/image_matrix.py --fashion-path data \
  --cifar-path data/cifar-10-batches-bin --checkpoint-root FRESH_TRAINING_ROOT \
  --train-samples 100 --output FRESH_PROBE_ROOT
```

Set up GeNN 5.4 and CUDA as for the existing baseline. Use
`--backend single_threaded_cpu` for GeNN CPU. Use `--backend nest --threads 16
--nest-prefix NEST_INSTALL --module MODULE.so` for the native threaded CPU port.
Build its extension with `-DCPU_MNIST_INPUT_COUNT=3072`; the default remains 784
for historical MNIST use. A 3,072-capacity module also supports Fashion without
creating extra input neurons. NEST uses its own RNG streams, so its spike
trajectory need not equal GeNN's. Threads and host contention affect timings.

```sh
CXX=g++ cmake -S cpu-benchmark/mnist_module -B FRESH_BUILD \
  -DNEST_PREFIX=/absolute/NEST_INSTALL -DCPU_MNIST_INPUT_COUNT=3072
cmake --build FRESH_BUILD --parallel 2
```

Training timing includes normalization, transfers, retries, stimulus, and
rest. Compilation, dataset loading, state diagnostics and evaluation are
excluded. Each attempt contributes 1,000 timesteps. These are one-pass pilot
timings, not the five-repetition benchmark protocol. Runs stop on retry
intensity above 20, at least 5,000 E spikes per stimulus, non-finite checked
state, negative weights, or invalid final threshold ranges. Activity, theta,
weights, class balance, timing, exact commands and source/data hashes are
retained. Checks at presentation boundaries do not bound every transient.

## Morrison zero-delay derivative

The accepted endpoint and performance summary are in
[MORRISON-FINAL.md](MORRISON-FINAL.md): settling at 1,250 simulated seconds,
mean/SD 44.450353/3.986298 pA, and 212.51 µs per global timestep over the
final 100 seconds on H800 (249.79 µs across the complete run).

`morrison.py` uses the 112,500-neuron network from Morrison, Aertsen and
Diesmann (2007), with the explicitly requested delay change. It is a
zero-delay derivative, not a replication of the paper's 1.5 ms network:

- 90,000 E + 22,500 I; fixed indegrees 9,000 E and 2,250 I.
- 810,000,000 plastic E→E and 1,265,625,000 total recurrent synapses.
- Fixed-indegree sampling with replacement; recurrent autapses excluded.
- Current-based alpha LIF: tau_m 10 ms, capacitance 250 pF, refractory
  0.5 ms, threshold 20 mV, reset/rest 0, alpha rise constant 0.33 ms.
- Initial E weight 45.61 pA, I/E strength ratio 5; no recurrent current scaling.
- Independent per-neuron external Poisson count, aggregate 20,880 Hz.
- All-pairs power-law STDP: lambda 0.1, mu 0.4, alpha 0.1057,
  tau_plus = tau_minus = 20 ms, no upper weight bound or normalization.
- Zero GeNN delay steps and zero backpropagation delay. Spikes reach the
  next global 0.1 ms tick. Simultaneous pairs use old traces and are excluded
  from the zero-lag update, as checked in the generated-code micro-test.

The shared GeNN STDP code now uses `uint32` tick timestamps and exact equality
for ties. Integer elapsed ticks are converted to FP32 for trace decay. This
removes the absolute-time precision limit exposed by the earlier float gap
guard, while retaining four bytes per synaptic timestamp. Clock overflow is
a hard error; the range is approximately 4.97 simulated days at this timestep.
See the [integer timing report](MORRISON-INTEGER-TICKS-20260920.md).

The original paper's connectivity and parameters are documented in
[sections 3–4 and the appendix](https://doi.org/10.1162/neco.2007.19.6.1437).
Existing tuned Brunel defaults are preserved. Only the GeNN adapter accepts
the new explicit alpha-current and trace constants; historical other ports
are not implicitly converted.

The full-size runner requires an allocated 80 GB A800/H800 and uses FP32.
It records a manifest before construction, then checks population rates and
finite neuron state every chunk. A rate of 100 Hz aborts the pilot. The final
100,000-weight sample selects real edges, excluding sparse row padding, and
rejects negative/non-finite weights or values above 1,000 pA. It transfers
the weight array only outside the simulation timing. This is sampled weight
validation, not a proof about every synapse or continuous-time stability.

```sh
python workloads/morrison.py --sim-ms 10000 --chunk-ms 100 \
  --output FRESH_DIRECTORY
```

`cluster_job.sh` records the allocation and loads the existing cluster
environment. It runs a copied source snapshot under
`~/meow/stdp-experiments/workload-pilots-20260920`; `WORKLOAD_SOURCE` selects
the immutable snapshot. Submit separate named jobs with one A800 or H800,
8 CPUs and 120 GB host RAM. The pilot records device allocation separately
from simulation timing. No CPU or smaller-GPU full-size Morrison run is
part of this extension.

## Validation

The pre-change suites passed 33 GeNN/Brunel/baseline and 13 NEST tests.
The expanded tests cover frequency ties, native RGB layout, held-out sample
selection, retained-class readout, dimension-specific checkpoint restoration,
fixed fan-out, alpha integration against a matrix exponential, generated
Morrison zero-delay/tie behavior, sparse weight sampling, last RGB channel
routing, and immutable inference including above-cap weights.
The combined suite passed 60 tests after implementation.

```sh
python -m pytest --import-mode=importlib -q reimpl/tests brunel/tests \
  genn-sweep/test_baseline.py workloads/test_workloads.py
CPU_BENCH_NEST_PREFIX=NEST_INSTALL CPU_BENCH_MNIST_MODULE=MODULE.so \
  python -m pytest --import-mode=importlib -q cpu-benchmark/test_nest_mnist.py
```

To collect manifests and results from a completed pilot root:

```sh
python workloads/summarize.py PILOT_ROOT --output FRESH_RESULTS.json
```
