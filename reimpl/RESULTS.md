# Zero-delay three-trace reimplementation results

Run dates: 2026-07-23 to 2026-08-05 UTC

Host: AMD Ryzen 9 7950X, 16 physical/32 logical CPUs, 30 GiB RAM,
NixOS under WSL2. Independent jobs were assigned by the scheduler across the
available CPUs; each Brian2 and GeNN CPU run itself used one thread, while NEST
used eight. Long-run wall timings include contention from pre-existing
reference experiments and the concurrently running framework cases; they are
measured operational throughput on this host, not isolated peak benchmarks.

Dataset and protocol: real MNIST from `data/mnist`; sequential training order;
first 1,000 test images for the same-activity assignment/scoring probe unless a
row explicitly says 100 images. This optimistic probe is compatible with the
local Brian 1 reference evaluator but is not the paper protocol.

Reference points:

| Brian 1 checkpoint | Probe accuracy | Spikes/image | Active neurons/image |
|---|---:|---:|---:|
| `inhib150`, 1,000 samples | 63.3% | 12.942 | 6.846 |
| `inhib150_full`, 30,000 samples | 89.2% | 23.967 | 8.055 |

## Pilot training

`seconds/cycle` includes normalization, simulator execution, retries,
stimulus, and rest, and excludes import/build/data/checkpoint/evaluation.

| Backend | Accepted | Attempts | Training wall | Seconds/cycle | Probe accuracy | Probe size |
|---|---:|---:|---:|---:|---:|---:|
| Brian2 runtime/Cython | 1,000 | 1,086 | 460.676 s | 0.000424195 | 60.2% | 1,000 |
| GeNN single-threaded CPU | 1,000 | 1,084 | 156.245 s | 0.000144138 | 58.7% | 1,000 |
| GeNN CUDA | 1,000 | 1,096 | 84.474 s | 0.000077074 | 59.2% | 1,000 |
| NEST CPU, 8 threads | 100 | 115 | 39.891 s | 0.000346880 | 81.0% | 100 |
| NEST CPU, explicit one-step delay, 8 threads | 1,000 | 1,133 | 339.970 s | 0.000300062 | 46.3% | 1,000 |

The older 100-image NEST accuracy is not comparable to the 1,000-image rows:
assigning and scoring from only 100 images substantially inflates this
optimistic protocol. That pilot uses lazy in-module column normalization and
the corrected pre-event update order
(`nest_cpu_pilot100_ordering_lazy_20260723_a`). NEST requires its minimum
0.5 ms transport delay, despite the earlier logical-zero description; Brian2
and GeNN deliver at zero delay.

The later fixed-step audit makes that NEST timing explicit rather than calling
it logical zero delay. Feedforward, E-to-I, and I-to-E connections all use one
0.5 ms step, and construction asserts the actual NEST connection delays. An
integration probe emits at 1.0 ms and verifies zero target conductance at 1.0
ms followed by the analytically decayed conductance at 1.5 ms. Thus NEST does
implement the requested next-cycle transport.

That change did not align learning. The 1,000-sample checkpoint differs from
column-normalized initialization by only `6.22e-7` RMS; its maximum absolute
change is `6.82e-6`, no weight changes by more than `1e-5`, and correlation is
`0.99999999994`. The 1,000-image probe scored 46.3%, compared with 63.3% for
the Brian 1 1k reference checkpoint, 60.2% for fresh Brian2, and 58.7% for
fresh GeNN. It averaged 9.45 spikes and 3.773 active neurons per image. The
artifacts are `nest_cpu_fixed_step_pilot1000_20260724_a` and
`nest_cpu_fixed_step_pilot1000_eval1000_20260724_a`; checkpoint SHA-256 is
`6df9b569b482751a620c2e17dcd7164eba939756f76016225df23bee8eb0cf66`.

The backend had already been subject to NEST's same one-resolution physical
minimum in the earlier 30k run, so this explicit representation does not
define a new long-run dynamics case. The failed 1k gate, together with the
existing 48.4% 30k result, does not justify another 30k CPU run.

## Imported reference controls

These rows evaluate an immutable Brian 1 checkpoint after conversion to the
portable format. They isolate inference scheduling from learned weights.

| Evaluator | Imported checkpoint | Accuracy | Reference in Brian 1 |
|---|---:|---:|---:|
| Brian2 runtime/Cython | 1,000 | 59.0% | 63.3% |
| GeNN single-threaded CPU | 1,000 | 58.7% | 63.3% |
| NEST CPU, 8 threads | 1,000 | 57.3% | 63.3% |
| GeNN single-threaded CPU | 10,000 | 74.5% | 80.6% |
| Brian2 runtime/Cython | 30,000 | 83.3% | 89.2% |
| GeNN single-threaded CPU | 30,000 | 88.2% | 89.2% |
| NEST CPU, 8 threads | 30,000 | 87.8% | 89.2% |

The 1k pilot gap is therefore largely reproduced with the exact Brian 1
weights/theta, rather than being unique to newly trained weights.

## Long runs

Fresh 30,000-sample runs and immutable 1,000-image checkpoint probes completed
for Brian2 CPU, GeNN CPU, GeNN CUDA, and NEST CPU.

GeNN's fresh checkpoints scored 71.6% at 10k and 82.8% at 20k. The imported
Brian 1 10k checkpoint scored 74.5% in the same GeNN evaluator, separating a
2.9-point training-state gap from the larger cross-simulator inference gap.
Brian2's fresh checkpoints scored 82.1% at 10k and 88.6% at 20k. The 10k
Brian 1-native reference probe scored 80.6% under the same optimistic
protocol; Brian2's 20k result is already within 0.6 points of the Brian 1-native
30k result.

NEST's fresh 10k checkpoint scored 47.4%. Its weights remained almost equal to
normalized initialization (RMS delta `3.57e-6`, correlation `0.999999998`, and
1.31% of weights changed by more than `1e-5`). The deterministic
pre/post/post/pre probe proves that the custom synapse can potentiate, but the
NEST network's 0.5 ms minimum transport scheduling does not reproduce the
burst-driven specialization of the zero-delay reference. This is a failed
alignment result, not an accuracy match.

| Backend | Accepted | Attempts | Training wall | Seconds/cycle | Accuracy | Spikes/image | Active/image |
|---|---:|---:|---:|---:|---:|---:|---:|
| Brian2 runtime/Cython | 30,000 | 30,329 | 18,408.285 s | 0.000606953 | 89.2% | 14.316 | 3.593 |
| GeNN single-threaded CPU | 30,000 | 30,748 | 5,455.013 s | 0.000177410 | 87.3% | 14.152 | 4.582 |
| GeNN CUDA | 30,000 | 30,746 | 2,245.391 s | 0.000073030 | 89.2% | 14.348 | 4.661 |
| NEST CPU, 8 threads | 30,000 | 35,544 | 17,181.876 s | 0.000483397 | 48.4% | 9.089 | 4.090 |

The fresh GeNN CPU 30k result is 0.9 percentage points below the imported Brian 1
30k checkpoint in the same GeNN evaluator (88.2%), and 1.9 points below the
Brian 1-native 89.2% probe. This is within the expected cross-simulator gap
shown by the imported-checkpoint controls.

The fresh Brian2 30k result exactly matches the Brian 1-native 89.2% probe.
The fresh GeNN CUDA result also scores 89.2%. It is 1.9 points above the prior
fresh GeNN CPU realization; because each run has backend-specific graph and
Poisson RNG streams, this is an accuracy consistency result rather than proof
that CPU and CUDA execute an identical random trajectory. CUDA processed a
training timestep cycle in 73.030 microseconds, 2.43 times faster than the
operational GeNN CPU 30k measurement. The pilot scored 59.2%, between the
compatible fresh GeNN CPU (58.7%) and Brian2 CPU (60.2%) controls.

NEST improved only from 47.4% at 10k to 48.4% at 30k and therefore does not
match the reference. Its minimum transport delay and resulting network event
scheduling are a material model difference, not just a performance detail.
Brian2 and GeNN also fire less than the Brian 1 reference (23.967 spikes and
8.055 active neurons per image), so their accuracy alignment is not full
dynamical equivalence.

Final checkpoint SHA-256 values:

- Brian2: `93dd752c07f3fe0550fd932c0d3747a708043a0be75e00b044f5114b12b6eaaf`
- GeNN: `512049dcfffd182a5ec70e6571e36a04ca1532b0e688e77da47c4a7f4799ca6d`
- GeNN CUDA: `7d182e42cd9e6d44ae09064106abcb99eb6458b3b8f16cd9c5747312d91c23fe`
- NEST: `aff9f4bd0bc03384a7c7632946c9f6b542763b45f032492fd8691a1768e8d290`

## GPU runtime results and limits

CUDA was verified with an actual CUDA 12.8 SAXPY kernel and all-zero numerical
error on an NVIDIA GeForce RTX 3090 (24 GiB, compute capability 8.6; driver
596.49). This is stronger than device enumeration: compilation, launch,
synchronization, and result transfer all completed.

GeNN CUDA is the complete host-controlled three-trace workload. The 1k pilot
used real MNIST, sequential training order, 1,096 attempts, and 1,096,000
simulated 0.5 ms ticks. It trained in 84.474 s (77.074 microseconds/cycle) and
its fresh 1,000-image probe scored 59.2%. The 30k run used 30,746 attempts and
30,746,000 ticks; simulator and controller execution took 2,245.391 s
(73.030 microseconds/cycle, 13.361 accepted samples/s). Its fresh 1,000-image
probe scored 89.2% in 74.090 s (72.567 microseconds/cycle). Build time was
4.777 s for training and is excluded from the cycle metric.

The primary artifacts are:

- `genn_cuda_mnist_pilot1000_20260725_a` and its `eval1000` directory;
- `genn_cuda_mnist_30k_20260725_a` and
  `genn_cuda_mnist_30k_eval1000_20260725_a`; and
- final portable checkpoint SHA-256
  `7d182e42cd9e6d44ae09064106abcb99eb6458b3b8f16cd9c5747312d91c23fe`.

The measured training and evaluation commands were:

```sh
python reimpl/run_genn.py train --backend cuda \
  --output reimpl/runs/genn_cuda_mnist_30k_20260725_a \
  --samples 30000 --stats-interval 1000 --checkpoint-interval 10000

python reimpl/run_genn.py evaluate --backend cuda \
  --output reimpl/runs/genn_cuda_mnist_30k_eval1000_20260725_a \
  --checkpoint reimpl/runs/genn_cuda_mnist_30k_20260725_a/checkpoints/checkpoint_030000.npz \
  --samples 1000
```

Brian2CUDA compiled and executed the real 784-400-400 network, midpoint
updater, and three feedforward traces for one 350 ms stimulus plus 150 ms rest
attempt. The attempt emitted 10 excitatory spikes and passed the five-spike
gate. Measured kernel execution was 0.308925 s for 1,000 cycles
(308.925 microseconds/cycle); compilation took 27.890 s and is excluded. This
artifact is `brian2cuda_mnist_one_attempt_20260725_a`.

```sh
python reimpl/run_brian2cuda_codegen.py --run \
  --output reimpl/runs/brian2cuda_mnist_one_attempt_20260725_a \
  --cuda-path /tmp/stdp-cuda-toolkit-genn-12.8 \
  --cuda-runtime-version 12.8 --gpu-id 0 --compute-capability 8.6
```

That Brian2CUDA number is a kernel/runtime validation, not training throughput
or accuracy. CUDA standalone queues the run before executing its binary and
cannot perform the Python-controlled normalization, adaptive retry decision,
intensity change, periodic report, and checkpoint action between attempts
while preserving hidden device state. Implementing those controls in generated
C++/CUDA is separate work.

The vendored NEST-GPU runtime was built and exercised by the Brunel workloads,
but it cannot represent this exact MNIST rule. `src/get_spike.h` and
`src/rev_spike.h` call `SynapseUpdate(syn_group, weight*, delta_t)`, while
`conn12b`/`conn16b` store target metadata and one float weight. An exact port
therefore requires a core connection/state ABI extension. The static MNIST
checkpoint validations below do not change that training limitation.

## One-trace GPU extension

Date: 2026-08-05 UTC. CUDA execution used the same RTX 3090, the CUDA 13.0
toolkit, and GeNN 5.4. Runs were executed sequentially after the GPU became
vacant. The common implementation now supports these derived variants:

- `one-trace-dense`: 313,600 feedforward synapses, `wmax=1`,
  `nu_post=0.0005`;
- `one-trace-bernoulli-0125`: the exact seeded Brian 1 mask with 39,001
  synapses, `wmax=8`, and `nu_post=0.002639015821545789`; and
- the existing dense three-trace rule as `triplet-dense`.

The GeNN runner exposes feedforward `postsynaptic` and `presynaptic`
parallelism plus `num_threads_per_spike`. A fresh 100-sample sweep found that
the dense default was fastest at 70.143 us/cycle; presynaptic x1/x8/x32/x64
took 79.355/75.608/72.552/77.482 us/cycle. For the 12.5% graph,
presynaptic x32 improved 65.397 to 64.574 us/cycle, only 1.3%. The 30k runs
therefore used postsynaptic x1 for dense and presynaptic x32 for sparse.

| Fresh GeNN CUDA training | Attempts | Simulation wall | E2E wall | us/cycle | 30k accuracy |
|---|---:|---:|---:|---:|---:|
| Dense [`run`](runs/genn_cuda_onetrace_dense_train30000_post_20260805_a/results/performance.json) | 30,698 | 2,206.580 s | 2,212.438 s | 71.880 | 83.2% |
| Bernoulli 12.5% [`run`](runs/genn_cuda_onetrace_sparse0125_train30000_pre32_20260805_a/results/performance.json) | 30,006 | 2,062.215 s | 2,067.943 s | 68.727 | 74.1% |

The compatible checkpoint trajectories are:

| Variant | Evaluator | 10k | 20k | 30k | 30k spikes | 30k active E | 30k assigned |
|---|---|---:|---:|---:|---:|---:|---:|
| Dense | Brian 1 | 84.3% | 88.4% | 86.2% | 22.551 | 8.284 | 400 |
| Dense | GeNN CUDA fresh | 74.4% | 82.7% | 83.2% | 15.313 | 4.420 | 387 |
| Sparse 12.5% | Brian 1 | 76.5% | 76.9% | 77.9% | 22.156 | 14.692 | 400 |
| Sparse 12.5% | GeNN CUDA fresh | 68.5% | 73.7% | 74.1% | 19.470 | 9.927 | 399 |

Dense GeNN is 3.0 points below the Brian 1 reference at 30k and sparse is 3.8
points below. Both GeNN probes are dynamically narrower. The dense final
training block had 16.037 spikes and 4.917 active neurons versus Brian 1's
20.135/6.889. Sparse GeNN had 17.812/7.938 versus Brian 1's 19.688/13.185.
The mismatch therefore appears in activity, theta, learned weights, and
accuracy rather than only in the readout.

The sparse GeNN run also exposed a missing reference safety check. A
post-normalization `wmax * 1.02` violation was first observed in the periodic
block ending at 17k; transient maxima reached 33.37 and the terminal maximum was 25.02. Brian 1 would have aborted
at 8.16 before simulating the next attempt. The common Brian2 and GeNN runners
now enforce that guard and have a deterministic unit test. The completed 20k
and 30k sparse artifacts predate the guard and are retained as diagnostic
performance/accuracy evidence, not a valid dynamics match.

The NEST-GPU synapse ABI has no accumulated source trace for the exact
one-trace power update, so no NEST-GPU training result is claimed. A new static
runner reconstructed the dense and sparse Brian 1 30k checkpoints with one-step
transport and an `aeif_cond_beta` conductance approximation:

| Static NEST-GPU inference | Accuracy | Spikes/image | Active E | E2E wall | us/cycle |
|---|---:|---:|---:|---:|---:|
| Dense [`probe`](runs/nestgpu_onetrace_dense_ref30k_eval1000_20260805_a/results/performance.json) | 85.4% | 26.940 | 11.977 | 374.148 s | 370.029 |
| Sparse [`probe`](runs/nestgpu_onetrace_sparse0125_ref30k_eval1000_20260805_a/results/performance.json) | 76.7% | 29.665 | 19.676 | 371.695 s | 369.503 |

The close scores result from imported Brian-trained weights, not matched
training. Both NEST-GPU probes fire more broadly than Brian 1, and the beta
conductance model is explicitly recorded as a non-equivalence in each manifest.

Brian2CUDA compiled and executed both one-trace graphs for one complete
stimulus/rest attempt. Dense took 448.015 us/cycle and sparse took 498.991
us/cycle. Compilation took about 430 s in each isolated single-job build and
dominates their 433.769/432.620 s build-and-run walls. These are CUDA runtime
validations without presentation-level training or checkpoint accuracy.

Final GeNN checkpoint SHA-256 values:

- dense: `5bc3a6c75488b797a201a664175cea1ff67e43729bcdeca9677c2b999db8bce8`;
- sparse diagnostic: `0c1ceb654ee9ec2a1e1e67dbd2b7f7f4bcb26016bf839f635580abb981776580`.

The directory `genn_cuda_onetrace_sparse0125_train30000_pre32_eval10k_20260805_a`
is a preserved failed launch: it stopped before compilation because `make` was
not on that shell's `PATH`. The successful 10k result uses the `_b` tag linked
through `../performance.md`; the 20k and 30k evaluations use `_a`.

Exact firing and synapse-update counters, amortized ns/update values, all sweep
artifacts, and the timing-scope caveats are in `../performance.md`.

## Source state

- Brian2 vendored commit: `1bfa1a9`
- Brian2CUDA vendored commit: `825c0c5`
- GeNN vendored commit: `563c45c`
- NEST vendored commit: `182eba4` (reported runtime 3.9 development build)
- NEST-GPU vendored commit: `830b15b`

Generated run directories, logs, manifests, checkpoints, scores, and timing
JSON files are all under `reimpl/runs/` and `reimpl/logs/`. Earlier smoke,
float-precision, pre-counter-fix, and pre-delivery-fix artifacts are preserved
but are not used in the tables above.
