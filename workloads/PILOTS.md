# Initial workload pilots — 2026-09-20

The implementation extends the FP32 GeNN and native NEST image baselines to
Fashion-MNIST and native RGB CIFAR-10. Morrison uses its original network
size with the requested zero configured delay, on 80 GB A800/H800 only.
Commands and model contracts are in [README.md](README.md). Machine-readable
results and per-run source/data hashes are in
[pilot-results-20260920.json](pilot-results-20260920.json).

## Workload sizes

| Workload | Input / E / I neurons | Plastic synapses | Total effective synapses |
|---|---:|---:|---:|
| Fashion-MNIST, full or labels 0/1 | 784 / 400 / 400 | 313,600 | 473,600 |
| CIFAR-10 RGB, full or labels 0/1 | 3,072 / 400 / 400 | 1,228,800 | 1,388,800 |
| Morrison | external drive / 90,000 / 22,500 | 810,000,000 | 1,265,625,000 recurrent |

Morrison has ten times as many neurons and 200 times as many plastic
connections as the current 5%-indegree Brunel default.

Image implementations also store 400 zero I→E diagonal entries. Morrison's
external drive is an independent aggregate Poisson count per neuron, not an
additional explicitly stored projection. All models use zero configured
GeNN delay: effective delivery is on the following global tick, 0.5 ms for
images and 0.1 ms for Morrison. NEST's image transport explicitly uses one
0.5 ms tick to match that delivery convention.

Filtering does not change the network size. Full Fashion has 60,000 training
and 10,000 test images; labels 0/1 have 12,000 and 2,000. Full CIFAR has
50,000 and 10,000; labels 0/1 have 10,000 and 2,000. Class frequency is
measured on the training split, with ascending label ID breaking ties.

## Image protocol and interpretation

Each of the four datasets is trained from scratch on RTX 3090 for 1,000
accepted images, separately for the existing dense one-, two- and three-trace
rules. Seed 20260724 controls initialization and presentation order. The
remaining MNIST settings, including normalization to 78 and 400 E/400 I,
are retained; CIFAR has no preprocessing beyond reading its native RGB
channels. These short runs establish executable workloads and expose early
learning behavior. They do not establish convergence or final accuracy.

Twenty training images per class are held out from STDP for neuron
assignment; twenty separate test images per class are scored. Thus accuracy
uses 200 test images for full datasets and 40 for two-class datasets. A
fresh inference network loads each checkpoint, freezes weights and adaptive
thresholds, and uses inhibition 17. Immutability is checked exactly. This is
not the reference demo's optimistic same-activity assignment/scoring protocol.

The Fashion two-class three-trace pilot collapses during inference: only
one neuron is assigned, to label 0, and the other 399 are silent during
assignment. All 40 predictions are label 0. Its 50% accuracy is an observed
failure of this early checkpoint/readout combination. The other Fashion
two-class variants retain both classes, although their assignment counts are
also imbalanced (385/15 and 382/18). No inference tuning was applied.

The CIFAR two-class three-trace checkpoint shows the same one-neuron
assignment collapse. Normalization is performed before training attempts,
as in the baseline; the saved final columns are not renormalized for
inference. Its largest final column sum is 200.53 versus the next training
attempt's target of 78. This makes checkpoint phase and inference inhibition
useful follow-up diagnostics; these results alone do not isolate the cause
of collapse. The maximum training retry intensity was 18, close to the
stopping limit of 20, so a long training run needs particular attention to
retry growth.

Each resulting checkpoint also starts a separate 100-image training
continuation on each of RTX 3090, A100, A800, H800, GeNN CPU (one thread),
and NEST CPU (16 threads). Checkpoint identity, input data, sample order and
class selection are checked before reuse. These are branched continuations:
weights, thresholds and sample count are restored; other runtime state and
RNG restart. Accuracy is measured on the initial 1,000-image checkpoints,
not on these timing continuations.

Timing includes normalization, weight/input transfers, all retries, stimulus
and rest. It excludes build/load, diagnostics and evaluation. Each attempt
has 1,000 simulation steps. These are single pilot measurements, not the
five-repeat historical benchmark protocol; CPU and GPU RNG streams differ.
The CPU runs use the local Ryzen 9 7950X. GeNN is pinned to logical CPU 9;
NEST uses 16 threads with active waiting and spread/core affinity.

## Image accuracy

| Dataset | One trace | Two traces | Three traces |
|---|---:|---:|---:|
| Fashion-MNIST — all classes | 52.0% | 57.0% | 50.0% |
| Fashion-MNIST — labels 0/1 | 92.5% | 95.0% | 50.0% |
| CIFAR-10 RGB — all classes | 25.5% | 22.5% | 15.5% |
| CIFAR-10 RGB — labels 0/1 | 57.5% | 77.5% | 50.0% |

All twelve learning runs completed their numerical checks. Both two-class
three-trace accuracy failures and the small evaluation sample sizes above
must be considered when interpreting these numbers.

## Agreement across GPUs

All six Fashion continuations had identical per-image, per-neuron activity
across the four GPUs. Five of the six CIFAR continuations also matched
activity and final weights exactly. The original H800 two-class two-trace
run diverged at accepted sample 6 (26 versus 25 E spikes); retry intensity
first differed at sample 54. It finished with 8,559 E spikes, versus 7,821
on the other GPUs, although total attempts were the same. Its final weight
RMS difference from RTX 3090 was 0.00130.

An independent H800 repeat, job 550894, matched RTX 3090 activity, weights
and thresholds exactly. Both records are retained. The source, checkpoints,
model parameters, structural masks and sample order match; the numerical
cause of the initial divergence has not been isolated. These results do
not establish guaranteed bitwise replay, even with the recorded seed.

## Image timing

Values are wall-clock microseconds per executed simulation step for the
100-image continuations. Each row starts from the same checkpoint across
platforms. All 72 primary continuations completed. The two additional
diagnostic repeats are retained separately.

| Dataset / rule | RTX 3090 | A100 | A800 | H800 | GeNN CPU, 1 thread | NEST CPU, 16 threads |
|---|---:|---:|---:|---:|---:|---:|
| Fashion, full, 1 trace | 38.89 | 22.14 | 17.44 | 14.39 | 17.15 | 131.29 |
| Fashion, full, 2 traces | 38.66 | 23.51 | 18.75 | 16.67 | 58.43 | 135.59 |
| Fashion, full, 3 traces | 47.71 | 26.12 | 21.06 | 18.07 | 31.25 | 131.75 |
| Fashion, two classes, 1 trace | 38.05 | 21.76 | 17.74 | 14.73 | 21.72 | 132.16 |
| Fashion, two classes, 2 traces | 38.61 | 23.94 | 19.42 | 16.50 | 55.18 | 135.68 |
| Fashion, two classes, 3 traces | 50.15 | 27.13 | 21.37 | 18.98 | 30.81 | 128.42 |
| CIFAR RGB, full, 1 trace | 54.99 | 50.96 | 41.03 | 36.21 | 45.39 | 488.12 |
| CIFAR RGB, full, 2 traces | 57.57 | 57.95 | 46.73 | 44.84 | 428.00 | 558.30 |
| CIFAR RGB, full, 3 traces | 64.63 | 64.07 | 51.03 | 46.36 | 158.74 | 511.57 |
| CIFAR RGB, two classes, 1 trace | 54.98 | 52.77 | 43.10 | 37.47 | 47.45 | 498.61 |
| CIFAR RGB, two classes, 2 traces | 56.12 | 62.55 | 51.06 | 45.75* | 526.37 | 528.90 |
| CIFAR RGB, two classes, 3 traces | 71.06 | 75.04 | 63.60 | 54.21 | 282.66 | 535.91 |

The marked H800 two-class two-trace timing is the original run; the repeat that
matched the other GPUs took 47.20 µs/step. The RTX 3090 full one-trace
entry uses the confirmation run without archive compression. Neither value is a median
over repetitions. These timings describe the recorded simulator/host/GPU
combinations, not an intrinsic hardware speed limit.

## Morrison results

Both full-size networks completed 10 simulated seconds (100,000 steps),
with diagnostics every 100 ms and plasticity enabled throughout.

| Device | Simulation wall time | µs / step | Allocated device memory | Mean E / I rate | Final E rate |
|---|---:|---:|---:|---:|---:|
| A800 80 GB | 74.25 s | 742.45 | 28,446 MiB | 6.091 / 6.387 Hz | 5.355 Hz |
| H800 80 GB | 46.23 s | 462.25 | 28,548 MiB | 6.058 / 6.360 Hz | 5.489 Hz |

Memory is the process allocation after construction, not a measured peak.
Build/initialization took 18.56 s and 16.57 s respectively, outside the
simulation timer. Rate and finite-state checks passed. The final sample of
100,000 actual E→E edges had mean weight 45.507/45.508 pA, standard deviation
1.178/1.174 pA, and ranges 40.399–50.698/40.694–51.182 pA (A800/H800).
No sampled weights were zero. Padding is excluded from the sample.

Activity declines over the pilot, so this does not establish a stationary
state or the paper's long-term learned distribution. The zero-delay network
is a deliberate derivative of the original 1.5 ms model. It is feasible at
full size on the authorized devices, but longer plastic runs still need
rate and weight monitoring. No full-size CPU or smaller-GPU Morrison run
was launched.

## Implementation state and validation

Parent Git revision: `9ad791ebc67086cb28047b82a884bceb55a565d3`.
Pilot source was copied into immutable experiment snapshots under
`copilot/tmp/workload_pilots_20260920/`. All source archives are hashed in
the result record. The initial worktree was clean apart from existing
untracked contents in the reference submodule; those were preserved.

- `source_v1`: initial smoke runs. Full-size Morrison simulated successfully
  but failed in its final diagnostic because PyGeNN does not expose a public
  sparse variable view. Those failed records are retained.
- `source_v2`: bounded sparse weight sampling fixed; successful Morrison
  10-second pilots and Fashion learning/evaluation runs.
- `source_v3`: inference guards ensure that disabled plasticity cannot clip
  weights above the learning cap; NEST adaptive thresholds are also frozen.
  All platform continuations and CIFAR learning/evaluation use this snapshot.
  Fashion inference already passed exact immutability checks under v2.

Pre-change tests: 33 GeNN/Brunel/baseline and 13 NEST tests passed. After the
extension, 60 tests passed with one existing NEST deprecation warning.
Tests include generated zero-delay/simultaneous-pair behavior, an independent
alpha-integrator check, native RGB layout and last-channel routing, class
ties, dimensions/checkpoints, structural masks, and frozen inference for all
three rules. Logs are `tests_initial.log`, `tests_nest.log` and
`tests_inference.log` under the experiment root.

GeNN is 5.4.0, NumPy 2.3.4, and NEST 3.9.0-post0.dev14. GPU compilation uses
CUDA 12.8, with the existing machine-specific environments. The local CUDA
environment is `copilot/tmp/power_env_3090_20260910.sh`; the older baseline
script referenced a garbage-collected Nix CUDA path and its failed initial
smoke attempt was preserved. CIFAR's official archive checksum was verified
on all hosts before extraction.

Per-run manifests contain the exact Python command, environment, model,
checkpoint/data/source hashes and hardware identity. Full local artifacts
remain under `copilot/tmp/workload_pilots_20260920/`, including source
snapshots, checkpoints, activity, generated code, logs and dataset integrity
records. Remote generated artifacts remain under
`/data/meow/stdp-workload-pilots-20260920/` on A100 and
`~/meow/stdp-experiments/workload-pilots-20260920/` on the cluster. Output
directories must be fresh; reruns do not overwrite earlier evidence.

The first CIFAR checkpoint upload began before compression completed and
failed extraction. Those partial copies were preserved; the completed
archive was transferred under a new name, SHA256-verified and extracted to
`verified_inputs_cifar/` before any remote simulation. The RTX 3090 full
one-trace continuation overlapped that compression, so its timing was
repeated without the overlap in `probe_cifar_3090_confirm/`; the original
record remains available.
