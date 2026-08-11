# Sparse One-Trace Networks

Date: 2026-07-23

## Goal

Test whether permanent structural sparsity in the input-to-excitatory projection
can reduce the cost of the zero-delay one-trace network without losing the
approximately 89% accuracy reached by the dense reference-sized variants.

The experiment matrix is:

| Topology | Nominal rates |
| --- | --- |
| Independent Bernoulli connection per input/excitatory pair | 50%, 25%, 12.5% |
| Fixed fan-in per excitatory neuron | 392, 196, 98 inputs (50%, 25%, 12.5%) |

## Motivation From The Dense 30k Checkpoint

Artifact: `ref/zero_delay_one_trace_v1/full_nupost0005_30000/weights/XeAe.npy`.

The 30,000-sample one-trace checkpoint has 59.045% exactly zero weights. A
threshold of 0.001 removes 60.49% of pairs but only 0.006% of total weight mass;
a threshold of 0.01 removes 63.27% of pairs and 0.131% of mass. The fraction of
strictly positive weights declined from 78.59% at 10k, to 59.23% at 20k, to
40.95% at 30k. The corresponding 1,000-image checkpoint-probe accuracies were
84.3%, 88.4%, and 86.2%.

These measurements show that the learned dense matrix becomes sparse, but they
do not imply that the same pairs can be removed before learning. Some weights
that reach zero can later revive under the one-trace update. The present test
therefore measures permanent random structural masks rather than pruning a
trained network.

## Brian 1 Feasibility

Brian 1 supports both requested sparse construction modes through a fixed
`SparseConnectionMatrix`:

- Bernoulli masks are generated once with a seeded independent score per pair.
- Fixed fan-in masks select the lowest seeded scores independently for each
  excitatory column.
- Sparse delayed propagation and compiled C STDP operate on the same structural
  entries.
- Column access permits in-place normalization without densifying the matrix.
- Checkpoint serialization writes every structural pair, including a pair whose
  current value is zero, so a zero-valued present synapse remains eligible to
  potentiate later.

The topology is immutable during a run: absent pairs cannot be created by STDP.
Fixed fan-out is also feasible with the analogous row-wise construction, but it
is not part of this 3x2 matrix.

`ref/zero_delay_one_trace_sparse_v1/run_verifiers.sh` checks mask determinism and
nesting, exact fixed fan-in, initial normalization, weight bounds, compiled
sparse delay/STDP operation, sparse column normalization, and explicit-zero
checkpoint reconstruction. All checks passed before the first smoke launch.

## Controlled Configuration

Parent initialization and recurrent matrices come from
`ref/zero_delay_one_trace_v1/full_nupost0005_30000/random/`. The learned 30k
weights are used only for the motivation statistics above; sparse training
starts from the parent's original random feedforward initialization.

All six cases use:

- real MNIST with the stock sequential training order;
- 784 inputs, 400 excitatory neurons, and 400 inhibitory neurons;
- zero feedforward delay and exponential-midpoint conductance integration at
  0.5 ms per tick;
- 350 ms presentation plus 150 ms rest and the stock retry rule;
- one-trace power STDP with `tc_pre=20 ms`, target 0.4, exponent 0.2;
- input-column normalization target 78 before every attempt;
- 1.5x lateral inhibition during training and 1.0x during evaluation;
- connectivity seed 20260723; and
- one process and one BLAS/OpenMP thread per case.

One shared 784x400 score matrix defines all masks. Masks are nested within each
topology family, which makes rate comparisons paired. Bernoulli and fixed fan-in
masks are derived from the same scores but are not identical.

## Rate Compensation

Removing connections while retaining column sum 78 increases weight per present
synapse. Keeping the dense `wmax=1` would make 12.5% initialization invalid and
would compress the learning rule into a different relative weight range. The
pilot starts with:

| Rate | `wmax = 1/p` | `nu_post = 0.0005 * (1/p)^0.8` |
| ---: | ---: | ---: |
| 50% | 2 | 0.0008705505632961241 |
| 25% | 4 | 0.001515716566510398 |
| 12.5% | 8 | 0.002639015821545789 |

This preserves the initial normalized weight as a fraction of `wmax`. The
learning-rate exponent compensates the `(wmax-w)^0.2` factor so the effective
post-event scale rises approximately as `1/p`, matching the scale change of a
present weight. This is a hypothesis to screen, not a claim of equivalence.

## Staged Protocol

Each case receives its own immutable runtime directory, copied source, mask,
manifest, environment file, and hashes. The stages are:

1. Train 100 accepted samples and reject non-finite state, runaway firing,
   normalization overflow, silence, or retry storms.
2. Train a fresh case for 1,000 accepted samples and evaluate the final
   checkpoint on the first 1,000 MNIST test images.
3. If dynamics are credible, train a fresh case through 30,000 accepted samples.
   Evaluate immutable 10k, 20k, and 30k checkpoints on the first 1,000 test
   images, then evaluate the final checkpoint on the same 1,000-image probe.

The checkpoint probe uses the same test activity for neuron assignment and
scoring. It is the optimistic simple-demo protocol and is comparable to the
earlier one-trace checkpoint probes, not to the paper's separate-assignment
protocol. The 1,000-image probe also has more sampling variance than the stock
10,000-image evaluation.

## Smoke Results

All six 100-sample smokes completed without aborts or non-finite state.

| Topology | Rate | Mean spikes | Mean active E neurons | Retry fraction | Final observed max weight |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bernoulli | 50% | 11.06 | 3.06 | 2.91% | 0.4637 |
| Bernoulli | 25% | 13.06 | 3.78 | 0.00% | 1.0612 |
| Bernoulli | 12.5% | 16.13 | 4.71 | 0.00% | 2.3537 |
| Fixed fan-in | 50% | 11.40 | 3.57 | 1.96% | 0.4742 |
| Fixed fan-in | 25% | 12.63 | 3.40 | 0.99% | 0.9531 |
| Fixed fan-in | 12.5% | 15.83 | 5.10 | 0.00% | 1.9150 |

These are early dynamics only. No parameter was changed between smoke and pilot.

## Pilot Results

All six fresh 1,000-sample pilots and their isolated 1,000-image evaluations
completed without aborts. Training values aggregate ten 100-sample diagnostic
blocks; evaluation values cover the full probe.

| Topology | Rate | Train spikes | Train active | Train retry | Present zero fraction | Accuracy | Assigned | Eval spikes | Eval active |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bernoulli | 50% | 11.636 | 3.957 | 2.44% | 1.480% | 64.3% | 397 | 14.379 | 7.812 |
| Bernoulli | 25% | 13.052 | 4.478 | 0.40% | 1.605% | 63.3% | 381 | 15.572 | 8.621 |
| Bernoulli | 12.5% | 16.173 | 6.399 | 0.00% | 1.959% | 65.6% | 383 | 19.077 | 10.615 |
| Fixed fan-in | 50% | 11.698 | 4.026 | 2.25% | 1.510% | 66.1% | 398 | 14.434 | 7.812 |
| Fixed fan-in | 25% | 13.147 | 4.529 | 0.70% | 1.685% | 63.7% | 385 | 15.801 | 8.743 |
| Fixed fan-in | 12.5% | 16.011 | 6.364 | 0.00% | 2.084% | 66.1% | 387 | 19.007 | 10.635 |

Every case assigned neurons to all ten classes. Class counts are retained in
`ref/logs/zero_delay_one_trace_sparse_v1_accuracy.log`.

The successful evaluation directories end in `_completed_final_1000`. Six
directories without `_completed` are preserved failed launch artifacts: an
initial systemd ordering attempt started the evaluator before training had
finished, so it exited before copying a checkpoint or running Brian. They are
not measurements and do not appear in the accuracy ledger.

The dense `nu_post=0.0005` pilot scored 70.2% with 399 assigned neurons, 14.852
training spikes and 8.429 active neurons, 6.89% training retries, and 16.908
evaluation spikes with 10.277 active neurons. Sparse pilot accuracy is therefore
4.1 to 6.9 points lower, but the cases cluster within 2.8 points of one another,
remain class-readable, and show no numerical or learning-bound failure. The
12.5% cases reproduce dense-like evaluation activity; the denser sparse cases
are quieter but still assign 397-398 neurons at 50%.

No parameter is changed for 30k. Retuning inhibition or normalization separately
by rate would erase attribution to structural sparsity, while the pilot provides
no single clearly failed rate or topology to correct. The 30k checkpoints will
show whether the early accuracy gap closes with learning.

## 30k Runs

All six approved configurations were launched as fresh cases at 2026-07-23
08:14:42 UTC. Every trainer completed 30,000 accepted samples and every watcher
completed the planned 10k, 20k, and 30k checkpoint probes plus a final-repeat
probe. Each run has matched numbered and unsuffixed final weights and theta,
source/input hashes, and a final artifact manifest. No trainer aborted and no
watcher stopped or reported an evaluation failure.

| Case | Training service | Watcher service |
| --- | --- | --- |
| `full_bernoulli_p050_30000` | `ref-zd-sparse-full-bernoulli-p050` | `ref-zd-sparse-full-bernoulli-p050-watch` |
| `full_bernoulli_p025_30000` | `ref-zd-sparse-full-bernoulli-p025` | `ref-zd-sparse-full-bernoulli-p025-watch` |
| `full_bernoulli_p0125_30000` | `ref-zd-sparse-full-bernoulli-p0125` | `ref-zd-sparse-full-bernoulli-p0125-watch` |
| `full_fixed_fanin_p050_30000` | `ref-zd-sparse-full-fixed-p050` | `ref-zd-sparse-full-fixed-p050-watch` |
| `full_fixed_fanin_p025_30000` | `ref-zd-sparse-full-fixed-p025` | `ref-zd-sparse-full-fixed-p025-watch` |
| `full_fixed_fanin_p0125_30000` | `ref-zd-sparse-full-fixed-p0125` | `ref-zd-sparse-full-fixed-p0125-watch` |

Training logs are
`ref/logs/zero_delay_one_trace_sparse_v1_<case>_train.log`; watcher logs use
the same prefix with `_watch.log`. Accuracy records append to
`ref/logs/zero_delay_one_trace_sparse_v1_accuracy.log`.

### Checkpoint accuracy

All values below use the fixed 1,000-image optimistic simple-demo probe defined
above. All six final probes assigned all 400 excitatory neurons, with every
class represented.

| Topology | Rate | 10k | 20k | 30k/final | Final spikes | Final active E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bernoulli | 50% | 79.6% | 82.1% | **84.2%** | 22.062 | 10.603 |
| Fixed fan-in | 50% | 81.3% | 81.0% | **84.2%** | 21.851 | 10.610 |
| Bernoulli | 25% | 76.8% | 79.8% | 77.5% | 21.554 | 12.513 |
| Fixed fan-in | 25% | 78.9% | 78.6% | 80.6% | 21.572 | 12.483 |
| Bernoulli | 12.5% | 76.5% | 76.9% | 77.9% | 22.156 | 14.692 |
| Fixed fan-in | 12.5% | 75.9% | 78.0% | 78.7% | 22.067 | 14.622 |

The final-repeat evaluations reproduced their corresponding 30k results
exactly. The two 50% cases tie at 84.2% and are the strongest sparse cases.
Fixed fan-in has no measured advantage at 50%, but finishes 3.1 points above
Bernoulli at 25% and 0.8 point above it at 12.5%. The Bernoulli 25% trajectory
peaked at 20k and then declined, so reduced connectivity does not merely require
more samples under the tested compensation.

### Terminal connectivity and dynamics

`Present zero` measures zero-valued weights among structurally present
synapses. `Global zero` also counts absent pairs, while `effective nonzero` is
the fraction of all 784x400 possible pairs that remain present and nonzero.

| Topology | Nominal rate | Actual structural rate | Present zero | Global zero | Effective nonzero | Final max weight | Last-block retry |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bernoulli | 50% | 50.023% | 60.311% | 80.146% | 19.854% | 2.0010 | 0.398% |
| Fixed fan-in | 50% | 50.000% | 59.986% | 79.993% | 20.007% | 2.0013 | 0.200% |
| Bernoulli | 25% | 24.869% | 62.985% | 90.795% | 9.205% | 4.0002 | 0.200% |
| Fixed fan-in | 25% | 25.000% | 62.931% | 90.733% | 9.267% | 4.0000 | 0.200% |
| Bernoulli | 12.5% | 12.437% | 64.662% | 95.605% | 4.395% | 8.0000 | 0.100% |
| Fixed fan-in | 12.5% | 12.500% | 65.020% | 95.628% | 4.372% | 8.0017 | 0.000% |

The compensation kept terminal column sums near 78 and saturation negligible
in every case. Last-block retry fractions remained below 0.4%. Lower structural
rates increased the number of active excitatory neurons during evaluation, but
did not cause silence, runaway firing, normalization failure, or class collapse.
The large global zero fractions are therefore the intended combination of a
fixed structural mask and learned zeros, not evidence of numerical failure.

### Outcome

The dense `nu_post=0.0005` one-trace control scored 84.3%, 88.4%, and 86.2% at
10k, 20k, and 30k under the same 1,000-image checkpoint protocol. The best
sparse result, 84.2% at 50% structural connectivity, is 2.0 points below that
dense 30k result and 4.8 points below the approximate 89% target. Its effective
nonzero fraction is only about 20% of all possible input-to-excitatory pairs,
however, so 50% structural connectivity is a credible accuracy/cost operating
point if that loss is acceptable.

Neither 25% nor 12.5% connectivity recovered dense accuracy. Fixed fan-in is
the safer of the two construction modes below 50%, but its 80.6% and 78.7%
terminal results remain materially below the dense model. On this single paired
mask family and seed, the experiment supports 50% as the sparsest tested rate
worth considering for further training or tuning; it does not support the more
aggressive rates as drop-in replacements. Because not all Brian randomness is
owned by the connectivity seed, small topology differences still require
replication before being treated as general effects.

The order-dependent accuracies over the preceding 10,000 training samples were
80.39%/80.86% for Bernoulli/fixed fan-in at 50%, 76.25%/76.71% at 25%, and
72.56%/73.35% at 12.5%. These are retained as training diagnostics only and are
not combined with the checkpoint-probe curve above.

## Bernoulli 12.5% GPU portability check

Date: 2026-08-05. The Bernoulli 12.5% case was ported using the exact saved
Brian mask: both implementations contain the same 39,001 structural pairs from
`RandomState(20260723)`. GeNN CUDA used presynaptic parallelism with 32 threads
per spike, selected by a 100-sample sweep that improved 65.397 to 64.574
us/cycle relative to the postsynaptic default.

The diagnostic 30k GeNN run used 30,006 attempts and 30,006,000 cycles.
Simulation took 2,062.215 s, whole-program time was 2,067.943 s, and throughput
was 68.727 us/cycle. Despite 87.6% fewer structural synapses than dense, this
was only 4.4% faster per cycle than dense GeNN because feedforward delivery is
not the sole runtime cost.

| Accepted samples | Brian 1 | GeNN CUDA diagnostic |
| ---: | ---: | ---: |
| 10k | 76.5% | 68.5% |
| 20k | 76.9% | 73.7% |
| 30k | 77.9% | 74.1% |

At 30k, GeNN assigned 399 neurons and produced 19.470 spikes with 9.927 active
neurons per image, versus 400 assignments, 22.156 spikes, and 14.692 active
neurons for Brian 1. Its final training block was similarly narrower:
17.812 spikes and 7.938 active neurons versus 19.688 and 13.185. The learned
dynamics do not match.

More importantly, the periodic block ending at 17k first exposed a violation
of this experiment family's normalization guard: normalization produced a weight above
`wmax * 1.02 = 8.16`. A few transient weights later reached 33.37 and the final
maximum was 25.02, even though column sums stayed near 78. Brian 1 would abort
before simulating the next attempt because the fractional power is undefined
above `wmax`. The common Brian2 and GeNN runners now enforce the same 2% guard.
The completed GeNN 20k/30k artifacts predate that fix and remain useful for
performance and mismatch diagnosis, but are not valid equivalent trajectories.

NEST-GPU static inference from the valid Brian 1 30k checkpoint scored 76.7%
with 29.665 spikes and 19.676 active neurons. This uses a documented
`aeif_cond_beta` approximation and has plasticity disabled because the current
NEST-GPU synapse ABI cannot implement the accumulated one-trace state. A
Brian2CUDA one-attempt validation also completed at 498.991 us/cycle. Neither
measurement is a full sparse training result.

Detailed timing, firing, and logical synapse-update counters are in
`performance.md`. The diagnostic GeNN checkpoint SHA-256 is
`0c1ceb654ee9ec2a1e1e67dbd2b7f7f4bcb26016bf839f635580abb981776580`.
