# Zero-Delay Midpoint Network

This document records the zero-delay Brian 1 reference experiment and the
workflow for deriving new network, learning-rule, and parameter variants from an
existing baseline. The authoritative implementation and run metadata are under
`ref/zero_delay_midpoint_v1/`; the logs under `ref/logs/` remain the source of
truth for progress after this document was written.

## Purpose

The stock Diehl-Cook demo delays each input-to-excitatory event by a value in the
0-10 ms interval. Setting all of those delays to zero aligns more excitatory
conductance arrivals within a 0.5 ms simulator tick. Earlier explicit-Euler
experiments became numerically unstable: large conductances caused the voltage
step to overshoot its reversal-potential envelope, after which conductance and
voltage values diverged.

The experiment had two goals:

1. stabilize the zero-delay conductance dynamics without changing the Brian tick
   scheduler or triplet STDP rule; and
2. recover accuracy comparable with the delayed reference implementation.

## Numerical Method

The zero-delay variant retains the reference 0.5 ms timestep and computes
analytically decayed midpoint conductances:

```text
ge_mid = ge(t) * exp(-dt / (2 * tau_e))
gi_mid = gi(t) * exp(-dt / (2 * tau_i))
g_mid  = g_leak + ge_mid + gi_mid
```

It then freezes those midpoint conductances over the voltage step and applies the
exact solution of that constant-coefficient membrane equation:

```text
v_inf  = weighted reversal potential / g_mid
v(t+dt) = v_inf + (v(t) - v_inf) * exp(-dt * g_mid / tau_m)
```

`ge` and `gi` themselves use exact exponential decay over the full step. This is
a second-order-style frozen-conductance approximation for the coupled system; it
is not a full closed-form solution of voltage with continuously changing
conductances. The generated Brian C state updater was checked with
`verify_midpoint.py` against an independent step calculation.

The following mechanisms remained unchanged:

- the 784-input, 400-excitatory, 400-inhibitory topology;
- the 350 ms stimulus and 150 ms rest schedule;
- the normalization target of 78;
- the triplet STDP event handlers and Brian scheduler;
- the 0.05 mV threshold increment and `10^7 ms` threshold time constant; and
- the retry rule requiring at least five excitatory spikes.

The feedforward delay is exactly 0 ms. Training uses 1.5 times the stock lateral
inhibition (`Ai -> Ae` weight 25.5 instead of 17). Plasticity-off inference uses
the stock inhibition weight 17; this split must be stated whenever reporting the
accuracy.

## Experiment Progression

An initial 1,000-sample inhibition sweep selected 1.5 times stock inhibition:

| Training inhibition | Probe accuracy |
| ---: | ---: |
| 1.00x | 58.4% |
| 1.25x | 62.5% |
| 1.50x | 63.3% |
| 1.75x | 57.6% |
| 2.00x | 59.9% |

The full stock-inhibition zero-delay control reached 75.2% at 10k samples but
collapsed to 51.8% at 20k and was stopped. The promoted 1.5x-training run reached
81.7% at 20k when it was also evaluated with 1.5x inhibition. A fixed-checkpoint
inference sweep showed that this readout was over-suppressed: evaluating the
same weights with stock inhibition raised the 20k score to 86.6%.

The promoted run then produced:

| Training checkpoint | Probe samples | Inference inhibition | Accuracy | Assigned neurons | Mean spikes | Mean active E neurons |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | 1,000 | 1.00x | 86.6% | 400 | 22.474 | 8.211 |
| 30k | 1,000 | 1.00x | 89.2% | 400 | 23.967 | 8.055 |
| 40k | 1,000 | 1.00x | 91.3% | 398 | 21.350 | 7.181 |

These are optimistic simple-demo probes: the same 1,000 test activities assign
neurons and score predictions. They are useful for a checkpoint curve but are
not interchangeable with the 10,000-image simple-demo score or the paper's
separate assignment protocol. The 40k result also shows that 30k was not an
accuracy-saturated endpoint.

No numerical runaway or saturated weights appeared through the validated 40k
checkpoint. The full 180k run was still active when this summary was written.

## Adaptive Threshold Observation

The spike predicate is:

```text
v > theta - 20 mV - 52 mV
```

so the actual firing threshold in millivolts is `theta - 72`. The promoted run
continued adapting after 30k:

| Checkpoint | Mean theta | Median theta | Theta standard deviation | Effective mean threshold |
| ---: | ---: | ---: | ---: | ---: |
| 20k | 32.175 mV | 35.965 mV | 8.225 mV | -39.825 mV |
| 30k | 37.534 mV | 38.070 mV | 3.959 mV | -34.466 mV |
| 40k | 40.486 mV | 40.621 mV | 3.112 mV | -31.514 mV |

At 20k, the failed stock-inhibition zero-delay run had mean `theta = 27.238 mV`,
about 4.94 mV below the successful run. This supports an association between a
higher realized adaptive threshold and a useful zero-delay operating point. It
does not prove that faster threshold adaptation caused the improvement: the
increment and decay parameters were unchanged, while lateral inhibition changed
the firing history. A causal speed experiment should hold the network and
inhibition fixed and scale `theta_plus` by a factor `k` while dividing
`tau_theta` by `k`, which changes response speed while approximately preserving
the rate-dependent equilibrium scale.

## Runtime And Artifact Layout

`ref/zero_delay_midpoint_v1/template/Diehl&Cook_spiking_MNIST.py` is the immutable
family source. Each training case has its own directory and each evaluation uses
a new directory below `evaluations/`. The launchers reject pre-existing outputs.

Important files are:

- `manifest.txt`: parameters, protocol, hashes, and selected results;
- `run_training.sh`: clean-directory training launcher;
- `evaluate_checkpoint.sh`: plasticity-off copied-checkpoint evaluator;
- `watch_accuracy.py`: stable-pair checkpoint watcher;
- `score_activity.py`: checkpoint probe scorer; and
- `verify_midpoint.py`: deterministic generated-state-updater check.

Training and evaluation identifiers have separate roles. `CASE` selects the
source training directory; `EVAL_TAG` identifies the inference configuration and
prevents several evaluations of one checkpoint from colliding. Every evaluation
records hashes of its source, weights, theta, recurrent matrices, activity, and
score.

The numbered Brian checkpoint files contain feedforward weights and theta only.
They do not contain membrane/conductance state, delay queues, refractory timers,
STDP traces, retry state, or RNG state. They are valid starting points for fresh
inference, but not exact resumable training snapshots.

## Deriving A New Experiment

Use the following sequence for new topology, STDP, delay, normalization,
inhibition, threshold, or integration experiments.

1. Choose the parent network and copy its source into a new experiment-family
   template. Do not edit a source file inside an active runtime directory.
2. Decide whether the change affects training or inference. Training/topology/
   learning-rule changes start from the same recorded random initialization.
   Inference-only changes copy a numbered learned checkpoint and use distinct
   evaluation tags.
3. Create a clean runtime directory per case, including private `random/`,
   `weights/`, `activity/`, and `results/` directories. Put logs in `ref/logs/`
   with the family and case in the filename.
4. Write the manifest and stopping rules before launch. Record the parent,
   changed variables, source and artifact hashes, Python/Brian/dependency
   versions, dataset order, training and inference parameters, service command,
   evaluation protocol, and expected checkpoint files.
5. Add a deterministic micro-test for the changed mechanism. For integration or
   delay changes, test one or a few ticks with known spikes and state. For STDP,
   test same-tick ordering, delayed arrival, trace sampling, and clamping.
6. Run a smoke test and a short pilot with runaway, silence, retry, and state
   guards. Compare against a paired parent case before promoting a parameter.
7. Save immutable weight/theta pairs and evaluate them every 10k accepted
   samples. Use 30k as a screening budget, at least 60k for one complete MNIST
   pass, and 180k for a reference-equivalent final comparison.
8. Judge results with accuracy and internal state: spike and active-neuron
   counts, retries/intensity, theta distribution, conductance/voltage bounds,
   weight sparsity/saturation, normalized column sums, and class-assignment
   balance.
9. Preserve failed cases, manifests, logs, and checkpoint hashes. Never replace a
   case in place with a tuned rerun.

For future memory-locality work, firing transcripts should be treated like other
checkpoint evaluations: use copied immutable checkpoints, a dedicated runtime
directory and tag, a recorded sample-selection seed/list, and an explicit trace
schema. The transcript must state whether plasticity was enabled and whether the
simulation began from fresh runtime state or a continuous training state; those
choices determine whether its implied STDP memory accesses match the original
training trajectory.
