# Diehl-Cook Differential Report

## Measured baseline

I reran the canonical Diehl and Cook evaluation against the cached reference activity files in `ref/activity/`.

- Reference artifacts used: `ref/activity/resultPopVecs10000.npy`, `ref/activity/inputNumbers10000.npy`
- Evaluation script: `ref/Diehl&Cook_MNIST_evaluation.py`
- Evaluation log: `copilot/tmp/ref_evaluation_rerun.log`
- Reproduced result: `91.56%`

The relevant output was:

```text
Sum response - accuracy:  91.56  num incorrect:  844
Sum response - accuracy --> mean:  91.56 --> standard deviation:  0.0
```

## Rust measurements

The Rust-side picture now has three informative checkpoints.

### 1. Historical conductance/cooldown pilot

- Log: `copilot/tmp/mnist_reference_like_quick_rerun.log`
- Config: 400 outputs, dense input connectivity, L1 normalization, post-renorm gain 78, 500 train / 200 mark / 500 test
- Result: `14.80%`

This run already showed that simply moving toward reference-style timing without fixing the rest of the stack was not enough.

### 2. Explicit E/I-only isolation with old numerics

- Log: `copilot/tmp/mnist_explicit_ei_mild_homeo.log`
- Config: 200 excitatory readout neurons plus 200 inhibitory interneurons, dense input connectivity, L1 normalization, post-renorm gain 1.0, slow scaling target 0.004, threshold increment 0.5, threshold homeostasis tau 20, 2000 train / 500 mark / 1000 test
- Result: `10.70%`

That run collapsed badly:

- Assigned class neurons: `[191, 0, 4, 0, 0, 0, 1, 0, 4, 0]`
- Average firing rate on test set: `2.35` spikes per neuron per sample
- Feedforward gains hit the minimum `0.25`
- Effective non-zero feedforward ratio after training: `11.21%`

This ruled out the simplest "missing E/I topology" explanation.

### 3. Current parameter-aligned explicit E/I run

I then kept the current-based formulation but aligned the main numerics to the Brian reference instead of reusing the earlier ad hoc scales.

- Log: `copilot/tmp/mnist_param_aligned_full.log`
- Config highlights:
	- `2000 / 500 / 1000` train / mark / test
	- `700` presentation ticks to match `0.35 s` at `dt = 0.5 ms`
	- excitatory tau `200`, inhibitory tau `20`
	- excitatory refractory `10` ticks, inhibitory refractory `4` ticks
	- threshold increment `0.003846154` and threshold homeostasis tau `20000000`
	- no slow feedforward scaling
	- corrected Poisson base rate `0.03125` and replay increment `0.015625`
	- independently converted `E -> I` and `I -> E` current weights: `1.56` and `-0.23`
	- pair-rule STDP kept, but with reference-like asymmetry and time constants: `lr+ = 0.01`, `lr- = 0.0001`, `tau+ = tau- = 40`
- Result: `43.50%`

Key diagnostics from that run:

- Assigned class neurons: `[43, 1, 15, 9, 7, 11, 10, 7, 18, 79]`
- Average firing rate on test set: `0.14` spikes per neuron per sample
- Feedforward gain statistics stayed fixed at `1.95 / 1.95 / 1.95`
- Effective non-zero feedforward ratio after training: `50.94%`

This is a material improvement:

- `10.70% -> 43.50%` relative to the explicit-E/I isolation run
- `37.20% -> 43.50%` relative to the earlier additive-current reduced-MNIST best

### 4. Parameter-aligned run with explicit rest phase

I then added the missing zero-input rest interval to the live presentation loop and reran the same reduced benchmark with `300` rest ticks, matching the reference `0.15 s` rest window at `dt = 0.5 ms`.

- Log: `copilot/tmp/mnist_param_aligned_rest_full.log`
- Config delta relative to the previous run:
	- same aligned neuron, weight, and STDP constants
	- same `2000 / 500 / 1000` split
	- same `700` stimulus ticks
	- plus `300` zero-input rest ticks between presentation attempts and accepted samples
- Result: `52.20%`

Key diagnostics from that run:

- Assigned class neurons: `[46, 3, 36, 11, 16, 34, 9, 7, 9, 29]`
- Average firing rate on test set: `0.11` spikes per neuron per sample
- Feedforward gain statistics stayed fixed at `1.95 / 1.95 / 1.95`
- Effective non-zero feedforward ratio after training: `56.59%`

This pushed the aligned current-based model substantially further:

- `43.50% -> 52.20%` after adding only the explicit rest phase
- `37.20% -> 52.20%` relative to the earlier additive-current reduced-MNIST best

## What the new run changes in the diagnosis

The parameter-aligned run falsifies the idea that the remaining gap is only about convergence speed.

The old explicit-E/I isolation run used the right graph but the wrong scales:

- same membrane tau for excitatory and inhibitory neurons
- no separate refractory/reset behavior for inhibitory neurons
- threshold adaptation that was orders of magnitude stronger and faster than the reference
- Poisson rates expressed on a different effective time base
- a borrowed `10.4 / 17.0` conductance ratio reused directly in current units, even though `E -> I` and `I -> E` land on different target populations

Once those scales were converted into the current-based model's units, the network stopped collapsing and exceeded the previous reduced-MNIST best.

So raw parameter mismatch was not a minor detail. It was a major part of the failure.

## Remaining mismatches

Important gaps still remain between the Brian reference in `ref/Diehl&Cook_spiking_MNIST.py` and the live Rust implementation.

### 1. STDP rule is still the strongest unresolved mismatch

Reference:

- Uses the Diehl-Cook trace-based update with `pre`, `post1`, `post2`, and `post2before`
- Potentiation depends on the delayed `post2before` term, not only elapsed pair timing

Rust:

- Still uses a simpler additive pair rule in `src/snn/synapse.rs`
- The new run only aligned the pair-rule constants, not the learning law itself

Why it matters:

- Parameter alignment alone got to `43.50%`
- The remaining `91.56 - 43.50 = 48.06` point gap is now most plausibly dominated by the learning-rule mismatch plus remaining dynamical simplifications

### 2. Synapses are still current pulses, not conductance state variables

Reference:

- Uses `ge` and `gi` conductances with `1 ms` and `2 ms` decay

Rust:

- Still adds recurrent and feedforward input as direct membrane current on the current tick
- No explicit synaptic state is carried across ticks

Why it matters:

- Even after scale conversion, the temporal shape of competition is still different from the reference

### 3. The presentation schedule is closer, but still not identical

Reference:

- `0.35 s` input, then `0.15 s` rest with zero input

Rust:

- The aligned run now matches the `0.35 s` presentation window through `700` ticks
- The live path now also includes an explicit `300`-tick zero-input rest phase between presentation attempts and accepted samples

Why it matters:

- Adding that rest phase alone improved the reduced benchmark from `43.50%` to `52.20%`
- The schedule is still not identical in every detail, but the missing rest phase was clearly a meaningful source of error

### 4. Evaluation is still reduced relative to the paper's pretrained protocol

Reference:

- Uses `10k` assignment samples and `10k` test samples

Rust aligned run:

- Uses `500` marking samples and `1000` test samples

Why it matters:

- This adds noise to the reported number
- But it does not explain a `48` point residual gap

## Conclusion

The canonical reference path is reproducible locally at `91.56%`.

The key Rust results are now:

- historical conductance/cooldown pilot: `14.80%`
- explicit E/I-only isolation with old numerics: `10.70%`
- parameter-aligned explicit E/I current-based run: `43.50%`
- parameter-aligned explicit E/I run plus explicit rest phase: `52.20%`

The main conclusion changed materially after the new benchmark.

Missing E/I topology alone was not the main bug, but neither was STDP merely a convergence-speed issue. The network was also badly numerically mis-scaled, and the missing rest interval mattered materially. Once tau values, refractory behavior, homeostasis magnitudes, Poisson rates, recurrent weight conversions, and the explicit rest phase were aligned, the same current-based architecture improved substantially.

That leaves the remaining gap concentrated in the parts that are still not reference-equivalent: the trace-based STDP rule and conductance-shaped synaptic dynamics.