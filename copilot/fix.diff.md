# Reference-Informed Fix Report

## Status

The live Rust source no longer matches the earlier "explicit E/I only" isolation state. I kept the explicit excitatory/inhibitory circuit, but added the missing numerical hooks needed to express the Diehl-Cook parameters in the current-based implementation.

This created a fourth fix checkpoint beyond the earlier measurements:

1. Historical conductance/cooldown pilot: `14.80%`
2. Explicit E/I-only isolation with old numerics: `10.70%`
3. Current parameter-aligned explicit E/I run: `43.50%`
4. Current parameter-aligned explicit E/I run plus explicit rest phase: `52.20%`

## Current code change set

### 1. Explicit E/I circuit remains in `src/snn/network.rs`

The live network still keeps:

- `N` excitatory readout neurons and `N` inhibitory interneurons
- input driving only the excitatory population
- one-to-one `E -> I` excitation
- dense `I -> E` inhibition without self-pairs

### 2. Neuron parameterization is now split by population

The neuron builder can now express separate excitatory and inhibitory numerics:

- independent membrane taus
- independent thresholds and reset potentials
- independent refractory durations
- separate inhibitory homeostasis overrides

This matters because the Brian reference uses very different excitatory and inhibitory time scales.

### 3. Recurrent directions are no longer forced to share a conductance ratio in current units

The old builder derived `E -> I` from `I -> E` using the raw `10.4 / 17.0` conductance ratio.

That was wrong after changing formulations, because `E -> I` and `I -> E` land on different target populations with different membrane constants. The live code now allows a separately converted `E -> I` strength.

### 4. STDP numerics are now configurable without changing the learning rule

The synapse builder still uses the pair rule, but it can now take reference-like constants for:

- potentiation and depression rates
- potentiation and depression time constants
- feedforward weight cap

### 5. The live presentation path now includes an explicit rest phase

The old code accepted `rest_ticks` on the CLI but did not use it. The live `run_presentation` path now executes a zero-input rest interval after each presentation attempt, including replay attempts.

This matches the reference structure more closely than the previous reset-only approximation.

## Measurement

### Reference baseline

- Log: `copilot/tmp/ref_evaluation_rerun.log`
- Result: `91.56%`

### Historical conductance/cooldown pilot

- Log: `copilot/tmp/mnist_reference_like_quick_rerun.log`
- Result: `14.80%`

### Explicit E/I-only isolation with old numerics

- Log: `copilot/tmp/mnist_explicit_ei_mild_homeo.log`
- Result: `10.70%`

### Current parameter-aligned explicit E/I run

I rebuilt the release binary and reran the comparable reduced real-MNIST condition with converted reference-like numerics.

- Log: `copilot/tmp/mnist_param_aligned_full.log`
- Config highlights:
	- `train=2000`, `mark=500`, `test=1000`, `output_num=200`
	- `per_sample_ticks=700`
	- excitatory tau `200`, inhibitory tau `20`
	- excitatory refractory `10`, inhibitory refractory `4`
	- threshold increment `0.003846154`, threshold homeostasis tau `20000000`
	- no slow feedforward scaling
	- base Poisson rate `0.03125`, replay increment `0.015625`
	- `E -> I = 1.56`, `I -> E = -0.23`
	- pair-rule STDP numerics aligned to the Brian magnitudes: `0.01`, `0.0001`, `40`, `40`
- Result: `43.50%`

Relevant output:

```text
Assigned class neurons: [43, 1, 15, 9, 7, 11, 10, 7, 18, 79]
Accuracy: 43.50%
Firing rate statistics (per sample): max 3.24, min 0.00, avg 0.14
Epoch 1/1 accuracy: 43.50%
```

Additional diagnostics from the same run:

- Feedforward gain statistics after training: min/avg/max `1.9500`
- Effective non-zero feedforward ratio after training: `50.94%`
- Replay increases stayed occasional rather than constant

### Current parameter-aligned explicit E/I run with rest phase

I then reran the same reduced real-MNIST benchmark after wiring `rest_ticks` into the live presentation loop and setting `--rest-ticks 300`.

- Log: `copilot/tmp/mnist_param_aligned_rest_full.log`
- Result: `52.20%`

Relevant output:

```text
Assigned class neurons: [46, 3, 36, 11, 16, 34, 9, 7, 9, 29]
Accuracy: 52.20%
Firing rate statistics (per sample): max 1.18, min 0.00, avg 0.11
Epoch 1/1 accuracy: 52.20%
```

Additional diagnostics from the same run:

- Feedforward gain statistics after training: min/avg/max `1.9500`
- Effective non-zero feedforward ratio after training: `56.59%`

## Assessment

This fix attempt materially improved the current Rust model.

Compared with the explicit E/I-only isolation run, the parameter-aligned run:

- raised accuracy from `10.70%` to `43.50%`
- brought average firing down from `2.35` to `0.14`
- avoided gain collapse to the minimum clamp
- produced a much broader class assignment instead of near-total single-class collapse

Compared with the earlier additive-current reduced-MNIST best (`37.20%`), it also improved the comparable benchmark by `6.30` points.

So the old assumption that STDP only affected convergence time was too optimistic. The network was also strongly mis-scaled numerically. Aligning tau values, replay probabilities, threshold adaptation magnitudes, and recurrent weight conversion changed the learned dynamics substantially.

Adding the explicit rest phase improved things again:

- `43.50% -> 52.20%` with the same aligned parameters
- average firing dropped from `0.14` to `0.11`
- class assignments became materially more balanced

That makes the rest phase one of the highest-impact single changes tested so far.

## Conclusion

This fix attempt did not close the full gap to `91.56%`, but it did recover a larger part of the lost performance. The current best reduced-MNIST number in the Rust implementation is now `52.20%` under the explicit E/I current-based model with converted reference-like numerics plus an explicit rest phase.

That leaves the remaining gap concentrated in the parts that are still not reference-equivalent:

1. The Rust code still uses a pair-rule STDP approximation instead of the Diehl-Cook trace-based update.
2. Synaptic input is still a direct current pulse, not a `ge` / `gi` conductance state with `1 ms` / `2 ms` decay.
At this point the highest-value next change is to implement the reference STDP rule on top of the newly aligned parameterization, rather than continuing to tune the old ad hoc scales.