# Improvement Notes

## Exact post-update renormalization

I replaced the earlier forward-pass gain-scaling approximation with exact post-update renormalization of the stored plastic feedforward weights.

The implementation now works like this:

1. Apply STDP updates to the plastic input-to-output synapses.
2. For each postsynaptic neuron touched by those updates, collect its incoming plastic feedforward weights.
3. Compute the L1 or L2 norm of that stored weight vector.
4. Rescale the stored weights so the incoming feedforward vector has unit norm.
5. Keep the fixed lateral inhibitory synapses outside the renormalization step.
6. Remove the old dynamic division by the current norm during the forward pass.

So the code is no longer doing an *effective normalization* of momentary drive. It is now enforcing a real synaptic budget on the stored feedforward weights.

That difference matters:

- the old version normalized gain
- the new version forces sibling synapses on one neuron to compete for a fixed budget

That second property is the important one. It is what the earlier approximation only imitated indirectly.

## Measured results

I reran the reduced experiment (`train=2000`, `mark=500`, `test=1000`, `outputs=200`) five times for each combination of sparse/dense connectivity and L1/L2 normalization.

| Connectivity | Normalization | Mean accuracy | Accuracy range | Mean zero classes | Mean avg firing |
| --- | --- | ---: | ---: | ---: | ---: |
| sparse (`0.05`) | `l1` | `31.78%` | `29.20%` - `35.40%` | `0.00` | `0.20` |
| sparse (`0.05`) | `l2` | `37.46%` | `34.00%` - `39.60%` | `0.00` | `0.32` |
| dense (`1.0`) | `l1` | `31.42%` | `24.70%` - `34.50%` | `0.00` | `0.33` |
| dense (`1.0`) | `l2` | `14.72%` | `10.30%` - `16.80%` | `0.00` | `0.47` |

For comparison against the earlier forward-pass gain-scaling approximation:

- sparse `l1`: `31.64%` -> `31.78%`
- sparse `l2`: `39.88%` -> `37.46%`
- dense `l1`: `12.42%` -> `31.42%`
- dense `l2`: `18.12%` -> `14.72%`

So the change mattered much more than I expected, especially in the dense regime.

## What changed in the diagnosis

### 1. Exact renormalization is not a minor implementation detail

I originally treated forward-pass gain scaling as a reasonable low-cost proxy for real normalization. The new results show that this was too optimistic.

In the sparse regime, the difference is modest. In the dense regime, it changes the conclusion completely. The earlier approximation missed the main effect of true post-update budget competition.

### 2. Dense connectivity is not inherently broken in this codebase

The most important new result is dense + `l1`:

- mean accuracy `31.42%`
- accuracy range `24.70%` to `34.50%`
- no empty classes in any run
- mean average firing `0.33`
- mean largest assigned class `38.6` neurons

That is still far below the literature, but it is no longer a chance-level collapse. The earlier blanket statement that "dense connectivity still fails even with normalization" is therefore too broad.

A better statement is:

- dense connectivity fails with the earlier gain-scaling surrogate
- dense connectivity becomes viable once the model enforces a true L1 budget on the stored feedforward weights

This is a real architectural change in the diagnosis, not just a small metric adjustment.

### 3. Dense L2 is still a bad fit for the current homeostasis

Dense + `l2` goes in the opposite direction:

- mean accuracy `14.72%`
- mean average firing `0.47`, the highest of the four conditions
- mean largest assigned class `45.6` neurons

So exact renormalization alone is not sufficient. The norm type still interacts strongly with the current inhibition, threshold adaptation, and STDP rates.

The simplest interpretation is that unit L2 renormalization leaves too much effective dense drive under the current hyperparameters, while unit L1 renormalization regularizes dense shared input statistics much more aggressively.

### 4. Sparse L2 is still the best raw reduced-run baseline

Sparse + `l2` remains the best condition by mean accuracy at `37.46%`.

Sparse + `l1` stays stable but not especially strong, and exact renormalization barely changes its aggregate result relative to the earlier approximation.

So the answer to "what should I use right now for the best reduced-run accuracy" is still:

- sparse connectivity
- `--normalization l2`

But the answer to "what changed the architectural diagnosis the most" is now:

- dense connectivity with exact `l1` renormalization

## Additional problems identified

### 1. The `--rng-seed` flag is still unused

The CLI exposes `--rng-seed`, but the code still samples through `rand::random` in the noise path, the Poisson input path, and weight initialization.

That means these experiments are not actually reproducible from the command line even though the flag exists. This is a practical problem for evaluating small differences between normalization modes.

### 2. The normalization target is hard-coded to unit norm

Right now the code couples two separate choices into one:

- which norm geometry to use
- what absolute post-renormalization gain to give the feedforward weights

That is likely part of why dense L1 and dense L2 diverge so strongly. The next step should probably separate `norm type` from `target norm` or from an explicit post-renormalization gain factor.

### 3. Near-dead neurons remain in every condition

Mean low-firing warnings per run were:

- sparse `l1`: `1.6`
- sparse `l2`: `0.4`
- dense `l1`: `2.4`
- dense `l2`: `2.8`

So even the better conditions still leave a few neurons chronically weak. The model still lacks a clean mechanism to revive neurons that fall behind.

### 4. There is still no slow synaptic scaling

Exact L1 renormalization fixes within-neuron budget competition, but dense L2 shows that threshold homeostasis alone is still not enough to keep all regimes in a healthy activity band.

A slower multiplicative scaling mechanism would address a different problem from renormalization:

- renormalization shapes the receptive field budget
- scaling keeps long-term activity in range

The current code now has the first property for feedforward weights, but still not the second.

### 5. The earlier forward-pass surrogate was methodologically misleading

This is itself an important lesson from the experiment. The gain-scaling approximation was good enough to suggest that normalization might matter, but it was not a reliable proxy for actual renormalization when making architectural claims about dense connectivity.

So the earlier dense conclusion should not be used as evidence that dense STDP is fundamentally broken in this repository.

## Updated direction

If I were continuing from the current state, I would do the following in order:

1. Keep sparse + `--normalization l2` as the best current reduced-run baseline.
2. Treat dense + `--normalization l1` as the most promising dense starting point, because exact renormalization now shows that dense connectivity is not fundamentally doomed in this codebase.
3. Add an explicit target norm or post-renormalization gain parameter instead of hard-coding unit norm.
4. Wire up `--rng-seed` so normalization experiments are actually reproducible.
5. Add slow synaptic scaling or another dead-neuron recovery mechanism before drawing stronger conclusions about dense `l2`.

## Bottom line

Changing from effective gain scaling to exact post-update renormalization did change the result, and in the dense L1 case it changed it a lot.

The updated takeaway is:

- exact renormalization is substantially different from the earlier approximation
- sparse `l2` is still the best raw reduced-run configuration
- dense `l1` is now viable enough to change the architectural diagnosis
- dense `l2` still fails, which means norm choice, target gain, and homeostasis are tightly coupled here

That is the conclusion I would carry forward from this round of experiments.
