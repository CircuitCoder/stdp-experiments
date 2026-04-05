# Improvement Notes 2

## What the remaining gaps meant

When I wrote that the strongest remaining gaps were the absence of slow synaptic scaling, the lack of a tunable post-renormalization gain, and the strong interaction between norm choice and the current inhibition/threshold homeostasis, I meant three separate but coupled problems.

### 1. Slow synaptic scaling

Exact post-update renormalization already enforces a fixed feedforward weight budget per neuron, but it only controls the *shape* and total norm of that neuron's incoming plastic weight vector.

It does **not** solve the slower question:

- is this neuron chronically too quiet?
- is this neuron chronically too active?
- should this neuron receive a bit more or a bit less total excitatory drive over long training windows?

That is what slow synaptic scaling is for.

In practical terms, the right object to scale is no longer the raw weight vector directly, because exact renormalization would just undo any naive multiplicative weight update. Once the code enforces

$$
\mathbf{w}_j \leftarrow g_j \frac{\tilde{\mathbf{w}}_j}{\|\tilde{\mathbf{w}}_j\|_p + \varepsilon}
$$

the useful slow variable is $g_j$, the post-renormalization gain for postsynaptic neuron $j$.

So in this repository, slow synaptic scaling should mean:

1. Measure each neuron's firing rate on a slow timescale.
2. Keep an exponential moving average $\hat r_j$ of that rate.
3. Move the neuron's post-renormalization gain $g_j$ up or down multiplicatively based on the error to a target rate $r^*$.

The update rule I implemented is:

$$
\hat r_j \leftarrow (1 - \alpha)\hat r_j + \alpha r_j
$$

$$
g_j \leftarrow \operatorname{clip}\left(g_j \exp\left(\eta (r^* - \hat r_j)\right), g_{\min}, g_{\max}\right)
$$

where:

- $r_j$ is the observed firing rate for neuron $j$ on the current training sample, measured in spikes per tick
- $\alpha$ is the EMA smoothing factor
- $\eta$ is the slow scaling rate
- $r^*$ is the target rate
- $g_{\min}, g_{\max}$ clamp the gain to a stable range

This preserves the neuron's learned receptive-field *shape* while still changing its total incoming feedforward strength over long timescales.

### 2. Tunable post-renormalization gain

Before this change, the code always renormalized to unit norm. That couples two choices that should be independent:

1. the geometry of the norm: L1 or L2
2. the absolute strength of the resulting feedforward drive

Those are not the same question.

The norm geometry decides how the fixed budget is distributed across synapses.

- L1 tends to spread a fixed total budget across many weights
- L2 permits a different balance, usually allowing fewer weights to dominate more strongly for the same nominal norm

The post-renormalization gain decides the overall size of the feedforward current that survives after renormalization.

Without a tunable gain, a configuration may fail for a trivial reason:

- not because the norm is wrong
- not because the receptive-field competition is wrong
- but because the entire excitatory current scale is mismatched to the current inhibition and threshold dynamics

So the right normalization is not just:

$$
\mathbf{w}_j \leftarrow \frac{\tilde{\mathbf{w}}_j}{\|\tilde{\mathbf{w}}_j\|_p}
$$

but:

$$
\mathbf{w}_j \leftarrow g_j \frac{\tilde{\mathbf{w}}_j}{\|\tilde{\mathbf{w}}_j\|_p}
$$

with $g_j$ either fixed globally or adjusted slowly per neuron.

That is what I mean by a tunable post-renormalization gain.

### 3. Norm interacting with current inhibition and threshold homeostasis

The repository already has two other homeostatic/competitive mechanisms:

- fixed lateral inhibition between output neurons
- adaptive thresholds in the LIF neurons during training

Those mechanisms were implicitly tuned against the previous feedforward current statistics.

Changing the normalization norm changes those statistics even if the nominal target norm is the same.

This matters because L1 and L2 do not produce the same current landscape:

- with dense connectivity, unit L1 makes the average individual feedforward weight smaller
- unit L2 allows a different concentration pattern of strong weights
- that changes how frequently neurons cross threshold
- that changes how often recurrent inhibition is triggered
- that changes how strongly threshold homeostasis pushes back

So the system is coupled:

- norm choice changes feedforward current geometry
- post-renorm gain changes its absolute scale
- fixed inhibition responds to the resulting spike competition
- threshold homeostasis responds to the resulting spike frequency

That is why a normalization result cannot be interpreted in isolation. A "bad L2 result" may actually mean "L2 at this gain, with this inhibition weight, and this threshold adaptation speed, is a bad combined operating point."

## Execution plan

The plan I executed was:

1. Keep exact post-update renormalization as the base mechanism.
2. Add a tunable post-renormalization gain so norm geometry and total feedforward scale can be adjusted independently.
3. Implement slow synaptic scaling as a per-neuron multiplicative update on that gain, driven by a firing-rate EMA.
4. Expose the new controls on the CLI so they can be tested without code changes.
5. Add tests that verify:
   - renormalization now targets an arbitrary gain instead of hard-coded unit norm
   - slow scaling changes the target gain while preserving the feedforward budget constraint
6. Run targeted reduced evaluations instead of a full exhaustive matrix, because the new mechanism adds several extra parameter axes.

The targeted evaluation set was:

1. dense `l1` + slow scaling at gain `1.0`
Reason: this is the best dense exact-renorm regime so far; test whether slow homeostasis improves it further.

2. dense `l2` + gain `0.5` without scaling
Reason: isolate whether dense `l2` was mostly failing because the post-renorm gain was too high.

3. dense `l2` + gain `0.5` + slow scaling
Reason: test whether dense `l2` needs both reduced gain and slow homeostasis together.

4. sparse `l2` + gain `0.75` + slow scaling
Reason: test whether the best sparse regime benefits from reduced drive and slow scaling, or whether the new machinery only helps dense cases.

I used 3 runs per targeted configuration, because the 5-run exact-renorm baselines from the previous round were already available.

## What I implemented

I carried out that plan in code.

### Network-side changes

In `src/snn/network.rs` I added:

- `FeedforwardHomeostasisConfig`
- per-neuron post-renormalization gains
- per-neuron slow-scaling firing-rate EMAs
- exact post-update renormalization to an arbitrary target gain instead of unit norm
- a slow-scaling update method that adjusts gains after each training sample
- gain and EMA diagnostic summary methods

### CLI changes

In `src/main.rs` I added:

- `--post-renorm-gain`
- `--slow-scaling-rate`
- `--slow-scaling-target-rate`
- `--slow-scaling-alpha`
- `--slow-scaling-min-gain`
- `--slow-scaling-max-gain`

Training now calls the slow-scaling update after each presented training sample.

### Validation

`cargo test --release` passed after the changes.

The unit tests now verify:

- L1 renormalization to an arbitrary target gain
- L2 renormalization to an arbitrary target gain
- slow synaptic scaling changes the post-renorm gain while preserving the feedforward budget constraint

## Important calibration result

The first slow-scaling target I tried, `0.20`, was wrong for this implementation because the code measures the target in **spikes per tick**, not spikes per sample.

That immediately saturated gains to the max clamp and produced misleading pilot results.

I recalibrated the target to `0.004`, which is roughly aligned with the previously observed dense/sparse firing scales:

- dense exact-renorm baselines were around `0.33` to `0.47` spikes per sample
- with `100` ticks per sample, that corresponds to roughly `0.0033` to `0.0047` spikes per tick

That calibration step was necessary before the final targeted sweep.

## Results

### Baselines from the previous exact-renorm round

These are the reference points from the earlier 5-run exact-renorm matrix:

| Baseline | Mean accuracy | Accuracy range | Mean avg firing |
| --- | ---: | ---: | ---: |
| dense `l1` | `31.42%` | `24.70%` - `34.50%` | `0.33` |
| dense `l2` | `14.72%` | `10.30%` - `16.80%` | `0.47` |
| sparse `l2` | `37.46%` | `34.00%` - `39.60%` | `0.32` |

### Targeted 3-run sweep

Results from `copilot/tmp/homeostasis_sweep/final/summary.tsv`:

| Configuration | Mean accuracy | Accuracy range | Mean avg firing | Mean gain range |
| --- | ---: | ---: | ---: | ---: |
| dense `l1` + scaling + target `0.004` | `34.90%` | `29.70%` - `38.90%` | `0.34` | `1.01 / 1.05 / 1.13` |
| dense `l2` + gain `0.5` | `11.13%` | `10.40%` - `11.90%` | `0.44` | `0.50 / 0.50 / 0.50` |
| dense `l2` + gain `0.5` + scaling + target `0.004` | `12.40%` | `10.40%` - `14.90%` | `0.44` | `0.34 / 0.45 / 0.60` |
| sparse `l2` + gain `0.75` + scaling + target `0.004` | `38.70%` | `38.50%` - `38.90%` | `0.25` | `0.45 / 0.86 / 1.00` |

All targeted runs completed successfully.

### Interpretation

#### 1. Slow scaling helps dense `l1`

Dense `l1` improved from `31.42%` to `34.90%` mean accuracy.

That is not an enormous jump, but it is directionally important:

- the dense `l1` regime was already the strongest dense exact-renorm case
- slow scaling did not destabilize it
- the calibrated slow scaling moved gains only mildly, with mean gain around `1.05`

So for dense `l1`, the evidence says slow synaptic scaling is useful when it acts as a mild gain-correction mechanism rather than a large redistributor.

#### 2. Gain reduction alone does not rescue dense `l2`

Dense `l2` with gain `0.5` but no scaling fell to `11.13%`.

So the dense `l2` failure is not just "the gain was too large." Simply shrinking the post-renorm gain made it worse.

#### 3. Gain reduction plus slow scaling still does not rescue dense `l2`

Dense `l2` with gain `0.5` plus slow scaling improved slightly over gain-only (`12.40%` vs `11.13%`) but still remained worse than the original dense `l2` baseline (`14.72%`).

That means the current dense `l2` problem is deeper than a single gain mismatch. Under the present inhibition and threshold adaptation, `l2` appears to create the wrong feedforward current geometry for the dense regime.

#### 4. Sparse `l2` benefits from mild gain reduction plus slow scaling

Sparse `l2` with gain `0.75` and calibrated slow scaling rose from `37.46%` to `38.70%`, while mean average firing dropped from `0.32` to `0.25`.

That is the cleanest success case from this round:

- slightly better accuracy
- clearly lower activity
- no empty classes

So the best current sparse regime is now not just "sparse `l2`" but more specifically:

- sparse `l2`
- post-renorm gain around `0.75`
- slow synaptic scaling with a low target (`0.004` spikes/tick)

## What this says about the three concepts

### Slow synaptic scaling

Slow synaptic scaling is worth keeping, but only when its target is specified in the correct units and kept close to the actual firing regime.

Used correctly, it improved dense `l1` and sparse `l2`.

### Tunable post-renormalization gain

This knob is necessary. Without it, there is no clean way to tell whether a norm is bad or whether the absolute feedforward scale is wrong.

The sparse `l2` improvement depended on reducing that gain below `1.0`.

### Norm interacting with inhibition and threshold homeostasis

This interaction is real and central.

The key evidence is:

- dense `l1` + calibrated slow scaling improves
- dense `l2` + lower gain does not
- dense `l2` + lower gain + slow scaling still does not

So the norm is not just a cosmetic choice. It changes the operating point seen by the fixed inhibition and threshold homeostasis.

## Additional problems found during execution

### 1. The slow-scaling target units are easy to misread

The current target is specified in spikes per tick, while most of the result summaries in this project are written in spikes per sample.

That mismatch is easy to get wrong, and I did get it wrong on the first pass.

This should be documented more clearly or renamed in the CLI.

### 2. Slow-scaling EMA diagnostics are printed with too little precision

The logs print EMA rate statistics with only two decimals, so values around `0.004` show up as `0.00` in most summaries.

That makes the diagnostics much less informative than they should be.

### 3. `--rng-seed` is still unused

The parameter still exists, but the code continues to use `rand::random` directly throughout the pipeline.

That means these stabilization sweeps are still not reproducible from the CLI.

## Bottom line

The three mechanisms are not interchangeable.

- slow synaptic scaling is a slow gain-correction mechanism, not a replacement for renormalization
- tunable post-renormalization gain is the knob that separates norm geometry from total feedforward strength
- the norm choice really does interact with inhibition and threshold homeostasis, which is why the same gain/scaling strategy helps dense `l1` and sparse `l2` but fails on dense `l2`

The best concrete outcomes from this round are:

1. dense `l1` + calibrated slow scaling improved to `34.90%`
2. sparse `l2` + gain `0.75` + calibrated slow scaling improved to `38.70%`

The most important negative result is:

1. dense `l2` was not rescued by either lower gain alone or lower gain plus slow scaling

That is the result I would carry forward into the next round of work.