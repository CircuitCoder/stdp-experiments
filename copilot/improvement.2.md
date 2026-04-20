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

## Follow-up changes carried out

I also carried out the next practical changes that fell out of this analysis.

### 1. Real MNIST can now be run separately from the bundled Fashion-MNIST files

The code now exposes:

- `--dataset {mnist,fashion}`
- `--data-path <path>`

and I downloaded a separate real MNIST copy into `data/mnist`.

That means future experiments no longer need to conflate:

- the bundled `data/` directory, which still contains Fashion-MNIST
- a true MNIST run, which can now use `--dataset mnist --data-path data/mnist`

### 2. Threshold-homeostasis settings are now tunable from the CLI

The code now exposes:

- `--neuron-tau`
- `--neuron-threshold`
- `--threshold-homeostasis-tau`
- `--threshold-homeostasis-inc`

That makes it possible to retune the threshold adaptation directly instead of treating the current hard-coded values as fixed.

### 3. Feedforward sparsity is now measured after training

The code now counts feedforward synapses whose absolute weight is below a configurable threshold:

- `--near-zero-weight-threshold`

After training, it prints:

- how many feedforward synapses remain effectively non-zero
- how many are near-zero and therefore plausible pruning candidates
- the resulting non-zero ratio across all feedforward slots

This is the effective sparsity signal you would use if you later prune near-zero synapses to recover structural sparsity.

## Real MNIST follow-up results

Once the code could point at real MNIST, I reran the dense `l1` regime on that workload instead of on Fashion-MNIST.

### Reduced real-MNIST pilots

All runs below used:

- `train=2000`, `mark=500`, `test=1000`
- `outputs=200`
- dense connectivity (`1.0`)
- `l1` exact renormalization
- post-renorm gain `1.0`
- slow scaling target `0.004`

| Configuration | Accuracy | Avg firing | Effective non-zero feedforward ratio |
| --- | ---: | ---: | ---: |
| baseline dense `l1` + scaling | `29.10%` | `0.24` | `23.26%` |
| milder threshold homeostasis: `inc=0.5`, `tau=20` | `37.20%` | `0.24` | `23.91%` |
| milder homeostasis + stronger inhibition: `-0.15` | `32.20%` | `0.17` | `29.40%` |
| stronger selectivity: inhibition `-0.2`, `inc=0.25`, `tau=50` | `32.90%` | `0.13` | `33.05%` |

### What that means

The immediate result is that the dataset mismatch was real, but it was not the whole story.

Switching from Fashion-MNIST to real MNIST did not suddenly produce literature-level behavior by itself. The strongest reduced-run result on real MNIST here was `37.20%`, reached by making threshold homeostasis much milder while leaving inhibition at `-0.1`.

That is still far below the published `80%+` range. So the gap is not explained by dataset mismatch alone.

At the same time, the retuning result does matter:

- stronger inhibition did **not** help in this regime
- milder threshold homeostasis **did** help materially

So the previous suspicion was correct: the threshold adaptation constants were part of the problem.

### Larger real-MNIST follow-up

I then scaled the best reduced real-MNIST setting to:

- `train=5000`, `mark=1000`, `test=2000`
- `outputs=200`

using:

- dense `l1`
- post-renorm gain `1.0`
- slow scaling target `0.004`
- `--threshold-homeostasis-inc 0.5`
- `--threshold-homeostasis-tau 20`

That larger run reached:

- accuracy: `25.45%`
- avg firing: `0.25`
- effective non-zero feedforward ratio: `29.52%`

## Epoch-by-epoch verification

To test whether the long-run degradation was only an artifact of the *final* voting set, I added explicit epoch support and per-epoch evaluation.

The code now supports:

- `--epochs <N>`

and slices the training and marking sets by epoch, so each epoch uses its own disjoint:

- training block of length `train_length`
- marking block of length `mark_length`

That means the epoch-level results below are **not** reusing the same voting set every time. Each epoch gets a fresh training window and a fresh voting window.

### Real-MNIST epoch runs

I ran two 5-epoch MNIST experiments with:

- `train=2000` per epoch
- `mark=500` per epoch
- `test=1000`
- dense connectivity (`1.0`)
- `l1` exact renormalization
- post-renorm gain `1.0`
- slow scaling target `0.004`

The two variants were:

1. baseline dense `l1` + scaling
2. dense `l1` + scaling + milder threshold homeostasis (`inc=0.5`, `tau=20`)

### Baseline epoch curve

| Epoch | Accuracy | Avg firing | Effective non-zero feedforward ratio | Mean gain |
| --- | ---: | ---: | ---: | ---: |
| 1 | `27.70%` | `0.25` | `24.08%` | `1.1589` |
| 2 | `22.40%` | `0.26` | `27.05%` | `1.3490` |
| 3 | `25.60%` | `0.25` | `30.43%` | `1.5679` |
| 4 | `15.40%` | `0.25` | `34.57%` | `1.8269` |
| 5 | `23.30%` | `0.26` | `37.86%` | `2.1226` |

### Milder-homeostasis epoch curve

| Epoch | Accuracy | Avg firing | Effective non-zero feedforward ratio | Mean gain |
| --- | ---: | ---: | ---: | ---: |
| 1 | `33.70%` | `0.24` | `24.28%` | `1.1612` |
| 2 | `34.00%` | `0.26` | `27.50%` | `1.3544` |
| 3 | `32.60%` | `0.25` | `31.52%` | `1.5767` |
| 4 | `23.90%` | `0.25` | `35.36%` | `1.8410` |
| 5 | `26.50%` | `0.26` | `38.76%` | `2.1456` |

### What the epoch runs prove

This answers the question directly.

The accuracy **does** regress after longer training even when:

- each epoch gets a fresh voting set
- each epoch is evaluated immediately after its own training block
- the final voting set is not reused across all epochs

So the long-run degradation is **not** explained by the final voting assignment alone.

In other words, the problem is not just that the last classifier readout is bad. The representation itself is drifting into a worse state as training continues.

### What drifts while accuracy falls

The most informative pattern is that three things happen together:

1. average firing stays almost flat, around `0.24` to `0.26`
2. the mean post-renormalization gain rises steadily from about `1.16` to about `2.12`
3. the effective non-zero feedforward ratio rises steadily from about `24%` to about `38%`

That is a strong sign that the network is not simply "going silent" or "spiking too much."

Instead, threshold homeostasis is holding the gross firing rate in a narrow band while the feedforward representation becomes:

- stronger in aggregate
- less sparse
- less class-selective

So the long-run failure looks more like **representation drift / densification / loss of specialization** than classical supervised overfitting.

### Interpretation of the larger run

This is an important negative result.

The reduced-run retune improved the short experiment, but the advantage did not survive a larger training schedule. In fact, the larger run became worse.

That points to a deeper remaining issue:

- the system is not only under-tuned at a short horizon
- it is still drifting into a poorer operating point over longer training

So after these follow-up changes, the leading explanation for the literature gap becomes more specific:

1. the repo originally had a task/data mismatch, and that is now corrected for follow-up experiments
2. the threshold homeostasis was indeed too aggressive, and milder settings help on short runs
3. per-epoch verification shows the long-run accuracy drop is real even with fresh voting sets, so the problem is not only in the final readout stage
4. the remaining gap is now primarily a stability/homeostasis issue, with gains and effective connectivity density rising while class selectivity worsens

That is the strongest updated conclusion from this round.

## Intra-epoch verification with multiple classification sets

To probe classical overfitting more directly on the current best aligned configuration, I added two more evaluation controls:

- `--verify-interval <N>`: evaluate every `N` training samples instead of only at epoch end
- `--classification-set-count <K>`: evaluate each checkpoint against `K` disjoint classification/assignment sets instead of a single voting set

The implementation also snapshots and restores the network's transient neuron/tracker state around verification, so these extra evaluations do **not** perturb the ongoing training dynamics.

### Current aligned + rest baseline parameters

Unless noted otherwise, the aligned + explicit-rest follow-up runs in the sections below use this parameter set:

- `dataset=mnist`, `data_path=data/mnist`
- `output_num=200`
- `connection_rate=1.0`
- `normalization=l1`, `post_renorm_gain=1.95`, `slow_scaling_rate=0.0`
- `poisson_rate=0.03125`, `poisson_rate_inc=0.015625`, `least_training_firing_rate=0.007142857`
- baseline `per_sample_ticks=700`, `rest_ticks=300`
- `base_noise=0.0`
- `lateral_inhib_strength=-0.23`, `excitatory_inhibitory_strength=1.56`
- `neuron_tau=200`, `neuron_refractory_ticks=10`
- `threshold_homeostasis_tau=20000000`, `threshold_homeostasis_inc=0.003846154`
- `inhibitory_neuron_tau=20`, `inhibitory_neuron_threshold=1.0`, `inhibitory_neuron_reset=0.75`, `inhibitory_neuron_refractory_ticks=4`
- `inhibitory_threshold_homeostasis_tau=1`, `inhibitory_threshold_homeostasis_inc=0.0`
- `stdp_lr_plus=0.01`, `stdp_lr_minus=0.0001`, `stdp_tau_plus=40`, `stdp_tau_minus=40`

The overfit-check and sweep sections then vary only the evaluation budget (`epochs`, `train`, `mark`, `test`, `classification_set_count`, `verify_interval`) and whichever single knob is being probed, such as `per_sample_ticks`.

### Verification run

I ran the current aligned + explicit-rest configuration with:

- `epochs=3`
- `train=2000` per epoch
- `verify_interval=1000`
- `classification_set_count=3`
- `mark=100` per classification set
- `test=200`
- `per_sample_ticks=700`
- `rest_ticks=300`

Log: `copilot/tmp/mnist_overfit_check.log`

### Checkpoint curve

| Epoch | Checkpoint | Mean accuracy | Classification-set range | Mean avg firing | Effective non-zero feedforward ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `1000 / 2000` | `29.17%` | `26.00%` - `31.00%` | `0.14` | `66.39%` |
| 1 | `2000 / 2000` | `42.00%` | `36.50%` - `45.50%` | `0.13` | `57.11%` |
| 2 | `1000 / 2000` | `41.67%` | `38.50%` - `45.00%` | `0.16` | `45.01%` |
| 2 | `2000 / 2000` | `44.50%` | `41.00%` - `48.00%` | `0.15` | `37.61%` |
| 3 | `1000 / 2000` | `44.50%` | `42.50%` - `46.50%` | `0.15` | `31.15%` |
| 3 | `2000 / 2000` | `51.17%` | `49.00%` - `53.50%` | `0.13` | `34.92%` |

### What this says about overfitting

This run does **not** look like classical overfitting.

- Mean checkpoint accuracy rises overall from `29.17%` to `51.17%`.
- The only small dip (`42.00%` to `41.67%`) is well within the contemporaneous classification-set spread.
- Different classification sets do matter, but the spread stays in the range of roughly `2` to `9` percentage points at a checkpoint rather than explaining the whole trajectory.
- Average firing stays tightly bounded around `0.13` to `0.16`.

That means the current aligned + rest configuration is much more stable than the older dense-`l1` experiments.

The earlier dense-`l1` regime showed degradation even with fresh voting sets each epoch. By contrast, this newer run improves steadily even when every checkpoint is tested against three disjoint classification sets.

### Updated interpretation

For the current best configuration, the evidence now says:

1. There is some sensitivity to which classification set is used for neuron assignment.
2. But there is no sign yet that the model is merely memorizing a specific classification set and then collapsing on fresh ones.
3. Over the first `6000` training samples, performance is still improving on average across fresh classification sets.

So at this stage, the remaining gap to the reference looks less like readout overfitting and more like an incomplete dynamical/learning-rule match. The trace-based STDP rule is still the highest-priority missing piece.

## Quick sweep: longer presentation versus longer training

To probe the "still underfitting?" hypothesis without waiting for a full 2k-train x multi-set x 2x/3x/4x sweep to finish, I ran a smaller comparison that kept the same aligned + explicit-rest parameterization and the same three-set verification method, but reduced the sample counts:

This quick sweep is **not directly comparable** to the earlier `51.17%` overfit-check endpoint. That earlier result came from a much larger run with:

- `epochs=3`
- `train=2000` per epoch, so `6000` training samples total by the final checkpoint
- `mark=100` per classification set
- `test=200`

By contrast, the quick sweep below uses only `250` training samples per epoch, so its `1x` epoch point is intentionally a much earlier and noisier checkpoint.

- `train=250` per epoch
- `mark=50` per classification set
- `test=100`
- `classification_set_count=3`
- baseline presentation: `per_sample_ticks=700`, `rest_ticks=300`

### Presentation-cycle sweep

Here I kept training at one epoch and only changed how long each image was presented.

| Presentation multiplier | `per_sample_ticks` | Mean accuracy | Classification-set range | Mean avg firing |
| --- | ---: | ---: | ---: | ---: |
| `1x` | `700` | `21.00%` | `20.00%` - `22.00%` | `0.09` |
| `2x` | `1400` | `32.67%` | `31.00%` - `35.00%` | `0.33` |
| `3x` | `2100` | `30.00%` | `30.00%` - `30.00%` | `0.33` |
| `4x` | `2800` | `35.33%` | `31.00%` - `41.00%` | `0.47` |

Logs:

- `copilot/tmp/mnist_cycle_quick_2x.log`
- `copilot/tmp/mnist_cycle_quick_3x.log`
- `copilot/tmp/mnist_cycle_quick_4x.log`

### Epoch sweep

Here I kept the presentation length at `700` ticks and increased the number of epochs. This came from a single 4-epoch run, so the 1x/2x/3x/4x epoch points are all on the same training trajectory.

| Epoch multiplier | Epochs | Mean accuracy | Classification-set range | Mean avg firing |
| --- | ---: | ---: | ---: | ---: |
| `1x` | `1` | `21.00%` | `20.00%` - `22.00%` | `0.09` |
| `2x` | `2` | `7.67%` | `6.00%` - `9.00%` | `0.10` |
| `3x` | `3` | `21.33%` | `20.00%` - `24.00%` | `0.09` |
| `4x` | `4` | `33.33%` | `26.00%` - `44.00%` | `0.10` |

Log: `copilot/tmp/mnist_epoch_quick_sweep.log`

### Interpretation

- The apparent drop back to about `21%` is **not** a regression from the earlier `51.17%` run. It is the baseline of a deliberately much smaller and earlier probe.
- Longer presentation time helps in this quick probe: `2x`, `3x`, and `4x` all beat the `1x` baseline, with `4x` best.
- Longer training is much less smooth at this tiny sample budget. The 2-epoch point collapses, the 3-epoch point recovers to baseline, and the 4-epoch point is clearly above baseline.
- That noisy small-data epoch curve should not be over-read. The earlier larger run with `train=2000`, `mark=100`, `test=200`, and three classification sets still showed a cleaner epoch trend: `42.00% -> 44.50% -> 51.17%` for `1x -> 2x -> 3x` epochs.

So the current evidence still leans more toward "not trained enough yet" than toward overfitting, but the extra exposure seems more reliable when it comes from longer per-sample presentation than from simply extending the number of epochs on a tiny training budget.

## Full-size follow-up: 4x presentation length

I then reran the most promising quick-sweep variant at the earlier full-size evaluation budget: keep the same aligned + explicit-rest baseline, but increase `per_sample_ticks` from `700` to `2800` while leaving everything else unchanged.

Run configuration:

- `epochs=1`
- `train=2000`
- `mark=100` per classification set
- `test=200`
- `verify_interval=1000`
- `classification_set_count=3`
- `per_sample_ticks=2800`
- `rest_ticks=300`

Log: `copilot/tmp/mnist_cycle_full_4x.log`

For comparison, the directly relevant baseline is the epoch-1 aligned + rest run from the earlier multi-set verification section above, which used the same evaluation budget but the baseline presentation length `per_sample_ticks=700`.

| Checkpoint | Baseline `700` ticks | `4x` presentation `2800` ticks | Delta |
| --- | ---: | ---: | ---: |
| `1000 / 2000` mean accuracy | `29.17%` | `38.67%` | `+9.50` points |
| `2000 / 2000` mean accuracy | `42.00%` | `42.50%` | `+0.50` points |
| `1000 / 2000` mean avg firing | `0.14` | `0.46` | `+0.32` |
| `2000 / 2000` mean avg firing | `0.13` | `0.32` | `+0.19` |
| `1000 / 2000` effective non-zero ratio | `66.39%` | `30.06%` | `-36.33` points |
| `2000 / 2000` effective non-zero ratio | `57.11%` | `29.48%` | `-27.63` points |

Final checkpoint classification-set accuracies:

- baseline `700` ticks: `36.50%`, `44.00%`, `45.50%`
- `4x` presentation `2800` ticks: `40.50%`, `44.00%`, `43.00%`

### Interpretation

- The longer presentation window clearly improves the early trajectory: after the first `1000` training samples, mean accuracy rises from `29.17%` to `38.67%`.
- By the end of the epoch, however, the gain is almost gone: `42.00%` versus `42.50%`.
- The `4x` run drives much more activity (`0.46` then `0.32` spikes/sample versus `0.14` then `0.13`) and also yields a much sparser effective feedforward matrix by the near-zero threshold used here.

So the full-size rerun does **not** support the simple story that "just show each image much longer" is the main missing ingredient. It helps the first half of training substantially, but at least over one `2000`-sample epoch it does not materially improve the final verification accuracy.

## Full-size follow-up: 16 epochs with short presentation

To test the alternative hypothesis directly, I kept the short-presentation aligned + rest baseline (`per_sample_ticks=700`, `rest_ticks=300`) and extended the full-size run to `16` epochs with three disjoint classification sets per epoch-end evaluation.

Run configuration:

- `epochs=16`
- `train=2000` per epoch
- `mark=100` per classification set
- `test=200`
- `classification_set_count=3`
- `per_sample_ticks=700`
- `rest_ticks=300`

Log: `copilot/tmp/mnist_epochs16_baseline.log`

### Epoch curve

| Epoch | Mean accuracy | Classification-set range | Mean avg firing | Effective non-zero feedforward ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 | `36.50%` | `34.00%` - `38.50%` | `0.12` | `56.77%` |
| 2 | `39.33%` | `35.00%` - `41.50%` | `0.14` | `36.39%` |
| 3 | `46.17%` | `42.50%` - `51.50%` | `0.11` | `34.39%` |
| 4 | `45.33%` | `41.00%` - `53.50%` | `0.09` | `32.58%` |
| 5 | `49.83%` | `41.50%` - `54.50%` | `0.08` | `32.46%` |
| 6 | `45.67%` | `40.00%` - `50.50%` | `0.08` | `35.17%` |
| 7 | `40.83%` | `39.50%` - `41.50%` | `0.08` | `31.86%` |
| 8 | `44.50%` | `39.00%` - `48.00%` | `0.08` | `31.96%` |
| 9 | `37.67%` | `34.50%` - `43.50%` | `0.08` | `32.91%` |
| 10 | `41.67%` | `39.50%` - `45.00%` | `0.08` | `33.95%` |
| 11 | `38.33%` | `36.50%` - `40.50%` | `0.08` | `32.43%` |
| 12 | `46.00%` | `42.50%` - `48.00%` | `0.08` | `32.36%` |
| 13 | `47.83%` | `46.50%` - `49.00%` | `0.08` | `33.38%` |
| 14 | `50.00%` | `45.00%` - `56.00%` | `0.08` | `33.73%` |
| 15 | `48.50%` | `45.00%` - `52.00%` | `0.08` | `33.18%` |
| 16 | `50.83%` | `49.50%` - `53.00%` | `0.08` | `34.53%` |

### Interpretation

- This run does **not** collapse over longer training. The mean accuracy oscillates, but it recovers repeatedly and finishes at `50.83%`.
- The final `50.83%` is essentially on the same scale as the earlier `51.17%` 3-epoch endpoint, even though the classification windows are different and the `--rng-seed` CLI knob is still unused.
- The best late-epoch regime is clearly better than the early one: epochs `12` through `16` stay in the `46%` to `51%` band, whereas epochs `1` and `2` are only `36.50%` and `39.33%`.
- Mean firing falls quickly and then stabilizes near `0.08`, while the effective non-zero feedforward ratio settles in the low-`30%` range after the early pruning phase.

So the strongest result from this run is that the short-presentation aligned + rest configuration can remain competitive over a much longer horizon than the old dense-`l1` regime. The evidence still points more toward a noisy, capacity-limited long training process than toward classical overfitting.

## Capacity and connectivity comparison to the reference implementation

The current full-size Rust run is still structurally smaller than the original Diehl & Cook reference.

### Neuron counts

| Model | Input channels | Excitatory neurons | Inhibitory neurons | Total non-input neurons |
| --- | ---: | ---: | ---: | ---: |
| Current Rust run | `784` | `200` | `200` | `400` |
| Reference implementation | `784` | `400` | `400` | `800` |

So yes: this run uses **half as many excitatory neurons** as the reference (`200` vs `400`), and therefore half as many inhibitory neurons as well.

### Synapse counts and connectivity pattern

Both implementations use the same high-level E/I wiring pattern:

- dense input -> excitatory connectivity
- one-to-one excitatory -> inhibitory connectivity
- dense inhibitory -> excitatory connectivity without self-pairs

But because the Rust run uses only `200` excitatory neurons, the realized synapse counts are much smaller.

| Model | Input -> E | E -> I | I -> E | Total |
| --- | ---: | ---: | ---: | ---: |
| Current Rust run | `784 * 200 = 156800` | `200` | `200 * 199 = 39800` | `196800` |
| Reference implementation | `784 * 400 = 313600` | `400` | `400 * 399 = 159600` | `473600` |

So the current Rust run has about `41.6%` as many modeled synapses as the reference.

### Connectivity rate

- In this Rust run, `connection_rate=1.0`, so the input -> excitatory projection is fully dense.
- In the reference random connection generator, `pConn['ee_input'] = 1.0`, so the `XeAe` projection is also fully dense.
- The recurrent E/I pattern also matches the reference structure: one-to-one `AeAi` and dense-without-self `AiAe`.

So the main structural mismatch here is **not** connectivity sparsity. It is primarily the **smaller population size** (`200` excitatory instead of `400`).

### Initial weight assignment

The initial feedforward weights are also not identical.

Current Rust run:

- each input -> excitatory synapse starts from `STDPSynapse::new_rand(feedforward_max_weight, 0.0, ...)`
- with this configuration, `feedforward_max_weight` resolves to `1.95`
- because `l1` renormalization is enabled, the network then renormalizes each excitatory neuron's incoming feedforward vector to a total `l1` gain of `1.95`
- recurrent `E -> I` and `I -> E` weights are fixed constants in this run because `new_rand(max=min)` is called with `1.56` and `-0.23`

Reference implementation:

- the random connection generator initializes `XeAe` weights as `(rand + 0.01) * 0.3`, so roughly in `[0.003, 0.303]`
- the training code then calls `normalize_weights()`, which rescales each excitatory column so its total incoming `XeAe` weight sums to `78.0`
- recurrent `AeAi` weights are fixed at `10.4`
- recurrent `AiAe` weights are fixed at `17.0` with zero diagonal

The absolute weight values are not directly comparable one-for-one because the reference is conductance-based while the Rust model is currently current-based with converted numerics. But structurally, the biggest differences are:

1. the Rust run uses half as many excitatory columns
2. the feedforward normalization target is different (`1.95` current-scale gain here versus column sum `78.0` in the reference conductance model)
3. the Rust input weights are re-randomized each run and `--rng-seed` is still unused

## Reference-side 200-neuron checkpoint run

I also ran a Brian reference-side variant with `n_e = 200` and `n_i = 200` under `copilot/tmp/ref200/`.

Important caveat: the default reference training schedule is `180000` accepted training samples (`60000 * 3`), and in this environment that path is too slow to finish within a reasonable interactive session because the Brian loop may retry a sample several times before accepting it. So instead of waiting for the full `180000`-sample training run, I used the **first saved checkpoint** at `10000` accepted training samples:

- training checkpoint: `copilot/tmp/ref200/weights/XeAe10000.npy`
- theta checkpoint: `copilot/tmp/ref200/weights/theta_A10000.npy`
- test activity: `copilot/tmp/ref200/activity/resultPopVecs10000.npy`
- evaluation log: `copilot/tmp/ref200_eval.log`

The resulting Brian reference-side accuracy for that `10000`-sample `200`-neuron checkpoint was:

| Reference variant | Training state | Test set | Accuracy |
| --- | --- | ---: | ---: |
| Brian `n_e = 200`, `n_i = 200` | first saved checkpoint at `10000` accepted training samples | `10000` examples | `32.53%` |

### Interpretation

- This is a **real Brian reference-side run** at `200` excitatory neurons, not a Rust approximation.
- It is **not** the same thing as a fully trained `180000`-sample reference run, so it should not be compared directly to the canonical `91.56%` pretrained/reference evaluation.
- It does show that, at least at the first `10000`-sample checkpoint, the smaller reference-side network is still far from convergence and is only at `32.53%`.
- Relative to the current Rust `200`-output aligned + rest results, this means the Rust model is no longer obviously losing to the reference on the specific combination of **small output layer plus early training checkpoint**. The remaining comparison gap is now mostly about **training budget and learning dynamics**, not just neuron count alone.

## Reference-side 200-neuron matched-budget follow-up

I then continued that same Brian `n_e = 200`, `n_i = 200` run from the first `10000`-sample checkpoint up to the same total accepted training budget as the Rust 16-epoch run:

- Rust 16-epoch run: `16 * 2000 = 32000` training samples
- Brian continuation: `10000 -> 32000` accepted training samples via `copilot/tmp/ref200_train_resume_10000_to_32000.log`

The continuation produced explicit checkpoints at `20000`, `30000`, and final `32000` total accepted samples. After normalizing the checkpoint filenames, the final pair used for the corrected evaluation was:

- feedforward weights: `copilot/tmp/ref200/weights/XeAe32000.npy`
- thresholds: `copilot/tmp/ref200/weights/theta_A32000.npy`
- corrected test log: `copilot/tmp/ref200_test_32000_fixed.log`
- corrected evaluation log: `copilot/tmp/ref200_eval_32000_fixed.log`

### Important correction

The first attempt to evaluate the final continuation checkpoint gave a bogus `9.95%` result. That number should be discarded.

Cause:

- the temporary continuation script accidentally changed `def save_connections(ending = '')` into `def save_connections(ending = '10000')`
- as a result, the final continuation save overwrote `XeAe10000.npy` with the final feedforward weights
- at the same time, the final theta values were still saved to `theta_A.npy`
- testing `XeAe.npy` together with `theta_A.npy` therefore used a **mismatched pair**: old `10000`-sample feedforward weights with final theta values

After fixing the naming and rerunning the test/evaluation using the correct final `32000` checkpoint pair, the real matched-budget Brian result was:

| Model | Training budget | Test set | Accuracy |
| --- | --- | ---: | ---: |
| Brian `n_e = 200`, first checkpoint | `10000` accepted samples | `10000` examples | `32.53%` |
| Brian `n_e = 200`, corrected final checkpoint | `32000` accepted samples | `10000` examples | `59.13%` |
| Rust aligned + rest baseline | `32000` training samples (`16 * 2000`) | three `200`-example classification sets | `50.83%` mean |

### Interpretation

- The Brian `200`-neuron network improves substantially with longer training at fixed width: `32.53% -> 59.13%`, a gain of `26.60` points from `10000` to `32000` accepted samples.
- At the matched `32000`-sample budget, the Brian reference-side model is now **better** than the current Rust aligned + rest run: `59.13%` versus `50.83%`, a gap of `8.30` points.
- The corrected final Brian evaluation does **not** show the catastrophic assignment collapse implied by the discarded `9.95%` artifact. The final assignment vector still labels `194/200` neurons, with only `6` unassigned.
- So the smaller `200`-neuron population size is not, by itself, enough to explain the Rust/reference gap. Once the Brian side is trained to the same sample budget, it still outperforms the current Rust model by a meaningful margin.
- That shifts the remaining explanation back toward the known dynamical mismatches: conductance vs current dynamics, the trace-based Brian STDP rule vs the current Rust pair rule, and the remaining differences in competition / threshold adaptation behavior.

## Reference-side 400-neuron train-from-scratch reproduction path

I then tried to reproduce the reference readme's fresh-training claim for the original `400`-neuron network rather than the shipped pretrained `91.56%` path.

### What was blocking the run

The existing compatibility runner disabled Brian's weave-backed code paths to avoid an immediate crash in the legacy threshold implementation.

That made evaluation easy, but it also forced the slow Python fallback path for training. At `n_e = 400`, that would make the full `180000`-sample reference schedule impractically slow in this environment.

The working path turned out to be:

- restore an old SciPy that still ships `scipy.weave`
- enable Brian codegen thresholds so the threshold path no longer dies on import
- keep the rest of the training script as close as possible to the original reference
- redirect outputs into a scratch directory instead of overwriting the bundled pretrained files

### Temporary runner and environment fix

I added a temporary runner at `copilot/tmp/run_ref_training_codegen.py` that:

- patches the reference script into train or test mode without editing the original file
- disables plotting and spike recording for long runs
- enables `usecodegenthreshold = True`
- rewrites the broken Brian compiler option list from `['-ffast-math -march=native']` to `['-ffast-math', '-march=native']`
- writes new checkpoints under a caller-specified scratch directory

I also staged an isolated SciPy `0.18.1` Python-2 wheel under `copilot/tmp/py2scipy018/`, because the local `scipy 1.2.3` install no longer exposes `scipy.weave`.

With that overlay active, the Brian logs now show the compiled path is live again:

- `Using codegen CThreshold`
- `Using codegen CStateUpdater`
- `Using experimental C STDP class`
- `Using new C based propagation function`

### Runtime probe

Short `400`-neuron train-from-scratch probes now complete successfully.

Warm-cache probe:

- training mode
- `n_e = 400`, `n_i = 400`
- `num_examples = 200`
- wall-clock time: `89.81s`

That is about `0.45` seconds per accepted training sample in steady state, which implies the full original training budget of `180000` accepted samples will take roughly `22` to `23` hours in this environment, plus later test/evaluation time.

### Full run result

The real fresh-training `400`-neuron run completed successfully with the full original schedule:

- `num_examples = 60000 * 3 = 180000`
- scratch output directory: `copilot/tmp/ref400_full_codegen/`
- training log: `copilot/tmp/ref400_full_codegen_train.log`

The final training log ends with:

- `save results`
- `save theta`
- `save connections`

The scratch directory contains the final trained checkpoint files:

- `weights/XeAe.npy`
- `weights/theta_A.npy`
- recurrent random matrices under `random/`

I then ran a fresh `10000`-example test-mode pass against those final weights and wrote the activity dump to the same scratch directory:

- test log: `copilot/tmp/ref400_full_codegen_test.log`
- saved activity: `activity/resultPopVecs10000.npy`, `activity/inputNumbers10000.npy`
- wall-clock time: `4361.91s` (about `72.7` minutes)

Finally, I evaluated that output with a small wrapper around the stock reference evaluator (`copilot/tmp/run_ref_evaluation.py`).

The stock `10000`-assignment / `10000`-test reference evaluation reports:

- **fresh `400`-neuron train-from-scratch accuracy: `89.04%`**

That essentially reproduces the readme's claim that a fresh run should land at "around `89%`" with the original parameters.

It is still below the shipped pretrained reference result of `91.56%`, but only by `2.52` percentage points.

Important caveat: this is the stock reference "simple demo" evaluation path, which uses the same `10000`-sample test-mode dump both to derive neuron assignments and to score the final predictions. The paper's stricter protocol instead derives assignments from a separate `60000`-sample training-set inference pass, which was not rerun here.

## Rust-side Brian-style full-pass follow-up

To test the same overall training budget on the Rust implementation, I first had to relax one scheduler assumption in [src/main.rs](/root/workspace/stdp-experiments/src/main.rs): the old CLI required the entire `train_length * epochs` budget plus all assignment/mark windows to fit into a single disjoint `60000`-sample training/validation split.

That made a Brian-style `180000`-presentation run impossible, even before evaluation.

I changed the Rust CLI so it can now:

- cycle over the loaded training pool with `--cycle-train-pool`
- choose whether assignment sets come from the validation split or the test split via `--mark-split {validation,test}`

That is enough to represent a Brian-like final-evaluation schedule such as:

- training presentations: `180000`
- assignment set: `10000`
- test set: `10000`
- assignment source: test split, matching the reference simple-demo style more closely than the previous validation-only Rust path

### Short release-mode pilots

I then ran two aligned `400`-output Rust pilots with the same reference-shaped parameter set used in the earlier aligned+rest experiments, except with:

- `output_num=400`
- `mark_split=test`
- `cycle_train_pool=true`

Results:

| Train / mark / test | Accuracy | Mean firing | Assigned class neurons | Wall time |
| --- | ---: | ---: | --- | ---: |
| `20 / 20 / 50` | `10.00%` | `0.05` | `[4, 0, 13, 9, 7, 0, 0, 0, 3, 364]` | `52.2s` |
| `100 / 100 / 100` | `12.00%` | `0.06` | `[25, 0, 4, 4, 4, 0, 0, 13, 1, 349]` | `4m11s` |

### Interpretation

- The new Rust scheduling path works end-to-end, so the earlier `60000`-sample budgeting restriction is no longer the blocker.
- But the dynamics are still very poor in the Brian-sized `400`-output setting: both pilots remain near chance and collapse most neurons onto a single class (`9` in these runs).
- The firing-rate summaries are also still extremely sparse (`0.05` to `0.06` average spikes per neuron per test sample), which is consistent with under-specialized or collapsed representations rather than a healthy early-learning regime.

### Full-run feasibility in this environment

The `100 / 100 / 100` release-mode pilot processed `300` total presentations in about `251` seconds, or roughly `0.84` seconds per presentation.

A Brian-matched Rust schedule would need roughly:

- `180000` training presentations
- `10000` assignment presentations
- `10000` test presentations
- total: about `200000` presentations

At the observed pilot throughput, that projects to roughly `46` hours of wall time. Even the smaller pilot implies that a full Brian-matched Rust run here is a **multi-day job**, not something that can be completed interactively in this session.

So the current Rust result is:

- the code can now express the Brian-style schedule
- short aligned `400`-output pilots still look qualitatively bad
- the real blocker to a literal full comparison is now runtime plus the remaining dynamics gap, not CLI expressiveness