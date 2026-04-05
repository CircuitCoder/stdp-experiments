# STDP/MNIST Investigation Report

## Executive summary

The low accuracy was not caused by a single bad hyperparameter. The implementation had several correctness issues that prevented it from behaving like an STDP-based unsupervised image classifier at all:

1. The repository task description and literature target MNIST, but the checked-in `data/` files are actually Fashion-MNIST.
2. The STDP rule was using the wrong elapsed-time variable, so the exponential timing window effectively collapsed to a constant-magnitude update.
3. Spike-history trackers were not reset between samples, so inhibition and plasticity leaked across image boundaries.
4. Intrinsic threshold adaptation was still active during validation and test, even though the literature freezes thresholds during evaluation.
5. The input-rate replay criterion was based on the most active single neuron instead of total layer activity, which over-favored winner neurons.

I fixed items 2-5. Item 1 turned out to be a task/data mismatch, not a code-path bug: my earlier loader change did not switch the local workload to MNIST because the repository's checked-in `data/` files are already Fashion-MNIST.

After the fix, the reduced experiment that previously collapsed and panicked now completes normally and assigns neurons across all ten classes. On a reduced run (`train=2000`, `mark=500`, `test=1000`, `outputs=200`) the code now reaches `35.0%` instead of failing with empty classes. On a somewhat larger run (`train=5000`, `mark=1000`, `test=2000`, `outputs=400`) it reaches `39.25%`.

I then implemented the requested normalization switch with modes `none`, `l1`, and `l2`. I first tested it as a forward-pass gain-scaling approximation, then replaced that with exact post-update renormalization of the stored plastic feedforward weights. The current implementation renormalizes only plastic feedforward input synapses, not the fixed inhibitory recurrent synapses.

The final 5x experiment matrix on the reduced configuration (`train=2000`, `mark=500`, `test=1000`, `outputs=200`) with exact post-update renormalization gives this summary:

- sparse + `l1`: mean `31.78%`
- sparse + `l2`: mean `37.46%`
- dense + `l1`: mean `31.42%`
- dense + `l2`: mean `14.72%`

Those numbers changed the diagnosis materially. Sparse `l2` remains the best raw reduced-run baseline, but dense `l1` is no longer near-chance once the code enforces an actual post-update synaptic budget. The strongest remaining gaps are now the absence of slow synaptic scaling, the lack of a tunable post-renormalization gain, and the fact that the norm choice interacts strongly with the current inhibition and threshold homeostasis.

## Literature baseline

The closest reference is:

- Peter U. Diehl and Matthew Cook, *Unsupervised learning of digit recognition using spike-timing-dependent plasticity*, Frontiers in Computational Neuroscience, 2015.

Important characteristics of that reference setup:

- Original MNIST digits, not Fashion-MNIST.
- Poisson input encoding with rates proportional to pixel intensity.
- Replay of a sample with increased input intensity if total excitatory activity is too low.
- Lateral inhibition to create competition.
- Adaptive thresholds during training.
- Learning disabled and thresholds frozen during evaluation.
- Dense input connectivity from all 784 pixels to each excitatory neuron.
- STDP with an actual timing-dependent exponential window.
- Weight stabilization via weight dependence or normalization.

The paper reports roughly `82.9%` with `100` neurons after extended training and up to `95%` with much larger networks. That is the right qualitative target for a simple two-layer unsupervised STDP model on MNIST. Since this repository currently ships Fashion-MNIST data under `data/`, those numbers are not directly comparable to the runs documented below. The important point for this investigation is that the reference architecture does not rely on lateral inhibition alone; it combines inhibition with adaptive thresholds and weight-stabilizing mechanisms.

## What this repository was implementing

Before the fixes, the repository was attempting a simplified version of that idea:

- Poisson-coded pixel inputs.
- LIF output neurons with adaptive thresholds.
- Lateral inhibitory recurrent connections.
- Additive pair-based STDP on input synapses.
- Class assignment by average neuron response on a labeled holdout set.

That overall direction is reasonable. The poor results came from implementation defects plus a few large departures from the reference design. One important nuance is that the repository already has horizontal/lateral inhibition, so the remaining problem is not the total absence of competition. The problem is that the existing competition mechanism is too weak and too local in time to replace proper synaptic stabilization.

## Root causes

### 1. Dataset mismatch between task text and repository contents

The instruction file says MNIST digit recognition, but the checked-in `data/` files are actually Fashion-MNIST. I verified this from the local label file: the first ten labels are `9 0 0 3 0 2 7 2 5 5`, which matches Fashion-MNIST rather than standard MNIST.

This also means my earlier removal of `use_fashion_data()` did not switch the actual workload to MNIST. In the `mnist` crate, `use_fashion_data()` only affects the download source. Once the code is reading existing local files from `data/`, the actual dataset is determined by those files themselves.

So the issue here is not "the code accidentally trained on Fashion-MNIST because of one builder call." The issue is that the repository's bundled dataset and the task description point to different workloads.

### 2. The STDP timing window was broken

In `src/snn/synapse.rs`, potentiation and depression used the wrong elapsed-time variable:

- Potentiation used `post_last_fire` in the branch where `post_last_fire == 0`.
- Depression used `pre_last_fire` in the branch where `pre_last_fire == 0`.

So both branches always applied `exp(0)` when they triggered. That means the implementation was no longer using a decaying timing window based on spike separation. In practice, it behaved like constant-magnitude Hebbian updates, not STDP.

### 3. Learning used stale spike events

In `src/snn/network.rs`, synapse updates were driven by trackers from the previous tick instead of the actual current pre/post event pairing. That further distorted the causal timing relationship required by STDP.

The fix was to split the rule into event-specific updates:

- `on_pre_spike(post_elapsed)`
- `on_post_spike(pre_elapsed)`

This makes the current event explicit and restores the intended pre-before-post vs post-before-pre logic.

### 4. Sample boundaries were not reset

The original `reset_neurons()` only reset membrane voltage. It did not reset:

- input spike trackers
- neuron spike trackers

That means the first ticks of a new image could still "see" spike history from the previous image. In other words, the model was learning across unrelated sample boundaries.

The literature inserts a silent period between examples. Resetting tracker state between samples is the simplified equivalent that this code needed.

### 5. Evaluation was still changing thresholds

The original code disabled synaptic updates during validation/test but still allowed the adaptive threshold to keep changing inside the LIF neuron.

The literature freezes thresholds for evaluation. If thresholds continue drifting during class assignment and testing, the response statistics are not stable.

### 6. Replay criterion pushed winner neurons too hard

The original code increased input intensity until the *most active single neuron* exceeded a target firing rate.

The reference setup instead checks whether total excitatory activity is too low. Those are not equivalent:

- total-activity replay encourages the network to respond at all
- single-neuron replay encourages one neuron to dominate

That makes winner-take-all collapse more likely.

### 7. The architecture still differs materially from the reference design

Even after fixing the hard bugs, the code still has important structural differences from the literature:

- the default feedforward connectivity is sparse (`connection_rate = 0.05`), so each output neuron sees only about 39 of 784 input pixels on average
- the original code had no weight normalization; it now has exact post-update feedforward renormalization, but it still lacks slow synaptic scaling and a tunable target gain
- the neuron/synapse model is much simpler than the conductance-based reference
- inhibition is not retuned for dense connectivity

These differences explain why the post-fix accuracy is better but still well below the published results. The dense-connectivity experiments below remain especially important here: without normalization dense connectivity collapses, while with exact L1 renormalization it becomes viable but still far below the literature. That points to stabilization and homeostasis as central remaining issues, but no longer supports the claim that dense connectivity itself is fundamentally broken in this repository.

### 8. Why the existing horizontal inhibition is not sufficient

The repository does already implement horizontal (lateral) inhibition: every output neuron connects to every other output neuron through a fixed negative synapse. So it is correct to say that there is a competition mechanism in the model.

However, that mechanism is not sufficient in the current implementation for four reasons:

- the inhibitory weights are fixed and symmetric, so they suppress all neurons in essentially the same way instead of encouraging long-term specialization
- the inhibition is non-plastic, so it cannot retune itself when the feedforward connectivity or firing regime changes
- recurrent inhibition is driven by output spikes from the previous tick, so it acts as delayed feedback competition rather than an instantaneous winner-take-all clamp
- inhibition only controls who can fire together on a given sample; it does not constrain the total incoming excitatory weight budget of a neuron, so many neurons can still learn very similar receptive fields over training

This distinction matters. Lateral inhibition is good at reducing simultaneous activity. It is not, by itself, a guarantee that different neurons will learn different prototypes. Long-term specialization usually also requires weight dependence, normalization, scaling, or some other explicit synaptic-budget mechanism.

The dense-connectivity experiments are direct evidence that the existing inhibition is not enough on its own. Dense connectivity collapses without normalization, and dense `l2` still performs poorly even with exact renormalization. Dense `l1` does improve markedly once a true synaptic budget is enforced, but that result still depends on more than inhibition alone.

## Code changes made

### Data path

- Switched the loader to the local `data` directory.
- Restored `use_fashion_data()` as an explicit marker of repository intent, although it does not change runtime behavior unless downloading is re-enabled.
- Removed the `mnist` crate `download` feature from `Cargo.toml`, because the repository already has local dataset files and the extra feature was only reintroducing OpenSSL/zlib/pkg-config build dependencies.

### STDP implementation

- Replaced the old `update(pre_last_fire, post_last_fire)` interface with explicit pre- and post-event handlers.
- Corrected the exponential timing dependence to use the actual elapsed time since the opposite spike.
- Added unit tests for potentiation and depression timing in `src/snn/synapse.rs`.

### Network state handling

- Reset spike trackers together with membrane state between samples.
- Evaluated current spikes first, then applied plasticity with the correct event semantics.

### Evaluation behavior

- Froze adaptive threshold changes outside training.
- Reused the low-activity replay loop for validation and test as well, based on total spike count.
- Removed the hard crash when some classes receive no assigned neurons and replaced it with a warning.

### Feedforward normalization

- Added a CLI switch `--normalization {none,l1,l2}`.
- Added a `FeedforwardNormalization` mode in `src/snn/network.rs`.
- Replaced the earlier forward-pass gain-scaling approximation with exact post-update renormalization of the stored plastic feedforward weights.
- Renormalized the feedforward weights of each touched postsynaptic neuron to unit L1 or unit L2 norm after STDP updates.
- Left inhibitory recurrent synapses unnormalized, because they are fixed competition weights rather than plastic feedforward evidence.
- Added unit tests that verify the renormalization path initializes unit L1 and unit L2 feedforward norms.

This change turned out to matter much more than expected in the dense regime. The earlier gain-scaling surrogate was not a faithful proxy for true renormalization when drawing conclusions about dense connectivity.

## Experimental evidence

### Baseline before the fix

Command:

```text
cargo run --release -- --train-length 2000 --mark-length 500 --test-length 1000 --per-sample-ticks 100 --output-num 200
```

Observed behavior:

- many neurons had zero validation firing
- assigned classes collapsed to `[1, 2, 4, 0, 111, 0, 0, 0, 45, 37]`
- the program panicked because several classes had no assigned neurons

These baseline results were on the repository's local Fashion-MNIST data.

### After the fix, same reduced configuration

Command:

```text
cargo run --release -- --train-length 2000 --mark-length 500 --test-length 1000 --per-sample-ticks 100 --output-num 200
```

Observed behavior:

- assigned classes spread across all ten classes: `[20, 15, 18, 3, 9, 21, 10, 25, 41, 38]`
- test accuracy: `35.00%`

This is still low, but it is a clear qualitative improvement over the pre-fix collapse.
This run was also on the repository's local Fashion-MNIST data.

### After the fix, larger reduced configuration

Command:

```text
cargo run --release -- --train-length 5000 --mark-length 1000 --test-length 2000 --per-sample-ticks 100 --output-num 400
```

Observed behavior:

- assigned classes: `[33, 40, 24, 17, 25, 22, 18, 52, 77, 92]`
- test accuracy: `39.25%`

This run was also on the repository's local Fashion-MNIST data.

### Dense input connectivity test without normalization

I also tested whether simply moving toward the literature's dense input connectivity would help:

```text
cargo run --release -- --train-length 2000 --mark-length 500 --test-length 1000 --per-sample-ticks 100 --output-num 200 --connection-rate 1.0
```

Observed behavior:

- all 200 neurons collapsed onto a single assigned class
- accuracy dropped to `10.30%`
- firing rates became extremely high (`~40.8` spikes/sample/neuron)

Interpretation:

- dense connectivity alone does not recover the behavior of a useful feed-forward classifier in this SNN
- this means sparsity is not the main standalone problem
- the existing horizontal inhibition did not prevent collapse, so inhibition alone is not sufficient here
- the current implementation needed some form of weight stabilization before the dense regime could even be evaluated fairly

### Normalization matrix: 5 runs per condition

I reran the reduced experiment (`train=2000`, `mark=500`, `test=1000`, `outputs=200`) five times for each combination of sparse/dense connectivity and L1/L2 normalization using the final exact post-update renormalization implementation.

| Connectivity | Normalization | Mean accuracy | Accuracy range | Mean zero classes | Mean avg firing |
| --- | --- | ---: | ---: | ---: | ---: |
| sparse (`0.05`) | `l1` | `31.78%` | `29.20%` - `35.40%` | `0.00` | `0.20` |
| sparse (`0.05`) | `l2` | `37.46%` | `34.00%` - `39.60%` | `0.00` | `0.32` |
| dense (`1.0`) | `l1` | `31.42%` | `24.70%` - `34.50%` | `0.00` | `0.33` |
| dense (`1.0`) | `l2` | `14.72%` | `10.30%` - `16.80%` | `0.00` | `0.47` |

Interpretation:

- Sparse L1 remains stable and lands almost exactly where the earlier gain-scaling approximation landed.
- Sparse L2 remains the best raw reduced-run baseline, although exact renormalization is slightly worse than the earlier approximation here.
- Dense L1 is the major new result: exact post-update renormalization lifts dense connectivity from near-chance behavior into a stable low-30% regime.
- Dense L2 remains poor and overactive, so norm choice is strongly coupled to the current inhibition and threshold dynamics.

The most important takeaway is that exact renormalization is not just a cosmetic implementation change. It changes the architectural conclusion: true L1 budget competition makes dense connectivity viable enough to study further, while dense L2 still exposes missing homeostasis and gain control.

## Conclusion

The original implementation was failing mainly because of correctness bugs, not because STDP is inherently unsuitable for this class of task.

The most damaging defects were:

- broken STDP timing window
- stale event timing in synapse updates
- cross-sample state leakage
- moving thresholds during evaluation

Those issues are now fixed. Separately, the repository still has a task/data mismatch: the instructions say MNIST, while the bundled data files are Fashion-MNIST.

If the goal is specifically MNIST, that requires replacing the contents of `data/` with MNIST or re-enabling a download path that fetches MNIST. Merely toggling `use_fashion_data()` is not enough once local data files already exist.

The user-proposed normalization direction turned out to be more consequential than I expected. The exact post-update implementation shows that real renormalization is materially different from the earlier forward-pass surrogate. In particular, dense L1 no longer behaves like a broken regime once the model enforces an actual synaptic budget on stored feedforward weights.

My updated conclusion is:

- sparsity is not the main standalone problem
- horizontal inhibition is necessary but not sufficient
- exact post-update renormalization matters more than the earlier approximation suggested
- sparse `l2` is still the best raw reduced-run baseline
- dense `l1` is now viable enough to change the dense-connectivity diagnosis
- the remaining gaps are slow homeostasis, tunable post-renormalization gain, and reproducible evaluation

## Recommended next steps

If the goal is to push this closer to literature-level MNIST performance, the next changes I would make are:

1. Keep `--normalization l2` as the best current sparse baseline.
2. Explore dense `--normalization l1` further, because exact renormalization shows it is no longer a collapsed regime.
3. Add an explicit target norm or post-renormalization gain parameter instead of hard-coding unit norm.
4. Add slow synaptic scaling and retune inhibition/threshold adaptation, especially if dense `l2` remains a target.
5. Wire up `--rng-seed` so these normalization comparisons are reproducible.
6. Align the dataset with the task goal: either replace `data/` with MNIST or explicitly document that the current repository is running Fashion-MNIST.
7. Run longer training schedules before judging final accuracy; the literature trains for much longer than these reduced experiments.

For more detail on why weight normalization and synaptic scaling matter here, and how the new L1/L2 results should be interpreted, see `copilot/improvement.md`.
