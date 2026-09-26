# Longer three-trace image runs

The strongest tested intervention was increasing the adaptive-threshold
increment from 0.05 to **0.50 mV per excitatory spike**. It achieved 92.00% and
91.00% in the two completed seeds. These were **two-class Fashion-MNIST**
experiments; the four-way intervention comparison was not repeated on CIFAR-10
or either ten-class dataset.

## Completed intervention comparison

Each case trained on 1,000 accepted images with 400 E / 400 I neurons. The
reported evaluation assigns neurons from 400 held-out training images and
scores 400 separate test images. Both sets exclude the earlier small probes;
assignment also excludes all images used for STDP. Inference uses the raw
checkpoint, frozen weights/theta, inhibition 17, 350 ms presentation and
150 ms blank rest. Training labels are not supplied to STDP.

| Intervention | Seed 20260724 accuracy | Seed 20260725 accuracy | Retries, respectively |
|---|---:|---:|---:|
| Unchanged three-trace control | 49.75% | 50.00% | 7 / 14 |
| Both STDP learning rates ×0.1 | 84.25% | 85.50% | 41 / 48 |
| Threshold increment ×4, to 0.20 mV | 90.25% | 88.75% | 419 / 429 |
| **Threshold increment ×10, to 0.50 mV** | **92.00%** | **91.00%** | **1,055 / 1,020** |
| Training presentation shortened to 100 ms | 81.25% | 80.75% | 484 / 454 |

The two adaptation interventions change homeostasis, rather than directly
scaling the STDP coefficients. The selected intervention approximately doubles
the presentation attempts at this training budget. Its stimulus exposure was
719.25 / 707.00 simulated seconds, compared with 352.45 / 354.90 for the
control. Thus this is a comparison at equal accepted-image counts, not equal
exposure or compute. Threshold decay remains 10,000 seconds; normalization,
minimum-five-spike retry and blank rest remain enabled.

The source records and checkpoints are described in
[SETTLING-HOMEOSTASIS-20260920.md](SETTLING-HOMEOSTASIS-20260920.md) and
[settling-homeostasis-results-20260920.json](settling-homeostasis-results-20260920.json).
No new four-method replication is claimed by this table.

## Full-length protocol

The four new cases use three passes through their respective training pools.
For every class, 200 training images are reserved for assignment and excluded
from STDP throughout all three passes. Each pass uses the same seeded shuffled
order. Selection and order indices, data hashes, initialization hashes and
source hashes are recorded. Class-frequency ties use ascending label IDs.

| Workload | Input neurons | Training pool per pass | Accepted-image target | Final test images |
|---|---:|---:|---:|---:|
| Fashion-MNIST, ten classes | 784 | 58,000 | 174,000 | 10,000 |
| Fashion-MNIST, labels 0 and 1 | 784 | 11,600 | 34,800 | 2,000 |
| CIFAR-10, ten classes | 3,072 | 48,000 | 144,000 | 10,000 |
| CIFAR-10, labels 0 and 1 | 3,072 | 9,600 | 28,800 | 2,000 |

All use three-trace dense STDP, 400 E / 400 I neurons, FP32 weights and neuron
state, seed 20260724, the selected 0.50-mV increment, 350/150-ms presentation/
blank-rest phases, normalization to 78, and the existing increasing-intensity
retry. Native RGB inputs retain all three CIFAR channels without preprocessing.
These are accuracy experiments running concurrently on an RTX 3090; their wall
times are not platform throughput benchmarks.

Checkpoints are saved after 1,000 images, every 5,000 images, at each epoch
boundary and at the final target. Periodic evaluation uses 200 test images per
class; final evaluation uses the complete retained official test split. Every
evaluation starts a new process and fresh inference runtime while the training
runtime remains loaded and paused. It verifies frozen weights/theta, unchanged
checkpoint content, and preserved training timestep and weights/theta. Training
therefore continues without resetting voltages, conductances, traces, or RNG.
The immutable weight/theta checkpoints are intended for inference, not exact
training resumption after process failure.

Per-100-image diagnostics record retries, theta, weights and column sums,
membrane/conductance ranges, event counts and the dominant neuron's recent
spike share. Inference records assignment balance, confusion matrices and
spiking activity. The runs abort on nonfinite state, 5,000 E spikes in an
attempt, exhaustion of the declared retry ceiling, or implausible
threshold/voltage ranges. All four active replacements now use a ceiling of 60;
the original ceiling-20 attempts are retained as failed controls.

### Retry-ceiling controls

The initial full and two-class CIFAR runs stopped after 462 and 409 accepted
images: the next image still produced fewer than five spikes at intensity 20.
Their last diagnostics had active, nearly evenly distributed firing, rather
than the original single-neuron collapse. The first unaccepted full-CIFAR
image is training index 25,184, with mean pixel 49.48 and maximum pixel 184.
The retry ceiling was a pilot stopping rule, not a bound in the reference's
increasing-intensity mechanism.

Fresh CIFAR cases, `cifar_full_retry60` and `cifar_top2_retry60`, retain the same
dataset indices, initialization, ×10 adaptation and learning parameters, with
the ceiling raised to 60 in training and inference. Initial intensity remains
2 and each unsuccessful attempt raises it by one. At the ceiling, a maximum-
brightness pixel has per-tick Bernoulli firing probability 0.95625. The runner
rejects ceilings that would make this probability reach one. This bound avoids
clipping impossible probabilities; it does not make a high-probability
Bernoulli input an exact continuous-time Poisson process.

The failed cases, source snapshot and logs are preserved. The replacement
cases use a separate immutable `source_retry60/` snapshot and
`launch_manifest_retry60.json`; a native RGB checkpoint/evaluation smoke test
passed before launch. They retain the same full-length targets and checkpoint
schedule. Their behavior beyond the first retry-limit failure must be measured;
the ceiling change is not an established accuracy improvement.
Manifest checks confirm identical model constants, learning rule, initial
weights/theta, connectivity mask, data hashes, training order and evaluation
selections. Equal seeds do not ensure bitwise identical full CUDA trajectories:
one replacement already differs in weights before using an intensity above 20.
These are matched configurations, not exact replay of the failed processes.

The ceiling-20 Fashion runs subsequently stopped too: two-class training
accepted 1,713 images before exhausting the ceiling on the next one, and the
full case failed during held-out assignment at the 1,000-image checkpoint.
The two-class checkpoint had scored **92.0% on 400 separate test images**, with
224 / 110 assigned neurons and 66 unassigned. Its training had 2,021 attempts
for 1,000 accepted images. This is consistent with the earlier short-pilot
result, but does not establish full-length feasibility at the original retry
ceiling.

Fresh `fashion_full_retry60` and `fashion_top2_retry60` cases therefore use the
same ceiling-60 runner and source snapshot. Their launch is recorded in
`launch_manifest_retry60_fashion.json`. They start from the same initialization;
no weight-only checkpoint is presented as an exact continuation. The larger
retry budget applies to inference as well as training.
The ceiling-60 two-class CIFAR run subsequently passed 500 accepted images and
had required intensity 32, providing direct evidence that the original ceiling
was restrictive. This is a training-progress result, not a completed accuracy
evaluation. `ceiling20_outcomes.json` preserves the four stopped attempts and
the completed 92% Fashion evaluation. Its attempt totals count attempts for
accepted images; the failed image's final attempts are not included.

## Long-run timestamp protection and validation

At the image timestep of 0.5 ms, a single-precision absolute clock loses tick
resolution beyond approximately 8,388.608 seconds of simulated time. Full
training can exceed that. The new opt-in `integer_timestamps=True` image backend
stores current, last-spike and previous-spike ticks as `uint32` on the input and
excitatory neurons. Three-trace learning compares integers and subtracts ticks
before converting elapsed intervals to FP32 milliseconds. A hard assertion
prevents counter wrap. This adds neuron-sized state, not per-synapse timestamp
arrays. The ordinary image backend default is preserved for existing benchmark
callers; `long_images.py` explicitly enables the integer option.

Deterministic CPU and CUDA checks exercise prescribed pre/post spike sequences,
including a same-tick pair with existing trace history. The new representation
matches the previous short-time FP32 result **bit for bit**, agrees with an
independent analytic calculation within 1e-7 weight units, and remains bitwise
invariant at offsets beyond 2^24, 2^28 and 2^31, and near unsigned wrap. The
relevant image regression suite passed 14 tests; the generated CUDA timing
test passed separately. CUDA smoke runs exercised both Fashion and native RGB
CIFAR, immutable checkpoint I/O, isolated evaluation, and frozen inference.

## Artifacts and execution

The reusable runner is [long_images.py](long_images.py). The immutable runtime
source snapshot, pinned environment, launch commands and all outputs are under
`copilot/tmp/workload_full_image_theta10_20260920/`.
`launch_manifest_v2.json` records the original ceiling-20 attempts;
`launch_manifest_retry60.json` and `launch_manifest_retry60_fashion.json`
record the active replacement launches.
The initial service attempts
failed in `nix-shell` before Python started; their logs are retained. Version 2
directly launches the pinned environment resolved outside the service.

The service names are:

```text
stdp-theta10-fashion-full-retry60-20260920
stdp-theta10-fashion-top2-retry60-20260920
stdp-theta10-cifar-full-retry60-20260920
stdp-theta10-cifar-top2-retry60-20260920
```

Each case has `manifest.json`, `selection.npz`, `progress.jsonl`, immutable
`checkpoints/`, separate `evaluations/`, and an `evaluations.jsonl` summary.
The valid prefix of the preallocated activity files is the accepted count in
the progress/result record. `result.json` is written on completion or a caught
failure. Inspect service state and the log as well, since a hard process abort
cannot write a final result.
