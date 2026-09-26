# Morrison settling and Fashion-MNIST homeostasis controls

This report preserves the measurements made with the old GeNN STDP tolerance.
The subsequently requested tolerance and gap guard, validation, and paired
reruns are recorded in
[MORRISON-TICK-TOLERANCE-20260920.md](MORRISON-TICK-TOLERANCE-20260920.md).

The previous 10-second Morrison measurements did **not** establish settling.
Longer runs now show continuing weight-distribution drift at 500 seconds in
the zero-delay case and instability at 46 seconds in the delayed case. A
separate deterministic experiment also exposes an FP32 event-timing defect
in their shared STDP implementation. Neither run establishes agreement with
the paper's equilibrium distribution.

For two-class Fashion-MNIST, stronger threshold adaptation, slower learning,
and shorter training presentations all avoid the baseline's collapsed
readout in two seeds. Stronger adaptation gives the highest accuracy in these
pilots, with a substantial retry cost.

The complete numerical record is
[settling-homeostasis-results-20260920.json](settling-homeostasis-results-20260920.json).
Commands, manifests, checkpoints, activity, raw weight samples, and generated
kernel code are under `copilot/tmp/workload_settling_homeostasis_20260920/`.
These results extend [the earlier investigation](INVESTIGATION-20260920.md).
No simulator source was changed in this investigation. The implementation is
the existing uncommitted workload code on commit
`9ad791ebc67086cb28047b82a884bceb55a565d3`; manifests record source hashes.

## Morrison: measuring settling

Both fresh runs use the full 90,000 E / 22,500 I network, 810 million plastic
synapses and 1.265625 billion recurrent synapses, on H800 80 GB GPUs. Seed and
initial-state seed are both 20260724. Configured delays are 0 and 1.5 ms;
GeNN's next-tick delivery makes the effective transport 0.1 and 1.6 ms.
The delayed case uses the existing dendritic timing convention, including a
3-ms configured postsynaptic learning path. No parameters were retuned.

The same 100,000 valid E–E edges are sampled every 10 simulated seconds.
Raw samples, histograms, moments, quantiles, and one-second population spike
counts are retained. This is a sample of the weight population, not a complete
weight checkpoint.

The stopping test was declared before launch: beginning at 200 seconds, check
every 50 seconds over the preceding 100 seconds. Require an absolute fitted
SD drift below 1% of mean SD, SD range below 2%, absolute fitted mean-weight
drift below 0.10 pA, and endpoint Kolmogorov–Smirnov distance below 0.02.
Two consecutive checks must pass. The run budget is 500 seconds, with a
100 Hz population-rate guard. The KS distance is descriptive; no independent-
sample significance claim is made for the repeatedly sampled synapses.

| Configured delay | Outcome | Last weight sample | Mean weight | SD | E rate at run endpoint |
|---|---|---:|---:|---:|---:|
| 0 ms | 500-second budget reached; all seven settling checks failed | 500 s | 44.7598 pA | 3.5492 pA | 2.594 Hz |
| 1.5 ms | Rate guard fired at 46 s | 40 s | 46.0312 pA | 3.4848 pA | 102.708 Hz |

For zero delay, SD increases from **3.414605 to 3.549163 pA between 400 and
500 seconds: +3.9407%**. The fitted relative drift is 3.8601% per window;
the fitted mean drift is −0.06235 pA and endpoint KS distance is 0.01667.
The mean and KS requirements pass, but the SD requirements do not. Mean E
rate over the final 100 seconds is 2.650 Hz. This is finite, ongoing activity,
not evidence that the weight distribution has stopped evolving.

The delayed case's E rate rises from 20.56 Hz at 40 seconds to 92.42 Hz at
45 seconds and 102.71 Hz at 46 seconds. Its last retained weight sample
precedes that rapid transition; there is no saved 46-second histogram, so
the plotted 40-second distribution must not be described as the final
unstable-state distribution.

Morrison et al. report settling around 200 seconds, mean 45.65 pA and SD
3.99 pA; their final histogram averages ten samples from the last 500 seconds
of a 2,000-second run. Their activity recording occurs at 400–450 seconds.
They also show that modest changes in depression can produce instability.
These observations make a short-lived, approximately Gaussian histogram an
insufficient replication criterion. [Paper, Sections 4.1–4.1.2 and Figure 4](https://brainworks.biologie.uni-freiburg.de/2007/journal%20papers/morrison-neco-2007.pdf).

![Weight width, center, firing rate, and retained histograms](../copilot/tmp/workload_settling_homeostasis_20260920/morrison_settling.png)

The zero-delay 500-second and delayed 40-second samples are approximately
bell-shaped, with skewness 0.177 and 0.186 and excess kurtosis 0.045 and 0.042.
Their similar instantaneous widths do not establish similar equilibrium
distributions: they have different centers, are observed at different times,
and neither passed a settling test. The reference density in the figure is
a Gaussian constructed from the paper's published mean and SD, not digitized
paper data. [PDF figure](../copilot/tmp/workload_settling_homeostasis_20260920/morrison_settling.pdf).

### Confirmed timing defect

The live GeNN STDP model was extracted into a two-neuron, one-synapse
experiment with prescribed spike times. When a single pre/post pair arrives
simultaneously and both traces start at zero, the selected causal-boundary
convention requires zero weight change. In FP32, some such events instead
change **45.610001 → 46.070919 pA**, an erroneous +0.460918 pA.

The global clock and reconstructed arrival timestamp can round differently.
The `1e-7 ms` tie tolerance is smaller than their rounding discrepancy, so the
pre handler misses a coincident post event and the post handler potentiates
using the newly incremented pre trace. This occurs on both CPU and CUDA:
five of twelve selected start offsets fail for each delay in FP32; all
corresponding FP64 cases pass. These tests cover early and late absolute
times and do not estimate the frequency of failures in a network simulation.
The existing earliest-tick smoke test did not expose this issue.

This is a confirmed implementation defect, but its contribution to the
full-network instability remains unmeasured. It affects the shared
Brunel/Morrison kernel; Fashion's three-trace implementation is separate.
The original clock/timestamp precision proposal is preserved in
[MORRISON-TIMING-PROPOSAL-20260920.md](MORRISON-TIMING-PROPOSAL-20260920.md).
The user subsequently chose a larger FP32 tie tolerance and a hard gap guard;
the linked timing report records that implementation and its controls.

### Work performed by the long runs

| Quantity | Zero delay, 500 s | Delayed, 46 s including guard-triggering interval |
|---|---:|---:|
| E spikes | 145,341,294 | 73,664,010 |
| I spikes | 45,242,155 | 16,749,471 |
| Total recurrent-neuron spikes | 190,583,449 | 90,413,481 |
| E–E postsynaptic visits enqueued | 1,308,071,646,000 | 662,976,090,000 |
| E–E presynaptic visits enqueued | 1,308,064,394,927 | Exact count unavailable after guard |
| Plastic pre + post visits enqueued | 2,616,136,040,927 | Bound: 1,296,044,591,940–1,356,228,088,110 |
| All recurrent presynaptic visits enqueued | 2,144,065,235,473 | Exact count unavailable after guard |
| Simulation wall time | 1,300.78 s | 603.10 s |

These are event visits, including visits with zero weight change, rather than
distinct synapses changed. Enqueued counts include events still pending at
the endpoint. Unlike the earlier 10-second investigation, these runs did not
record endpoint spike times to subtract pending deliveries. The delayed
guard fires before the final per-neuron count vector is saved; population
totals are recoverable from one-second records, while the stated outgoing-
visit bound uses the saved actual minimum and maximum E–E outdegrees.
External Poisson events are not included. The zero-delay run contains
562.5 billion neuron time steps; the delayed run contains 51.75 billion.

Jobs 551329 and 551330 completed and failed at the intended rate guard,
respectively. Their process times, including build and diagnostics, were
1,332.16 and 629.66 seconds. Seventy downloaded artifacts were checked against
remote SHA-256 values in `remote_artifact_sha256.json`.

## Fashion-MNIST: threshold adaptation, learning rate, and presentation time

All conditions use the same first 1,000 accepted training images, labels 0/1
(T-shirt/top and trouser), and 784 → 400 E / 400 I topology. Training order is
fixed across conditions and seeds; seeds 20260724 and 20260725 vary initial
weights and Poisson generation. STDP does not use labels. Runs use FP32 GeNN
on RTX 3090. Normalization to column sum 78 before every attempt, retry below
five population spikes, and 150-ms zero-input rest remain enabled.

The larger evaluation uses **400 new held-out training images for assignment
and 400 new test images**, balanced across the two labels. Assignment excludes
all 1,000 training images; both sets exclude the earlier 40-image evaluation
sets. Every case starts fresh inference state with seed 20260724, inhibition
17, frozen weights/theta, and **350-ms inference presentations**, including
networks trained with 100-ms presentations. Checkpoints are used without
additional normalization. All ten frozen-state checks passed.

| Training condition | Accuracy, seed 20260724 | Accuracy, seed 20260725 | Training retries per 1,000 images, two seeds | Stimulus exposure, two seeds |
|---|---:|---:|---:|---:|
| Baseline: θ increment 0.05 mV, 350 ms | 49.75% | 50.00% | 7 / 14 | 352.45 / 354.90 s |
| Both learning rates ×0.1 | 84.25% | 85.50% | 41 / 48 | 364.35 / 366.80 s |
| θ increment ×4: 0.20 mV | 90.25% | 88.75% | 419 / 429 | 496.65 / 500.15 s |
| θ increment ×10: 0.50 mV | 92.00% | 91.00% | 1,055 / 1,020 | 719.25 / 707.00 s |
| Training presentation 350 → 100 ms | 81.25% | 80.75% | 484 / 454 | 148.40 / 145.40 s |

These comparisons hold accepted-image budget fixed. Increasing adaptation
raises the number of retries and hence stimulus exposure; it is not a
constant-simulation-time comparison. The tenfold condition averages about
two attempts per training image and about 3.1–3.2 attempts per inference
image. The fourfold condition averages about 1.42 training attempts and 1.81
inference attempts. Slower learning requires only about 1.04–1.05 training
attempts per image.

The baseline's most active neuron contributes 76.69% and 95.24% of all test
spikes in the two seeds. Those fractions fall to 0.66% / 0.64% with slower
learning, 1.39% / 1.43% with fourfold adaptation, and 2.26% / 3.11% with
tenfold adaptation. Thus the improvements accompany reduced single-neuron
dominance, not merely a different tie in the classifier.

The homeostasis hypothesis is supported by these interventions: increasing
the spike-triggered threshold increment strengthens the response to a
winning neuron's firing and avoids the observed collapse. The decay time
constant remains 10,000 seconds. Making that decay faster would instead
erase adaptation sooner. Normalization and the minimum-spike retry rule are
useful but do not guarantee class selectivity or balanced competition; a
single neuron can satisfy the population's five-spike criterion.

Reducing both potentiation and depression by ten preserves their coefficient
ratio and also avoids collapse. Shorter presentations likewise help, but
reduce stimulus exposure to about 41–42% of baseline despite extra retries.
Including rest, their total simulated time falls from 503.5–507.0 seconds to
363.5–371.0 seconds. These experiments therefore do not separate faster
switching/normalization from reduced total exposure. That requires a further
matched-exposure control, not reinterpretation of the present results.

For completeness, the original smaller protocol remains separately recorded:

| Condition | 40-test raw accuracy, two seeds | 40-test normalized-copy accuracy, two seeds |
|---|---:|---:|
| Baseline repeat | 50 / 50% | 67.5 / 47.5% |
| Learning rates ×0.1 | 77.5 / 87.5% | 80 / 77.5% |
| θ increment ×4 | 80 / 97.5% | 82.5 / 82.5% |
| θ increment ×10 | 80 / 77.5% | 80 / 77.5% |
| 100-ms training stimulus | 72.5 / 85% | 72.5 / 65% |

Raw and normalized inference can take different spike/retry paths even when
the final column sums are close to 78. The larger results above use one fixed
raw-checkpoint protocol, rather than selecting each case's best evaluator.
Do not compare the larger-set scores directly with earlier one-/two-trace
40-image scores. These remain 1,000-image training pilots with two seeds;
they do not establish convergence or a general ranking of learning rules.
