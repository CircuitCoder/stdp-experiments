# Fashion-MNIST collapse and Morrison delay/count investigation

The two-class Fashion-MNIST failure is real: the original three-trace
checkpoint predicts label 0 for all 40 test images. The evidence points to
aggressive learning, competition, and sensitivity to the checkpoint's position
relative to normalization. Increasing inhibition did not resolve it. Zero-input
rest is implemented correctly, and independent spike-event replay found no
error in the tested three-trace weight updates.

Morrison's zero-delay and delayed controls both sustain irregular firing over
10 seconds, but their rates, population fluctuations, and weight drift differ.
The zero-delay derivative should not be described as reproducing the original
network's dynamics or equilibrium.

The machine-readable record is
[investigation-results-20260920.json](investigation-results-20260920.json).
Full commands, manifests, activity, checkpoints, generated code, and logs are
under `copilot/tmp/workload_investigation_20260920/`. Simulator source and
original pilot artifacts were not modified. The source remains the uncommitted
workload implementation based on `9ad791ebc67086cb28047b82a884bceb55a565d3`;
each diagnostic manifest records source hashes. These are follow-up experiments
to [PILOTS.md](PILOTS.md), not replacements for its measurements.

## Fashion-MNIST and CIFAR-10

All two-class evaluations below use the original, fixed selection: 20 held-out
training images per class for assignment and 20 separate test images per class.
The labels are 0/1, selected by training frequency with ascending-label ties.
Networks have 400 E and 400 I neurons; Fashion has 784 inputs. Each inference
starts with fresh runtime state, uses seed 20260724 and inhibition 17 unless
specified otherwise, and verifies that weights and thresholds remain frozen.
Labels do not enter STDP. Normalization experiments change an in-memory copy
of the checkpoint; they do not change the saved training checkpoint.

### What collapsed

In the original Fashion two-class three-trace checkpoint, only neuron 177 fires
during assignment. It responds to both classes, averaging 23.55 spikes for
T-shirt/top and 16.65 for trouser, so assignment labels it 0. The remaining 399
neurons are silent. Every test image then receives label 0. This is not simply
an all-zero score tie in the classifier.

The training run used all 400 neurons over its 1,000 accepted images. The most
active neuron contributed 0.90% of its accepted-image spikes, although activity
was becoming less evenly distributed late in training. Thus the inference
collapse is more severe than the aggregate training activity suggests.

Column normalization occurs **before every training attempt**, while the
checkpoint is saved after the final stimulus and rest. Inference loads those
weights without normalization. Neuron 177's final incoming weight sum is
105.2329, versus approximately 78 for the other 399 neurons. It also emitted
all 22 counted spikes on the last training image. The one- and two-trace
Fashion checkpoints have maximum column sums of only 78.1798 and 78.0790.

CIFAR's two-class three-trace checkpoint has the same pattern: only neuron 19
fires during assignment; its column sum is 200.5276, versus approximately 78
elsewhere. It emitted all 13 counted spikes on the last training image.

### Controlled inference and training probes

| Diagnostic | Fashion two-class accuracy | Assigned / silent E neurons |
|---|---:|---:|
| Original three-trace checkpoint, unchanged | 50.0% | 1 / 399 |
| Original checkpoint, inference inhibition 17 → 25.5 | 50.0% | 1 / 399 |
| Original checkpoint, columns normalized to 78 | 50.0% | 323 / 77 |
| Fresh baseline repeat, 1,000 training images | 50.0% | 158 / 242 |
| Same fresh baseline checkpoint, normalized inference copy | 67.5% | 399 / 1 |
| Fresh training with both STDP coefficients divided by 10 | 77.5% | 400 / 0 |
| Same slower-learning checkpoint, normalized inference copy | 80.0% | 400 / 0 |
| Fresh training with inhibition doubled, 25.5 → 51 | 50.0% | 1 / 399 |
| Same stronger-inhibition checkpoint, normalized inference copy | 50.0% | 371 / 29 |

Fresh controls use identical initial weight arrays, seed, image order, selection,
and 1,000 accepted-image budget. Slower learning means potentiation
`0.01 → 0.001` and depression `0.0001 → 0.00001`, preserving their coefficient
ratio. The original one-/two-trace rules use potentiation coefficient 0.0005,
but their equations and trace conventions differ, so coefficient ratios alone
are not ratios of effective learning speed.

The original Fashion checkpoint remains dominated by neuron 177 even after
normalization: it averages 16.90 and 10.65 assignment spikes per image across
the two classes. The next most active neuron averages only 0.05 and 0.25.
Normalization recruits other neurons but does not recover useful competition
in that checkpoint. Increasing lateral inhibition also cannot directly inhibit
the winner through its own inhibitory partner: the I→E diagonal is excluded.

The fresh baseline repeat illustrates checkpoint sensitivity. At sample 999
it scored 65.0%; after sample 1000 it scored 50.0%. On that last attempt, the
largest column grew from 78.0000 before presentation to 95.2445 after stimulus
and 96.5419 after rest. With slower learning the final maximum was 78.0353;
with doubled training inhibition it was 98.8075. Slower learning also produced
a narrower threshold range, 29.17–33.76 mV, versus 22.63–47.04 mV for the fresh
baseline repeat. These observations support a learning-dynamics problem.

The fresh baseline is a **repeat, not a bitwise replay** of the initial pilot.
Its initial weight hash matches, and its per-neuron accepted-image activity
matches through sample 59, but it diverges at sample 60. The precise numerical
cause is unresolved. The current source includes the previously added
plasticity-off guards; the original Fashion pilot used source_v2. Do not infer
exact reproducibility from the seed or attribute a small accuracy difference
to one parameter from these single runs.

For CIFAR's original two-class checkpoint, normalizing the inference copy
raises accuracy from **50.0% to 77.5%**, with 398 assigned neurons and two
silent. For full Fashion, however, normalization changes the original
three-trace result from 50.0% to 45.5% on its 200-image test set. Normalization
is therefore a diagnostic and a protocol choice, not a demonstrated universal
accuracy fix.

These are early 1,000-image pilots with small evaluation sets. Earlier MNIST
experiments used different training budgets and, in several cases, the
optimistic same-activity assignment/scoring protocol. Also, the inherited
pixel-to-rate conversion is not rate-matched across datasets: mean training
pixel intensity is 72.94 for Fashion versus 33.32 for real MNIST. Identical
topology and parameters consequently do not imply identical operating regimes
or a guaranteed ordering of the learning rules.

### Rest and implementation checks

`GeNNNetwork.run_rest()` sets every input rate to zero and advances 300 ticks
(150 ms). There is no added background noise or image during rest. The input
Poisson generator still draws random values, but a zero rate produces no spikes.
The network is not reset: pending deliveries, E/I activity, traces, thresholds,
and training plasticity continue to evolve.

Measured rest activity was:

| Run | Input spikes during rest | E spikes during rest | I spikes during rest |
|---|---:|---:|---:|
| Original Fashion checkpoint, 80 inference attempts | 0 | 2 | 61 |
| Fresh Fashion baseline, 1,007 training attempts | 0 | 91 | 1,186 |
| Slower-learning Fashion, 1,041 training attempts | 0 | 254 | 1,911 |
| Stronger-inhibition Fashion, 1,016 training attempts | 0 | 222 | 1,482 |

The fresh baseline emitted 63,518 E spikes during stimuli; its 91 rest E spikes
are about 0.14% of the combined count. Rest can still change weights through
residual spikes and the final stimulus tick's pending events, as the final-image
column measurements above demonstrate.

The counter baseline advances at stimulus end, so a presentation's returned
count includes the preceding rest's spikes. This is explicitly inherited from
Brian 1 and documented in `reimpl/README.md`, rather than a new workload-loader
mistake. In the collapsed original Fashion inference it adds only two spikes;
it does not explain a one-neuron, one-class readout.

An independent recurrence replayed the actual recorded input/E spikes from
generated three-trace models, including first-post events and simultaneous
pre/post events. A small 784-input, 4 E/4 I diagnostic uses the live kernels and
the normal stimulus/rest schedule. Maximum final-weight error was
`1.075e-7` on GeNN CPU and `1.153e-7` on CUDA, against a `3e-6` tolerance.
This checks the tested STDP event mechanics; it does not prove all aspects of
the simulator correct. Frozen inference checks also passed in every probe.

A sensible next experiment is slower three-trace learning with several seeds,
longer training, and a larger fixed held-out evaluation set, while explicitly
recording raw versus normalized checkpoint evaluation. The current evidence
does not support increasing inhibition as the first remedy or promoting any
of these pilot accuracies to convergence results.

## Morrison: delay comparison and workload counts

Jobs 551205 (H800) and 551206 (A800) each ran a full-size 10-second zero-delay
case and a fresh 10-second delayed control. Both used 90,000 E/22,500 I,
810,000,000 plastic E→E synapses, 1,265,625,000 total recurrent synapses,
FP32, 0.1 ms steps, seed/state seed 20260724, and plasticity throughout.
All 112,500 neurons fired during each run, and finite-state/rate guards passed.

The control sets the configured delay to 1.5 ms and uses the existing
`nest_dendritic` STDP convention, including its 3 ms postsynaptic learning-path
delay. The original paper uses entirely dendritic delay. Simply retaining
arrival-timed STDP with a 1.5 ms transport delay would be a different control.
GeNN's update order makes physical delivery one global tick later than the
configured delay: 0.1 ms for the zero-delay case and 1.6 ms for this control.
This is a comparison within the current port, not an exact historical NEST
reproduction. Initial voltage arrays and all four outdegree arrays match
between the delay cases on each device; full connectivity was not copied for
an independent edge-by-edge identity check.

### Observed behavior

| Device | Configured delay | Mean E / I rate, whole run | E ISI CV, 1–10 s | 3 ms count Fano, 1–10 s | Final sampled weight mean ± SD |
|---|---:|---:|---:|---:|---:|
| A800 | 0 ms | 6.058 / 6.360 Hz | 0.879 | 2.584 | 45.505 ± 1.168 pA |
| H800 | 0 ms | 6.098 / 6.392 Hz | 0.889 | 2.518 | 45.498 ± 1.177 pA |
| A800 | 1.5 ms | 7.593 / 7.646 Hz | 0.874 | 7.811 | 45.710 ± 1.438 pA |
| H800 | 1.5 ms | 7.719 / 7.749 Hz | 0.869 | 8.184 | 45.721 ± 1.462 pA |

ISI CV is averaged over the first 1,000 E neurons; the Fano factor bins their
combined spike train in 3 ms bins. Excluding more of the transient leaves the
same distinction: over 5–10 s, zero-delay Fano is 2.34–2.41 versus 8.09–8.25
for the delayed control. Late E rates are 5.79–5.80 versus 7.98–8.22 Hz.
The zero-delay mean weight moves downward from the initial 45.61 pA, while
the delayed mean moves upward. Weight statistics sample 100,000 actual edges,
excluding sparse-storage padding.

Thus irregular single-neuron firing survives zero delay in this pilot, but
population fluctuations are much smaller and the plastic trajectories differ.
The paper reports equilibrium near 8.8 Hz, ISI CV 0.88, Fano 8.5, and weights
45.65 ± 3.99 pA, with the weight distribution settling after approximately
200 simulated seconds. Our 10-second comparisons cannot establish that
equilibrium. [Morrison, Aertsen & Diesmann (2007), sections 3–4 and appendix](https://brainworks.biologie.uni-freiburg.de/2007/journal%20papers/morrison-neco-2007.pdf).

### Counts over 10 simulated seconds / 100,000 steps

| Device | Configured delay | E spikes | I spikes | Total E+I spikes | Plastic update visits | All recurrent presynaptic deliveries |
|---|---:|---:|---:|---:|---:|---:|
| A800 | 0 ms | 5,452,401 | 1,431,037 | 6,883,438 | 98,142,780,937 | 77,438,400,656 |
| H800 | 0 ms | 5,488,012 | 1,438,188 | 6,926,200 | 98,783,197,079 | 77,918,747,107 |
| A800 | 1.5 ms | 6,833,835 | 1,720,460 | 8,554,295 | 122,968,575,145 | 96,213,142,492 |
| H800 | 1.5 ms | 6,946,922 | 1,743,496 | 8,690,418 | 124,974,419,615 | 97,725,573,083 |

Here **plastic update visits** means E→E presynaptic depression-handler visits
plus E→E postsynaptic potentiation-handler visits, including events whose
weight delta is zero. It is not the number of distinct synapses, nonzero
floating-point weight changes, or all historical pre/post spike pairs.
Presynaptic E→E visits also deliver current and are included in the last
column; do not add those two columns as if their work were disjoint.

Presynaptic visits are calculated exactly from each neuron's measured spike
count times its actual outgoing row length for EE, EI, IE, and II. E→E
postsynaptic visits use the exact fixed indegree of 9,000. Recorded spike times
subtract events still queued beyond the simulation endpoint, accounting for
the global tick and the delayed postsynaptic path. Counts were cross-checked
against per-neuron counters, chunk totals, and saved degree arrays. The raw
records retain both enqueued and processed totals.

For zero delay, the plastic breakdown is:

| Device | E→E pre visits | E→E post visits | Plastic visits per global step | Static recurrent deliveries |
|---|---:|---:|---:|---:|
| A800 | 49,071,333,937 | 49,071,447,000 | 981,427.81 | 28,367,066,719 |
| H800 | 49,391,674,079 | 49,391,523,000 | 987,831.97 | 28,527,073,028 |

External Poisson multiplicities are generated inside neuron updates and were
not recorded as spikes. They are excluded from the E/I and recurrent-delivery
counts above. Their configured expectation is 23.49 billion events over ten
seconds across all neurons, represented as aggregate counts rather than an
explicit external synapse projection. All runs execute 11.25 billion neuron
timesteps.

The synchronized simulation timers were 73.98/46.47 seconds for A800/H800 at
zero delay and 91.95/57.99 seconds with configured 1.5 ms delay. They exclude
recording retrieval, diagnostics, construction, and final weight sampling.
Zero delay entails about 20% fewer plastic visits in these runs; its shorter
runtime should not be attributed solely to cheaper delay handling.

The initial unrecorded pilots remain separate measurements: their E+I totals
were 6,919,355 on A800 and 6,882,830 on H800. The instrumented reruns above
provide the detailed visit accounting. Their transferred non-build artifacts
were verified against 26 remote SHA-256 digests, and both Slurm jobs completed
successfully.
