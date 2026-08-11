# Zero-delay derived network experiments

## Scope and comparison protocol

These experiments derive three variants from
`ref/zero_delay_midpoint_v1/inhib150_full` without modifying that active
baseline: a direct-voltage synapse model, a one-trace learning rule, and a
two-trace learning rule. Every training and evaluation run has its own runtime
directory, copied source, random/recurrent artifacts, `case.env`, hashes, and
log. All runs use real MNIST in fixed sequential order, 400 excitatory and 400
inhibitory neurons, 0 ms feedforward delay, 350 ms stimulus plus 150 ms rest,
1.5x training inhibition, and 1.0x inference inhibition unless a case manifest
says otherwise.

Accuracy is the optimistic simple-demo protocol: the same fixed test activity
assigns neurons and scores predictions. It is not the paper protocol with a
separate assignment pass. The comparable zero-delay triplet baseline scored
63.3% after a 1,000-sample pilot, 80.6% at 10,000 accepted samples, 86.6% at
20,000, 89.2% at 30,000, 91.3% at 40,000, and 92.4% at 50,000 on 1,000-image
probes. The 10k point was re-evaluated under the 1.0x inference-inhibition
setting used by the derived checkpoint watchers; the older 78.5% 10k record
used 1.5x and is not the comparison control.

Parent source SHA-256:
`ab30ebbbf01276ca15ea20f05de76ef90c8cd69933d43702d98b1eb4174f1b85`.

## Learning-rule variants

The one-trace rule uses a presynaptic trace `x` with 20 ms decay:

```text
pre:  x <- x + 1
post: w <- w + nu_post (x - x_target) (w_max - w)^mu_post
```

Its paper defaults are `nu_post=0.01`, `x_target=0.4`, `mu_post=0.2`, and
`w_max=1`. Brian 1 requires both pre- and post-equation groups, so the compiled
implementation includes a fixed, weight-neutral post marker. A compiled CSTDP
micro-test matches the independently calculated event sequence.

The two-trace rule adds a postsynaptic trace `y`, also with 20 ms decay:

```text
pre:  x <- x + 1; w <- w - nu_pre y w^mu_pre
post: w <- w + nu_post (x - x_target) (w_max - w)^mu_post; y <- y + 1
```

Paper defaults are `nu_pre=0.0001`, `nu_post=0.01`, `x_target=0.4`, and
`mu_pre=mu_post=0.2`. Its compiled CSTDP event sequence also matches an
independent calculation. Family manifests and sources are under
`ref/zero_delay_one_trace_v1/` and `ref/zero_delay_two_trace_v1/`.

At 1,000 training samples, the default one- and two-trace rules produced only
1.615 and 1.755 active excitatory neurons per image and zeroed 6.54% and 6.77%
of feedforward weights. Their 1,000-image probes scored only 36.8% with 144
assigned neurons and 37.2% with 116 assigned neurons. The triplet control had
about 5.9-9.7 active neurons and nearly no zero weights at the same stage, so
the paper-rate cases were rejected. Reducing only `nu_post` to `0.001` restored
baseline-like activity. The one-trace pilot scored 62.2% with 397 assignments,
16.13 inference spikes, and 8.99 active neurons; the two-trace pilot scored
65.9% with 396 assignments, 15.85 spikes, and 8.89 active neurons. Both tuned
rules were promoted to new 30,000-sample cases with immutable 10k, 20k, and 30k
checkpoint probes.

The `nu_post=0.001` runs formed zero weights quickly: about 3.4% at 1k, 7.8%
at 2k, and 12% at 3k, although firing remained balanced. Matched hedge pilots
with `nu_post=0.0005` reduced the 1k zero fraction to about 1.9%, kept maximum
weights near 0.30-0.33 instead of 1.0, and improved the one-/two-trace probes to
70.2% and 69.6% with 399 assigned neurons. New `full_nupost0005_30000`
branches and checkpoint watchers were therefore launched while preserving the
`0.001` runs as controls.

A further matched `nu_post=0.00025` pilot tested whether reducing weight
collapse again would improve learning. It cut the 1k zero-weight fractions to
0.59% (one trace) and 0.63% (two trace), with no saturated weights, but the
1,000-image scores fell to 63.3% and 63.5%. Both cases assigned 399 neurons and
had about 17.3 inference spikes and 10.8-11.0 active neurons per image. This is
evidence of under-learning at the 1k screen, so `0.00025` was rejected and was
not promoted to 30,000 samples.

At the first immutable checkpoint, `nu_post=0.001` scored 82.0% for one trace
and 82.1% for two trace at 10k accepted samples. Both assigned all 400 neurons;
the probes produced 25.66/24.45 spikes and 10.81/10.73 active neurons per image.
This slightly exceeds the corrected 80.6% triplet control at the same checkpoint
and protocol. Although the training checkpoints contained 41.5% and 40.1% zero
weights, firing remained balanced, so this is useful sparsification rather than
an accuracy collapse at 10k. The hash-verified `0.001` controls therefore remain
valid candidates alongside the `0.0005` branches.

The hash-verified `nu_post=0.0005` checkpoints subsequently scored 84.3% for
one trace and 82.8% for two trace at 10k accepted samples, again with all 400
neurons assigned. Their probes produced 19.90/18.83 spikes and 9.40/9.23 active
neurons per image. Under the same checkpoint, image set, and inference protocol,
the slower rate therefore improves one trace by 2.3 points over `0.001`, while
the two-trace difference is only 0.7 point. All four branches were continued so
their 20k and 30k trajectories could distinguish an early advantage from a
stable one.

At 20k, the hash-verified `nu_post=0.001` probes diverged. One trace reached
85.9% with 25.74 spikes and 9.00 active neurons per image, only 0.7 point below
the 86.6% triplet control. Two trace reached 83.5% with 25.98 spikes and 9.32
active neurons, 3.1 points below control. Both assigned all 400 neurons and had
similar firing statistics, so the gap reflects learned selectivity rather than
gross readout suppression. These intermediate results warranted continuing both
rules to the 30k checkpoint.

The hash-verified `nu_post=0.0005` checkpoints did better at 20k. One trace
reached 88.4% with 23.22 spikes and 9.44 active neurons per image, exceeding
the 86.6% triplet control by 1.8 points and its `0.001` branch by 2.5 points.
Two trace reached 84.7% with 22.13 spikes and 9.33 active neurons, improving
1.2 points over its `0.001` branch but remaining 1.9 points below control. Both
assigned all 400 neurons. Their training checkpoints had 40.8%/37.8% zero
weights, substantially below 68.5%/66.8% for `0.001`; the slower rate preserves
more receptive-field support and is the preferred parameter for both simple
rules going into the 30k checkpoint.

At 30k the hash-verified `nu_post=0.001` checkpoints converged to nearly the
same sub-target result: 86.1% for one trace and 86.0% for two trace. They
assigned 399/400 neurons and produced 20.51/20.28 spikes with 7.28/7.27 active
neurons per image. Their terminal training states were also nearly identical,
with 72.6%/72.7% zero weights and 6.19/6.44 active neurons. These branches are
complete and have plateaued about three points below the requested 89%; the
`0.0005` branches remain the preferred candidates.

The preferred `nu_post=0.0005` branches are also complete. Their hash-verified
30k probes scored 86.2% for one trace and 84.5% for two trace, with all 400
neurons assigned, 22.55/22.28 spikes, and 8.28/8.62 active neurons per image.
One trace therefore peaked at 88.4% at 20k, within 0.6 point of the requested
89%, but fell 2.2 points by 30k as its zero-weight fraction rose from 40.8% to
59.0%. Two trace was effectively flat from 84.7% at 20k to 84.5% at 30k, ending
with 54.0% zero weights. The simpler one-trace rule can recover near-reference
accuracy with early stopping around 20k; neither simple rule matches the 89.2%
triplet control at the requested 30k checkpoint under these tuned parameters.

### Full-length dense trace runs

The preferred dense one- and two-trace checkpoints were subsequently extended
from 30,000 to a cumulative 180,000 accepted samples. These are **branched
resumes**, not exact continuations: each case loaded the 30k feedforward weights
and adaptive thresholds, then processed 150,000 new accepted samples after
resetting membrane voltages, conductances, refractory state, delay queues, STDP
traces, Poisson state, and RNG state. The cumulative checkpoint labels are
therefore useful for studying further training, but the 30k-to-40k boundary is
not a checkpoint-exact trajectory.

Both trainers completed their 150,000-sample branches and wrote matched
`XeAe180000.npy` and `theta_A180000.npy` artifacts plus unsuffixed final copies
and artifact hash manifests. Both checkpoint watchers completed all 15 planned
10k-interval evaluations from 40k through 180k and a final-repeat evaluation.
No training abort, watcher stop, or evaluation failure occurred.

The complete preferred-rate trajectory is below. The 10k, 20k, and 30k rows
come from the original uninterrupted runs; rows from 40k onward come from the
branched resumes. Every value is an optimistic simple-demo accuracy on the
same fixed 1,000-image test probe, with assignment and scoring performed on the
same activity.

| Cumulative accepted samples | One trace | Two trace |
| ---: | ---: | ---: |
| 10k | 84.3% | 82.8% |
| 20k | 88.4% | 84.7% |
| 30k | 86.2% | 84.5% |
| 40k | 88.0% | 86.4% |
| 50k | 85.9% | 87.8% |
| 60k | 85.9% | 86.5% |
| 70k | 88.2% | 86.9% |
| 80k | 87.5% | 87.4% |
| 90k | 86.7% | 87.8% |
| 100k | **88.5%** | 87.9% |
| 110k | 88.0% | 87.6% |
| 120k | 87.2% | 86.9% |
| 130k | 86.9% | 87.2% |
| 140k | 87.6% | 87.7% |
| 150k | 86.6% | 87.5% |
| 160k | 88.2% | 87.3% |
| 170k | 86.9% | **88.1%** |
| 180k/final | 86.8% | 87.1% |

One trace reached its full-trajectory maximum of 88.5% at 100k, 0.1 point
above its original 20k maximum, while two trace reached 88.1% at 170k. Neither
rule reached the approximately 89% target on a 1,000-image checkpoint probe,
and neither shows a sustained upward trend at the end. For comparison, the
matched zero-delay midpoint triplet control reached a best 1,000-image probe of
92.7% at 160k and a final 180k probe of 90.5%.

At 180k, the one-trace probe assigned 395 neurons and produced 19.695 spikes
and 6.578 active excitatory neurons per image. The two-trace probe assigned 397
neurons and produced 19.729 spikes and 6.522 active neurons. Their terminal
training diagnostics were also close:

| Terminal diagnostic | One trace | Two trace |
| --- | ---: | ---: |
| Final 1k checkpoint probe | 86.8% | 87.1% |
| Assigned neurons | 395 | 397 |
| Probe spikes/image | 19.695 | 19.729 |
| Probe active E/image | 6.578 | 6.522 |
| Last training block spikes/image | 21.441 | 21.220 |
| Last training block active E/image | 6.253 | 6.259 |
| Last-block retry fraction | 0.50% | 0.70% |
| Feedforward zero-weight fraction | 73.64% | 73.89% |
| Feedforward saturated fraction | 0.149% | 0.115% |
| Theta mean (mV) | 46.625 | 46.464 |

The preceding 10,000-sample training blocks reported 89.53% for one trace and
89.22% for two trace. Those values use order-dependent assignments and activity
from the preceding training block. They are not checkpoint test-set probes and
must not be interpreted as recovery of the reference test accuracy. The fixed
probe results above are the compatible evidence for comparing the two dense
learning rules with the midpoint control.

## Direct-voltage variant

For a conductance impulse of one unit at rest, with zero initial voltage
displacement and conductance, independent RK4 integration gives these extrema:

| Connection | Peak voltage impact | Peak time |
| --- | ---: | ---: |
| XeAe | +0.617408270 mV | 4.647 ms |
| AeAi | +4.445302493 mV | 2.526 ms |
| AiAe | -0.640040884 mV | 7.966 ms |

The initial conversion therefore uses a 48.157845060 mV feedforward column
target, 0.617408270 mV `w_max`, proportionally scaled triplet endpoints,
46.231145927 mV AeAi events, and -16.321042542 mV training AiAe events. The
neuron retains exact passive leak integration but has no `ge` or `gi`; all
synapses update `v` directly.

Brian's `brian_no_units` mode still defines `mV=0.001`. The first voltage
smokes incorrectly inserted manifest values as raw volts, making all converted
weights and STDP limits 1,000 times too large. Those isolated cases are
preserved as invalid unit-mismatch evidence. The corrected implementation keeps
case parameters and diagnostics in mV, stores Brian matrices and learned
checkpoints in volts, and passes a volt-typed direct-delivery verifier. Its
100-sample smoke is stable and scored 30.0% with 168 assigned neurons, 13.17
inference spikes, and 6.8 active neurons per image. A separate 1,000-sample
pilot scored 54.3% with 376 assignments, 14.63 spikes, and 8.65 active neurons,
so the unmodified analytical recurrent conversion was rejected.

Corrected-unit recurrent tuning keeps the analytical 0.617408270 mV
feedforward scale. The one-step recurrent scales (`AeAi=3.0`, `AiAe=0.175`)
scored 89.0% on 100 images with 383 assignments but a broad 30.68 active
neurons. Increasing `AiAe` to 0.30 scored 87.0% with 331 assignments and reduced
activity to 18.46 neurons. Both were screened on 1,000-sample pilots before
promotion. On those probes, the weak-inhibition case scored 68.0%
with 34.09 spikes and 30.09 active neurons; the `AiAe=0.30` case scored 70.9%
with 24.27 spikes and 19.29 active neurons. The latter was promoted as
`full_peakff_reci0300_units_fixed_30000`. A fully one-step-scaled case scored
83.0% on 100 images, but its 25.35 mV
feedforward target caused a 23% training retry rate and was not promoted.

At 10k, `full_peakff_reci0300_units_fixed_30000` scored only 55.3% with all
400 neurons assigned, 17.01 spikes, and 15.99 active neurons per inference
image, versus 80.6% for the triplet control. An inference-only sweep on this
immutable checkpoint scored 60.1% at 1.25x inhibition, 58.1% at 1.5x, and
52.3% at 2.0x; stronger suppression narrowed activity from 13.40 to 9.39 active
neurons but could not recover the missing accuracy. These are readout settings
for one trained network, not separately trained cases. The parent voltage
checkpoint's maximum weight was only 0.142 mV, compared with about 0.387 mV
after scaling the triplet control's 10k maximum into voltage units, so weak
training specialization is the primary diagnosis.

At 20k the same direct-voltage run declined further to a hash-verified 44.5%,
despite assigning all 400 neurons. The probe produced 14.03 spikes and 13.50
active neurons per image. Training retries rose to 21.3%, active neurons
remained broad at 12.12 per image, and the maximum feedforward weight was still
only 0.145 mV. The flat maximum and falling accuracy rule out merely delayed
specialization; the scaled direct-voltage dynamics progressively collapse under
the otherwise matched triplet rule.

The terminal direct-voltage checkpoint confirms that trend. Its hash-verified
30k probe scored 38.3% with all 400 neurons assigned, 12.46 spikes, and 11.90
active neurons per image. Training ended with a 32.2% retry fraction, only a
0.154 mV maximum feedforward weight, and 5.21% zero weights. The compatible
accuracy curve is therefore 55.3% at 10k, 44.5% at 20k, and 38.3% at 30k.
Neither faster STDP nor stronger competition rescued this model in the isolated
pilots, so the direct-voltage approximation does not recover reference accuracy
under the derived peak-impact scaling and tested tuning range.

`ref/zero_delay_voltage_stdp_tuning_v1/` is a new immutable tuning family that
adds a recorded `STDP_RATE_SCALE` while retaining the direct-voltage model,
normalization, integration, and topology. Three isolated 1k screens were run:
3x STDP with `I_TO_E_SCALE=0.30`, 1x STDP with `I_TO_E_SCALE=0.40`, and the
combined 3x/0.40 case. They isolated faster feedforward learning, stronger
training competition, and their combination before any new 30k promotion.

Those first screens rejected both large changes. The 3x/0.30 case scored 42.4%
despite assigning all neurons, while 3x/0.40 scored 35.1% with 388 assignments;
their 82-89 mV theta outliers and low accuracy show winner-dominated learning.
The 1x/0.40 case scored 66.3%, below the original 70.9% pilot. All result
artifacts verify. A narrower follow-up screen therefore tested 1.5x and 2x STDP
at `I_TO_E_SCALE=0.30`, plus 1.5x STDP at the intermediate scale 0.35, in three
new isolated 1k cases with automatic inference evaluators.

The narrower screen also failed to justify faster learning. At 1k, 1.5x/0.30
scored 65.7%, 2x/0.30 scored 66.0%, and 1.5x/0.35 scored 69.4%, compared with
70.9% for the original 1x/0.30 pilot. The latter two tuned cases also developed
70.6-88.5 mV theta outliers. All three evaluation artifacts verify. Since the
base 10k run had 12.65 active training neurons versus 6.18 for the conductance
control, a final competition-focused pilot keeps STDP at 1x, raises
`I_TO_E_SCALE` to 0.50 during training, and uses a 0.75 evaluation multiplier
to reproduce the effective inhibition that gave the best base-checkpoint
inference result.

That final pilot reduced training activity to 6.53 active neurons, but produced
a 75.8 mV theta outlier and scored only 47.7% with 394 assignments. Its
hash-verified result shows that matching the control's mean sparsity by stronger
inhibition creates winner collapse rather than useful specialization. No tuning
case beat the original 1x/0.30 pilot, so no second voltage branch was promoted;
the completed `full_peakff_reci0300_units_fixed_30000` remains the
representative direct-voltage case.

Corrected voltage source SHA-256:
`e3f91f98940a544d26e34800c05684948a5fbe529d2de56ed20b7f81daa5008d`.

Voltage-STDP tuning source SHA-256:
`90db992e4eb478a97fedbdc6924a476be701a1eaed61c2c844488a114d20d06d`.

## Promotion and completion status

All five originally promoted 30k branches and their 10k, 20k, 30k, and
final-repeat evaluators completed. The preferred dense one- and two-trace cases
also completed their branched extensions through cumulative 180k, including all
15 periodic checkpoint probes and final-repeat evaluations. Every reported
checkpoint result below passed its source, checkpoint, activity, and score hash
checks.

| Variant | Selected parameter | 10k | 20k | 30k | Best full-trajectory probe | Final probe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| One trace | `nu_post=0.0005` | 84.3% | 88.4% | 86.2% | 88.5% at 100k | 86.8% at 180k |
| Two trace | `nu_post=0.0005` | 82.8% | 84.7% | 84.5% | 88.1% at 170k | 87.1% at 180k |
| Direct voltage | peak feedforward, recurrent 3.0/0.30 | 55.3% | 44.5% | 38.3% | 55.3% at 10k | 38.3% at 30k |

Both dense simple-trace variants approach the requested accuracy during the
extended trajectory: one trace peaks at 88.5% after 100k cumulative accepted
samples, and two trace reaches 88.1% after 170k. At neither 30k nor the final
180k checkpoint do they recover the matched triplet control's accuracy. Further
training therefore narrows the two-trace gap but does not produce stable
reference-level performance. All training and watcher services exited cleanly;
the individual run and evaluation directories remain preserved under their
family directories.

## Dense one-trace GPU portability check

Date: 2026-08-05. The `nu_post=0.0005` dense one-trace case was ported to the
shared GeNN CUDA runner and trained fresh for 30,000 accepted samples. GeNN's
default postsynaptic delivery was retained after a 100-sample scheduling sweep:
70.143 us/cycle for postsynaptic x1 versus 72.552 us/cycle for the best tested
presynaptic setting, x32.

The complete run used 30,698 attempts and 30,698,000 cycles. Simulation took
2,206.580 s, whole-program time was 2,212.438 s, and throughput was 71.880
us/cycle. The fixed 1,000-image checkpoint probes were:

| Accepted samples | Brian 1 | GeNN CUDA |
| ---: | ---: | ---: |
| 10k | 84.3% | 74.4% |
| 20k | 88.4% | 82.7% |
| 30k | 86.2% | 83.2% |

The GeNN 30k probe assigned 387 neurons and produced 15.313 spikes with 4.420
active neurons per image, versus 400 assignments, 22.551 spikes, and 8.284
active neurons for Brian 1. Its final training block was also quieter at
16.037 spikes and 4.917 active neurons, compared with 20.135 and 6.889. This is
not a dynamics match even though the accuracy gap narrows to 3.0 points by 30k.

NEST-GPU cannot train this exact rule because its synapse callback does not
expose an accumulated source trace. Static inference from the Brian 1 30k
checkpoint scored 85.4%, but used an `aeif_cond_beta` approximation and fired
more broadly at 26.940 spikes and 11.977 active neurons. Brian2CUDA also ran the
real graph for one 1,000-cycle attempt at 448.015 us/cycle; its standalone
controller does not yet provide full adaptive training or checkpoint accuracy.
These two measurements validate runtime portability, not learning equivalence.

Artifacts and exact event-normalized performance counters are linked from
`performance.md`; the GeNN final checkpoint SHA-256 is
`5bc3a6c75488b797a201a664175cea1ff67e43729bcdeca9677c2b999db8bce8`.
