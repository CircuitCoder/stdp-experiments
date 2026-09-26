**The two-class CIFAR-10 run has completed at 54.40%; the other three runs remain active. The longer measurements do not sustain the early accuracy benefit of ×10 threshold adaptation.**

Snapshot: 2026-09-20T16:44:27.985015+00:00 (execution host UTC clock). Progress counts below are the latest complete diagnostic records at that time; training can be ahead by up to 99 accepted images. The launch and configuration record is [FULL-IMAGE-THETA10-20260920.md](FULL-IMAGE-THETA10-20260920.md).

| Workload | Accepted / target | Completed | Latest evaluated checkpoint | Accuracy | Test images | State |
|---|---:|---:|---:|---:|---:|---|
| Fashion-MNIST, 10 classes | 18,100 / 174,000 | 10.4% | 15,000 | 24.05% | 2,000 | Training |
| Fashion-MNIST, labels 0/1 | 29,500 / 34,800 | 84.8% | 25,000 | 70.50% | 400 | Training |
| CIFAR-10, 10 classes | 20,000 / 144,000 | 13.9% | 15,000 | 12.50% | 2,000 | Evaluating 20,000-image checkpoint |
| CIFAR-10, labels 0/1 | 28,800 / 28,800 | 100.0% | 28,800 | 54.40% | 2,000 | Finished, including full test |

The CIFAR-10 two-class case completed all three passes through its 9,600-image training pool, using 142,667 training presentation attempts and 113,867 retries. End-to-end wall time was 10.12 hours, including evaluations. The four experiments shared an RTX 3090, so this is not an isolated platform performance measurement. The two full datasets are still less than halfway through their first training pass.

**Accuracy trajectories.** All nonfinal values below use the same fixed, separate test probe within each case. The final CIFAR two-class value uses the larger official retained test set and is marked separately.

| Accepted images | Fashion 10 classes | Fashion 2 classes | CIFAR 10 classes | CIFAR 2 classes |
|---:|---:|---:|---:|---:|
| 1,000 | 51.90% | 89.50% | 12.45% | 48.75% |
| 5,000 | 34.40% | 64.25% | 11.95% | 56.50% |
| 9,600 | — | — | — | 54.00% |
| 10,000 | 26.80% | 59.50% | 11.70% | 53.50% |
| 11,600 | — | 56.75% | — | — |
| 15,000 | 24.05% | 60.50% | 12.50% | 55.25% |
| 19,200 | — | — | — | 54.50% |
| 20,000 | — | 68.50% | — | 53.75% |
| 23,200 | — | 78.00% | — | — |
| 25,000 | — | 70.50% | — | 53.75% |
| 28,800 | — | — | — | 54.40% (full test) |

Fashion two-class accuracy fell from 89.50% at 1,000 accepted images to 56.75% after the first pass, recovered to 78.00% after the second pass, and was 70.50% at 25,000 images. It is not stable. Fashion ten-class accuracy declined at every completed checkpoint, from 51.90% to 24.05%. CIFAR ten-class probes remain near the 10% chance baseline; the final two-class score is only modestly above 50%.

![Accuracy over accepted training images](../copilot/tmp/workload_full_image_progress_20260920T164427Z/accuracy_progress.png)

**Participation and retry cost.** Recent training activity covers the last 1,000 accepted presentations. Test statistics cover the latest completed evaluation listed above. The largest-neuron share is its fraction of all E spikes in the stated activity window; it is not a per-image winner count.

| Workload | Attempts per accepted image, whole run | Retries, whole run | Recent training active E / 400 | Largest E spike share, recent training | Latest test active E / 400 | Largest E spike share, test |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST, 10 classes | 4.58 | 64,759 | 400 | 0.307% | 379 | 3.05% |
| Fashion-MNIST, labels 0/1 | 4.34 | 98,472 | 400 | 0.370% | 331 | 2.11% |
| CIFAR-10, 10 classes | 5.25 | 84,977 | 400 | 0.289% | 351 | 5.29% |
| CIFAR-10, labels 0/1 | 4.95 | 113,867 | 400 | 0.281% | 367 | 2.81% |

All 400 E neurons participate in each recent training window. The largest neuron accounts for only 0.28–0.37% of recent training spikes, compared with 0.25% under perfect equality. Latest test activity is less uniform, but its largest neuron contributes 2.11–5.29%, far from a single neuron producing most population spikes. Thus poor accuracy currently coexists with broad neuron participation. These measurements support stronger homeostasis balancing activity, but do not establish that single-neuron domination was the only cause of the earlier classifier failure or that the intervention fixes long-term learning.

Increasing the threshold increment changes homeostasis; it does not directly reduce the STDP coefficients. Retries increase both stimulus exposure and the number of opportunities for plasticity. Training currently takes 4.34–5.25 attempts per accepted image over the whole runs, compared with 2.01 in the first 1,000 accepted images of this Fashion two-class run. Comparing accepted-image counts alone does not equalize simulation time or learning exposure.

For the completed CIFAR two-class test, the confusion matrix is:

| Actual class | Predicted airplane | Predicted automobile | Recall |
|---|---:|---:|---:|
| Airplane (0) | 740 | 260 | 74.0% |
| Automobile (1) | 652 | 348 | 34.8% |

Predictions are 69.6% airplane and 30.4% automobile. Assignment produced 207 airplane neurons, 97 automobile neurons and 96 unassigned neurons. The readout is biased, but it predicts both labels, and no test image was silent. Of the 400 E neurons, 367 fired during the full test; neurons silent during assignment remain unassigned even if they later fire during testing.

Latest diagnostic threshold means are 50.55 mV (Fashion full), 40.39 mV (Fashion two-class), 62.54 mV (CIFAR full), and 62.12 mV (CIFAR two-class). These are the adaptive theta state values, not absolute membrane firing thresholds. Weights remain finite and below their cap at the recorded checkpoints, and feedforward column sums stay near 78. The active replacement services have not hit their declared stopping guards at this snapshot. The earlier ceiling-20 failed cases remain distinct artifacts and are excluded from these curves.

**Implementation and evaluation identity.** These runs use the immutable `source_retry60/` snapshot under `copilot/tmp/workload_full_image_theta10_20260920/`, based on Git revision `9ad791ebc67086cb28047b82a884bceb55a565d3` plus the recorded uncommitted source. The current live long-run driver, image helpers, dataset scorer and GeNN backend match the frozen snapshot. All cases use seed 20260724, dense three-trace STDP, 400 E / 400 I neurons, uint32 spike timestamps with FP32 weights/neuron state, zero configured delays, theta increment 0.50 mV, nominal theta decay time 10,000 seconds, 350 ms stimulus and 150 ms blank rest. Each training attempt normalizes incoming columns to 78; fewer than five population spikes triggers an intensity increment and retry, up to intensity 60. The default starting intensity is two. Training inhibition is 25.5; frozen inference uses 17.

Fashion uses 784 inputs and CIFAR uses 3,072 native RGB inputs. No training labels enter STDP. There are 200 held-out training images per retained class for neuron assignment; those images are excluded from all three training passes. Each pass repeats the same seeded shuffled pool. Periodic testing uses 200 separate official test images per class; final testing uses all 1,000 official test images per retained class. Top-two selection uses frequency, then ascending label ID for ties: T-shirt/top and trouser for Fashion, airplane and automobile for CIFAR.

Training remains loaded and paused during each fresh child-process evaluation, then continues with its runtime state intact. Recorded checks confirm frozen weights/theta during inference, an unchanged checkpoint, and preserved training timestep and weights/theta. The saved checkpoints are weights/theta snapshots for inference, not complete restart checkpoints. The earlier 91–92% short-pilot results used different held-out selections and are not points on these long-run curves; the compatible starting comparison here is 89.50%.

**Verification and artifacts.** I verified 31 frozen source hashes, all 25 completed evaluation checkpoint hashes, disjoint assignment/training indices, and all cumulative training-attempt totals. Recomputing predictions, confusion matrices, assignment counts and neuron spike shares from saved activity reproduces all 25 reported scores. This checks the recorded outcomes and their identities; it is not a new simulator-mechanics test or a repeated training seed.

The captured progress logs, launch manifests, detailed statistics and evaluation records are in [summary.json](../copilot/tmp/workload_full_image_progress_20260920T164427Z/summary.json). Every original case manifest contains the complete command, model constants, source hashes, data hashes and selection hashes. Original commands are also in `launch_manifest_retry60.json` and `launch_manifest_retry60_fashion.json`. Exportable plots are available as [PDF](../copilot/tmp/workload_full_image_progress_20260920T164427Z/accuracy_progress.pdf) and [SVG](../copilot/tmp/workload_full_image_progress_20260920T164427Z/accuracy_progress.svg). This status check leaves all active services running.

The completed Morrison result is unchanged: the operationally settled zero-delay run ended at 1,250 simulated seconds with mean 44.450 pA and SD 3.986 pA, versus the paper's 45.65 pA and 3.99 pA. Agreement is close for the distribution width, while the mean is 2.63% lower and firing statistics differ. The endpoint passed two successive declared stability checks; this is not a claim of exact asymptotic stationarity. See [MORRISON-INTEGER-TICKS-20260920.md](MORRISON-INTEGER-TICKS-20260920.md) for the criterion, paper comparison and spike/update counts.
