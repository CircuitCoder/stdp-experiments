**Three runs completed their three-pass targets. Full Fashion-MNIST stopped after 80,206 of 174,000 accepted images because one sample exhausted the retry limit. None of the four services is still running.**

This report checks the ×10 adaptive-threshold runs launched from the immutable `source_retry60/` snapshot. Observation time from the execution host: 2026-09-21T22:51:06.588015+00:00. The preceding report is [IMAGE-PROGRESS-20260920T1644Z.md](IMAGE-PROGRESS-20260920T1644Z.md).

| Workload | Accepted / target | Outcome | Latest evaluated samples | Accuracy | Test size |
|---|---:|---|---:|---:|---:|
| Fashion-MNIST, 10 classes | 80,206 / 174,000 | Stopped: retry limit (46.1% of target) | 80,000 | 49.75% checkpoint probe | 2,000 |
| Fashion-MNIST, labels 0/1 | 34,800 / 34,800 | Completed three passes | 34,800 | 82.20% final | 2,000 |
| CIFAR-10, 10 classes | 144,000 / 144,000 | Completed three passes | 144,000 | 12.47% final | 10,000 |
| CIFAR-10, labels 0/1 | 28,800 / 28,800 | Completed three passes | 28,800 | 54.40% final | 2,000 |

Final testing uses every retained official test image: 2,000 for two-class cases and 10,000 for full CIFAR-10. The full Fashion value is a 2,000-image checkpoint probe; its full-test evaluation was never reached. Each case assigns neurons using 200 separate held-out training images per class, excluded from STDP for all passes. Periodic probes use 200 test images per class. Final scores and probe scores therefore have different sample counts and inference orders.

**The Fashion result recovered substantially after the earlier decline.** Its two-class probe rose from 56.75% at the first-pass boundary (11,600 accepted images) through 78.00% at the second-pass boundary to 83.50% at 30,000. The final 2,000-image score is 82.20%. This is a useful recovery, but it does not establish sustained 91–92% performance from the earlier short pilots, which also used different held-out selections. Within this run the comparable 1,000-image checkpoint probe was 89.50%.

Full Fashion improved from 24.05% at 15,000 accepted images to 39.10% after one pass (58,000), 46.75% at 75,000 and 49.75% at 80,000. The long run stopped during the second pass while this recent trend was upward. It is incomplete and should not be described as converged.

**CIFAR remained weak throughout training.** Full CIFAR probes ranged from 10.55% to 13.65%, and the final 10,000-image score is 12.47%, compared with 10% chance. The two-class final score remains 54.40%, compared with 50% chance. The selected intervention was originally screened only on two-class Fashion; these outcomes do not show that its benefit transfers to CIFAR.

![Accuracy trajectories](../copilot/tmp/workload_full_image_results_20260921T225106Z/accuracy_results.png)

**Fashion retry failure.** The next scheduled image, accepted-sample position 80,207, is training-set index 9,230 (zero-based), label 5, sandal. Only 89 of its 784 pixels are nonzero, and mean pixel intensity is 4.9439 / 255. The same image appeared at position 22,207 in the first pass, when it needed 57 attempts and final intensity 58. On the second pass, all 59 allowed intensity levels from 2 through 60 failed to meet the minimum-five-spike acceptance requirement. The log then records `RuntimeError('Retry intensity exceeded 60')`.

This is the recorded stopping condition. Exact spike counts for the failed attempts were not saved, so the failure record alone does not determine how learned weights and adaptive thresholds combined to suppress that response. The most recent complete diagnostics are at 80,200 accepted images and show finite state, all 400 E neurons active in the recent training window, and no population-wide single-neuron dominance.

![Sparse sandal that exhausted the retry limit](../copilot/tmp/workload_full_image_results_20260921T225106Z/failed_fashion_image.png)

The result records 80,206 accepted images and 333,380 attempts for those accepted images; its attempt and retry totals exclude the final failed image's 59 attempts. The latest immutable checkpoint is at 80,000. These checkpoints contain weights and theta for inference, not all runtime state, so they do not support an exact training continuation after process exit. The stopped case, source snapshot, logs and checkpoints are preserved.

**Activity and compute cost.** Training activity below covers the latest recorded 1,000 accepted images. For the failed case this ends at 80,200. Test activity covers each latest evaluation above. A neuron is active if it emitted at least one spike in that window; the largest-neuron share is a fraction of all E spikes in the window.

| Workload | Active E, recent training | Largest E share, training | Active E, test | Largest E share, test | Attempts / accepted image | Wall hours incl. evaluation |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST, 10 classes | 400 / 400 | 0.375% | 379 / 400 | 2.32% | 4.16 | 22.63 |
| Fashion-MNIST, labels 0/1 | 400 / 400 | 0.375% | 373 / 400 | 2.26% | 4.33 | 11.81 |
| CIFAR-10, 10 classes | 400 / 400 | 0.303% | 398 / 400 | 1.70% | 3.99 | 27.71 |
| CIFAR-10, labels 0/1 | 400 / 400 | 0.281% | 367 / 400 | 2.81% | 4.95 | 10.12 |

All 400 neurons participate in recent training, and the largest contributes only 0.28–0.38% of the spikes; perfect equality would be 0.25%. Latest test activity is also broad: 367–398 active neurons, largest-neuron shares of 1.70–2.81%, and zero silent test images. Poor CIFAR accuracy persists without a single neuron producing most population spikes. Stronger adaptation maintains broad participation in these measurements, but broad participation alone does not ensure useful class selectivity.

The runs shared an RTX 3090 and finished at different times. These end-to-end wall times include separate-process evaluations and are not isolated platform-throughput benchmarks. Changing the threshold increment changes homeostasis, not STDP coefficients; repeated attempts also increase stimulus exposure and opportunities for plasticity.

| Workload | Training attempts for accepted images | Retries | Input spikes | E spikes | I spikes |
|---|---:|---:|---:|---:|---:|
| Fashion-MNIST, 10 classes (spikes through 80,200) | 333,380 | 253,174 | 2,337,003,680 | 1,035,423 | 2,071,414 |
| Fashion-MNIST, labels 0/1 | 150,779 | 115,979 | 1,170,845,201 | 475,693 | 951,733 |
| CIFAR-10, 10 classes | 575,165 | 431,165 | 30,940,140,994 | 1,835,649 | 3,672,437 |
| CIFAR-10, labels 0/1 | 142,667 | 113,867 | 9,228,709,006 | 527,280 | 1,054,706 |

Spike counters include training retries and rest phases; fresh evaluation processes are excluded. The failed case has different accepted-count boundaries for attempt totals (80,206) and last recorded spike/state diagnostics (80,200).

**Final two-class readout.**

| Dataset | Actual class | Predicted label 0 | Predicted label 1 | Recall |
|---|---|---:|---:|---:|
| Fashion | T-shirt/top (0) | 893 | 107 | 89.3% |
| Fashion | Trouser (1) | 249 | 751 | 75.1% |
| CIFAR | Airplane (0) | 740 | 260 | 74.0% |
| CIFAR | Automobile (1) | 652 | 348 | 34.8% |

Fashion assignment produced 222 / 100 neurons for labels 0 / 1 and 78 unassigned. CIFAR produced 207 / 97 and 96 unassigned. The score averages spike counts over neurons assigned to each class. Both classifiers predict both labels: Fashion predicts label 0 on 57.1% of test images, CIFAR on 69.6%. More neurons can fire during testing than were assigned during marking; assignment remains fixed after the marking phase.

**Recorded model state.**

| Workload | Diagnostic sample | Theta mean (mV) | Theta min–max (mV) | Zero weights | Max weight | Column sum min–max |
|---|---:|---:|---:|---:|---:|---:|
| Fashion-MNIST, 10 classes | 80,200 | 42.855 | 38.582–60.000 | 3.79% | 0.410426676 | 77.951–78.000 |
| Fashion-MNIST, labels 0/1 | 34,800 | 39.763 | 36.684–52.985 | 8.81% | 0.324315190 | 77.919–78.000 |
| CIFAR-10, 10 classes | 144,000 | 59.995 | 58.927–64.716 | 74.93% | 1.000000477 | 77.699–78.000 |
| CIFAR-10, labels 0/1 | 28,800 | 62.121 | 61.627–63.193 | 75.11% | 0.562714517 | 77.904–78.000 |

Theta is the adaptive offset state, not the absolute membrane firing threshold. Full CIFAR has a maximum saved weight of 1.00000048 against nominal cap 1; the small overshoot is retained in this report rather than described as strictly below the cap. Its fraction of weights at or above cap is 0.1555%. Column normalization occurs before each training attempt; reported columns are measured after its stimulus and rest.

**Run identity and validation.** The implementation is dense three-trace STDP with 400 E / 400 I neurons, uint32 spike timestamps, FP32 weights/neuron state, seed 20260724 and theta increment 0.50 mV (default 0.05 ×10). Fashion uses 784 inputs and CIFAR uses 3,072 native RGB inputs. Configured delays are zero; GeNN delivers on the following global tick. Stimulus and blank rest are 350 / 150 ms, dt 0.5 ms, column target 78, minimum accepted E-spike count five, starting intensity two, increment one and ceiling 60. Training inhibition is 25.5 and inference inhibition is 17. Assignment and testing have frozen weights/theta. The training pool repeats the same seeded shuffled order for three passes, excluding assignment images throughout. Class-frequency ties use ascending label IDs.

Each checkpoint is evaluated in a fresh child process while training stays loaded and paused; recorded checks verify unchanged checkpoint content, frozen inference weights/theta, and preserved training timestep and weights/theta after evaluation. The current live runner, image helpers, scorer and GeNN backend still match the frozen source. Git revision is `9ad791ebc67086cb28047b82a884bceb55a565d3` plus the manifest-recorded worktree source. No training was restarted or simulator code changed for this check.

I verified 31 frozen source hashes, all 69 completed evaluation checkpoint hashes, assignment/training separation, training-attempt totals and the recorded frozen-state checks. Recomputing scores, confusion matrices, class assignments and activity shares from saved spike activity reproduced all 69 evaluations. Three services exited successfully; full Fashion exited with code 1 and the retry error above.

The [machine-readable results](../copilot/tmp/workload_full_image_results_20260921T225106Z/summary.json) contain all checkpoint scores, original manifests and exact commands, diagnostics, prediction counts, hashes and service states. Captured logs and result files accompany them. The [failure record](../copilot/tmp/workload_full_image_results_20260921T225106Z/fashion_failure.json) identifies the problematic image and its first-pass history. Exportable accuracy plots: [PDF](../copilot/tmp/workload_full_image_results_20260921T225106Z/accuracy_results.pdf), [SVG](../copilot/tmp/workload_full_image_results_20260921T225106Z/accuracy_results.svg). Full launch/protocol details remain in [FULL-IMAGE-THETA10-20260920.md](FULL-IMAGE-THETA10-20260920.md).

**Every completed checkpoint score** (a dash means that case did not evaluate at that count):

| Accepted images | Fashion 10 classes | Fashion 2 classes | CIFAR 10 classes | CIFAR 2 classes |
|---:|---:|---:|---:|---:|
| 1,000 | 51.90% | 89.50% | 12.45% | 48.75% |
| 5,000 | 34.40% | 64.25% | 11.95% | 56.50% |
| 9,600 | — | — | — | 54.00% |
| 10,000 | 26.80% | 59.50% | 11.70% | 53.50% |
| 11,600 | — | 56.75% | — | — |
| 15,000 | 24.05% | 60.50% | 12.50% | 55.25% |
| 19,200 | — | — | — | 54.50% |
| 20,000 | 25.55% | 68.50% | 12.25% | 53.75% |
| 23,200 | — | 78.00% | — | — |
| 25,000 | 25.20% | 70.50% | 12.55% | 53.75% |
| 28,800 | — | — | — | 54.40% (full test) |
| 30,000 | 25.55% | 83.50% | 11.55% | — |
| 34,800 | — | 82.20% (full test) | — | — |
| 35,000 | 29.70% | — | 12.60% | — |
| 40,000 | 30.60% | — | 11.95% | — |
| 45,000 | 35.80% | — | 11.60% | — |
| 48,000 | — | — | 11.70% | — |
| 50,000 | 33.20% | — | 13.30% | — |
| 55,000 | 35.45% | — | 11.75% | — |
| 58,000 | 39.10% | — | — | — |
| 60,000 | 38.40% | — | 13.65% | — |
| 65,000 | 42.50% | — | 12.00% | — |
| 70,000 | 40.95% | — | 12.95% | — |
| 75,000 | 46.75% | — | 10.55% | — |
| 80,000 | 49.75% | — | 12.35% | — |
| 85,000 | — | — | 12.55% | — |
| 90,000 | — | — | 13.00% | — |
| 95,000 | — | — | 11.90% | — |
| 96,000 | — | — | 11.80% | — |
| 100,000 | — | — | 13.20% | — |
| 105,000 | — | — | 12.80% | — |
| 110,000 | — | — | 12.60% | — |
| 115,000 | — | — | 12.80% | — |
| 120,000 | — | — | 12.55% | — |
| 125,000 | — | — | 12.75% | — |
| 130,000 | — | — | 11.90% | — |
| 135,000 | — | — | 12.70% | — |
| 140,000 | — | — | 13.60% | — |
| 144,000 | — | — | 12.47% (full test) | — |
