**Progress at 2026-09-22 13:54:49 UTC: all 16 CIFAR pilots have ended (12 completed evaluations, four retry failures). One of six Fashion runs is complete, one is running, and four remain queued. The user service is active and cases remain sequential.**

The completed Fashion two-class run with both learning rates ×0.1 reached **92.25% on the full 2,000-image test set**, compared with 82.20% for the previous adaptation ×10 run. This is a 10.05-percentage-point improvement under matched initialization, training order, assignment and test selections. It is a single-seed comparison.

The ten-class learning-rate ×0.1 run has accepted **97,000/174,000 images (55.7%)**. Its latest completed checkpoint, at 95,000 accepted images, scored **67.00% on the fixed 2,000-image probe**. This is an interim score; its final evaluation will use all 10,000 official test images. A simple elapsed-time extrapolation gives roughly 1.7 more hours for this case, subject to retry costs and checkpoint/final inference. It is not an estimate for the entire remaining queue.

**CIFAR pilots: 1,000 accepted training images, one seed.**

| Intervention | Two classes (400 test images) | Ten classes (2,000 test images) |
|---|---:|---:|
| Default control | 64.50% | 10.00% |
| Adaptation ×10 | 55.25% | 11.75% |
| Adaptation ×20 | Retry failure during assignment | 11.70% |
| Adaptation ×50 | Retry failure after 70 training images | Retry failure after 77 training images |
| Both learning rates ×0.1 | 50.00% | 15.05% |
| Both learning rates ×0.01 | 60.75% | 17.50% |
| 100-ms training presentation | Retry failure during assignment | 12.35% |
| 50-ms training presentation | 59.25% | 10.00% |

The stronger-adaptation hypothesis is unsupported by these pilots: ×20 did not improve ten-class accuracy relative to ×10, while ×50 could not train even 100 images. The strongest learning-rate reduction was best on ten classes (17.50%, versus 10.00% for the control), but remains weak and needs a longer run before making a final-performance claim. The unmodified control was best for the two-class subset at this short budget. No additional CIFAR full runs were selected or launched.

All four failures exhausted the existing maximum input intensity of 60. The two-class ×20 and 100-ms cases completed training but failed during frozen assignment on the same image (assignment position 245, filtered training index 2653). Each exhausted 59 attempts and produced zero excitatory spikes on its last attempt. Their final training mean adaptive thresholds were about 68.17 and 66.72 mV respectively. The ×50 cases stopped after 70 and 77 accepted training images for two and ten classes respectively, also with zero spikes on their last failed attempts. These failed cases are preserved; the queue continued as designed. Successful-attempt counters exclude the final failed presentation's attempts.

Accuracy alone does not identify the failure mechanism. The ten-class control had only one test-active excitatory neuron and predicted one class. However, the two-class learning-rate ×0.1 pilot and ten-class 50-ms pilot each had all 400 neurons active, yet all neurons were assigned label 0, producing 50% and 10% accuracy. Those cases exhibit a collapsed class assignment without one neuron dominating total spikes. The best ten-class learning-rate ×0.01 pilot still assigned 376/400 neurons to label 0, so broad firing alone has not fixed class specialization.

**Fashion full-training status.**

| Order | Intervention | Classes | Accepted training images | Status / latest score |
|---:|---|---:|---:|---|
| 17 | Both learning rates ×0.1 | 2 | 34,800 / 34,800 | Complete: 92.25% on all 2,000 tests |
| 18 | Both learning rates ×0.1 | 10 | 97,000 / 174,000 | Running: 67.00% at 95,000, 2,000-image probe |
| 19 | Adaptation ×4 | 2 | 0 / 34,800 | Queued |
| 20 | Adaptation ×4 | 10 | 0 / 174,000 | Queued |
| 21 | 100-ms training presentation | 2 | 0 / 34,800 | Queued |
| 22 | 100-ms training presentation | 10 | 0 / 174,000 | Queued |

The completed two-class learning-rate run required 46,816 attempts for 34,800 accepted samples (1.35 attempts/image), compared with 150,779 attempts (4.33/image) for adaptation ×10. All 400 neurons were test-active, with 349 assigned T-shirt/top and 51 assigned trouser; there were no unassigned neurons or silent tests. Its most active neuron contributed 0.55% of test spikes. The final confusion matrix was [[947, 53], [102, 898]], with actual classes as rows and predictions as columns.

At the matched 80,000-image checkpoint, ten-class learning-rate ×0.1 scored 64.15% versus 49.75% for adaptation ×10 on the same 2,000-image probe. The old run subsequently stopped at 80,206 accepted images. The new run passed that difficult sparse sandal in both its first and second passes, requiring 20 and 16 attempts respectively. The latest evaluation has 400 active excitatory neurons, 0 unassigned neurons and 0 silent tests. These are encouraging checkpoints, not evidence of convergence yet.

**Protocol and implementation identity.** These measurements use the frozen GeNN CUDA source in [the queue source snapshot](../copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/source/), based on Git revision 9ad791ebc67086cb28047b82a884bceb55a565d3 plus the recorded worktree. The relevant live source hashes still match this snapshot. The [launch plan](../copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/plan.json) records all 22 complete commands, source hashes, resource limits and exact intervention values. The [original launch report](SEQUENTIAL-COUNTERMEASURES-20260922.md) describes the implementation and prelaunch validation.

All cases use 400 excitatory and 400 inhibitory neurons, three-trace STDP, FP32 weights/neuron arithmetic, integer spike timestamps, dt=0.5 ms, and configured zero delay. Fashion uses 784 grayscale inputs; CIFAR uses 3,072 native RGB inputs. Training labels are excluded from STDP. Top-two filtering retains labels 0 and 1, breaking frequency ties in ascending label order. Seed is 20260724. Each assignment set has 200 held-out training images per class, excluded from every training pass; scoring uses separate official test images. Frozen inference always uses 350-ms presentations, 150-ms blank rest, inhibition 17, and disables plasticity and threshold adaptation. Training retains inhibition 25.5, column normalization to 78 before every attempt, blank 150-ms rest, at least five E spikes and the same intensity/retry limits. Learning-rate and shorter-presentation cases retain default adaptation; interventions are not stacked.

Fashion full training is three passes through the remaining pool: 11,600 images for two classes and 58,000 for ten classes. Intermediate probes have 200 tests per class; final tests use the complete official retained test set. Do not join the final full-test score to a fixed-probe curve without marking the protocol change. Comparison against previous adaptation ×10 verified matching hashes for initial weights, initial thresholds, connectivity mask, training order, marking set, probe set and full test set. Matching seeds do not prove bitwise GPU trajectory replay. Wall-clock training times are not a controlled intervention comparison because the earlier family ran concurrent cases.

**Verification and artifacts.** Independently recomputed all 43 completed evaluations available in this snapshot from saved spike activity, including confusion matrices, assignments, silence and neuron activity. All accuracy, checkpoint hash and training-manifest hash checks passed; manifests disable plasticity and adaptation, and evaluation results record unchanged checkpoints and frozen weights/theta. The latest old Fashion endpoints were independently verified too. This status check did not alter simulator code or queued experiments.

The [dated machine-readable snapshot](../copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/status_20260922T135449Z/summary.json) preserves progress, all completed evaluations, failed-case evidence, comparison identities and verification results. The [pilot summary](../copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/pilot_summary.json) contains all 16 pilot outcomes. Detailed checkpoints, activity, logs and manifests remain under the queue's runs directory.

Service: stdp-countermeasures-20260922-083636.service in the user manager. It is active/running, with Nice=10, CPUQuota=200%, single-thread library limits, and one advancing simulation at a time; training pauses while its checkpoint child evaluates. Free space at capture was 2.57 GiB on the artifact filesystem and 2.47 GiB in /tmp. The queue retains its free-space guards. Four remaining Fashion cases will start automatically in the listed order.
