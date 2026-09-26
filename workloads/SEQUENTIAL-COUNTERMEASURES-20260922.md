**A single background queue runs 16 CIFAR-10 pilots, followed by six full-length Fashion-MNIST experiments. All cases execute sequentially.**

The queue was submitted on 2026-09-22 at 08:43:36 UTC (execution host clock) as `stdp-countermeasures-20260922-083636.service`. The exact commands and source hashes are in the [plan](../copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/plan.json), with the [service launch](../copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/launch.json) and [submission record](../copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/submission.json).

**CIFAR pilot matrix.** Each of the eight conditions below runs on labels 0/1 and on all ten classes, for 1,000 accepted training images per case. The two-class cases execute first, then the ten-class cases. All use seed 20260724, 400 E / 400 I neurons, native RGB inputs, dense three-trace STDP and integer spike timestamps.

| Condition | Theta increment | Scale of both STDP learning rates | Training presentation |
|---|---:|---:|---:|
| baseline | 0.05 mV (1× default) | 1 | 350 ms |
| theta10 | 0.50 mV (10× default) | 1 | 350 ms |
| theta20 | 1.00 mV (20× default) | 1 | 350 ms |
| theta50 | 2.50 mV (50× default) | 1 | 350 ms |
| lr0p1 | 0.05 mV (1× default) | 0.1 | 350 ms |
| lr0p01 | 0.05 mV (1× default) | 0.01 | 350 ms |
| stim100 | 0.05 mV (1× default) | 1 | 100 ms |
| stim50 | 0.05 mV (1× default) | 1 | 50 ms |

The default-adaptation and ×10 cases provide contemporaneous controls. ×20 and ×50 directly test whether stronger adaptation helps CIFAR. The 0.01 learning-rate scale and 50-ms presentation strengthen the other countermeasures relative to the earlier 0.1 and 100-ms pilots. These are separate interventions: learning-rate and short-presentation cases retain the default 0.05-mV threshold increment. Both depression and potentiation coefficients are scaled together.

All pilot evaluations use 350-ms presentations, 150-ms blank rest, 200 held-out training images per class for assignment and 200 separate official test images per class for scoring. Thus two-class scores use 400 test images and full-class scores use 2,000. Assignment images are excluded from the entire training pool. Selection, initialization and training-order hashes are recorded, using the same seeded selections as the previous theta10 family. Identical seeds do not establish bitwise replay of GPU trajectories.

The pilot budget screens activity, retries, numerical health and initial classification. One seed and 1,000 accepted images cannot establish final accuracy or convergence, particularly after the earlier Fashion pilots overstated sustained performance. No CIFAR full training is automatically selected from these short results.

**Six Fashion runs, in execution order after all CIFAR pilots have finished or failed.**

| Queue position | Condition | Classes | Training target | Final test |
|---:|---|---:|---:|---:|
| 17 | lr0p1 | 2 | 34,800 | 2,000 |
| 18 | lr0p1 | 10 | 174,000 | 10,000 |
| 19 | theta4 | 2 | 34,800 | 2,000 |
| 20 | theta4 | 10 | 174,000 | 10,000 |
| 21 | stim100 | 2 | 34,800 | 2,000 |
| 22 | stim100 | 10 | 174,000 | 10,000 |

These are the other three interventions from the prior comparison: both learning rates ×0.1 with default adaptation and 350-ms presentations; threshold increment ×4 (0.20 mV) with default learning rates and 350-ms presentations; and 100-ms training presentations with default adaptation and learning rates. The ×10 intervention is not stacked onto the learning-rate or presentation cases.

Each full case trains for three uninterrupted passes through its fixed seeded training order. Reserving 200 training images per class leaves pools of 58,000 for ten classes and 11,600 for two classes, giving targets of 174,000 and 34,800. Checkpoints occur at 1,000 accepted images, every 5,000, epoch boundaries and completion. Regular evaluation uses the same 200 test images per class; the final evaluation uses all retained official test images. Assignment is always separate from training and testing. Labels 0/1 are T-shirt/top and trouser, selected by ascending ID for class-frequency ties.

**Shared mechanics and stopping rules.** The runs retain normalization to column sum 78 before every training attempt, initial intensity 2, increment 1, minimum five E spikes, retry intensity ceiling 60, and 150-ms blank rest. Training inhibition is 25.5, inference inhibition 17, timestep 0.5 ms and configured delays zero. The integrator, Poisson/Bernoulli input mechanism, STDP timing, theta decay and readout are unchanged.

The previous full Fashion run exhausted the retry ceiling on a sparse sandal. This queue preserves that stopping rule for comparison; it does not skip a difficult image or silently accept a subthreshold response. Retry failures now record the exact sample index, attempt count, last per-neuron spike counts and attempt wall time. Inference failures retain their assignment/test phase and sample index too. A failed case stays in the results and the queue proceeds to the next case. Source/result integrity failures or insufficient free space stop the queue.

Training stays loaded and paused while a fresh child process evaluates each immutable checkpoint. Inference verifies frozen weights/theta; training verifies its timestep and weights/theta before continuing. Only one case advances at a time. Checkpoints remain inference snapshots of weights/theta, not exact training-restart snapshots.

**Implementation and validation.** `long_images.py` now accepts independent threshold increment, learning-rate scale, training duration and inference duration. It records distinct training/inference models and reconstructs the saved variant for evaluation. Old theta10 manifests remain readable. `queue_images.py` runs blocking child processes under an exclusive queue lock and records stage transitions and every case outcome. Source templates and prior experiment artifacts are preserved.

The pre-change suite passed 11 tests. The expanded suite passed 16, including learning-rate/short-presentation metadata, preserved retry/rest ordering, sequential stage order, continuation after a failed case, and refusal of overlapping queue instances. Sequential CUDA smoke runs exercised both Fashion and native RGB CIFAR, reduced learning rates, stronger adaptation, 100/50-ms training with 350-ms inference, immutable checkpoints, preserved training state and frozen inference. See [unit validation](../copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/validation/unit_tests.json) and [CUDA smoke validation](../copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/validation/cuda_smokes.json).

**Resources and artifacts.** The RTX 3090 handles one simulation at a time. The service uses Nice=10, CPUQuota=200%, and OMP/OpenBLAS/MKL thread counts of one. The immutable runtime source is `source/` under the artifact root. All durable results stay under `copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/`. New compilation directories are under `/tmp/stdp_countermeasure_builds_20260922T083636Z` on the separate /tmp filesystem; these can disappear after a machine restart.

At preparation, the artifact filesystem had 2.89 GiB free and /tmp had 3.03 GiB. Estimated additional use is 1.55 GiB of durable artifacts and 2.05 GiB of compilation files. The queue checks each case's estimated storage plus a 512-MiB reserve before starting it. No existing experiment files were deleted.

The queue writes `queue_events.jsonl` as cases start and finish, individual summaries under `case_results/`, `pilot_summary.json` at the transition to Fashion, and `queue_result.json` after all six Fashion cases terminate. Per-case manifests, progress, checkpoints, activity and evaluation records are under `runs/<case>/`, with separate logs under `logs/`. The complete pilot matrix and the six full-length runs are still pending.

**First completed scientific pilot.** At 08:49:14 UTC, the two-class CIFAR control with default adaptation (0.05 mV), default learning rates and 350-ms training completed 1,000 accepted images. It scored **64.50% on 400 separate test images**, with 400 held-out training images for assignment. Recomputing the score from saved activity and checking the checkpoint hash passed. All 400 E neurons fired in testing; the largest contributed 1.02% of test spikes. Assignment was strongly imbalanced (394 airplane / 6 automobile neurons), while the class-normalized readout predicted both labels. This single short control does not establish full-length accuracy or the value of stronger adaptation. The queue then started the matched ×10-adaptation pilot. See [verified first result](../copilot/tmp/workload_sequential_countermeasures_20260922T083636Z/validation/first_pilot_result.json).
