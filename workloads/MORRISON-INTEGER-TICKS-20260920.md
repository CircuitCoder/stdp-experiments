# Integer ticks for GeNN STDP

The zero-delay experiment is complete. See the consolidated
[final report](MORRISON-FINAL.md) for the accepted endpoint, two-trace rule,
per-timestep spike/update counts, H800 runtime and comparison with Brunel.
This document retains the timing investigation and original control results.

The shared GeNN Brunel/Morrison implementation now stores STDP timestamps as
`uint32` tick indices. Event coincidence uses exact integer equality. Elapsed
times are computed by integer subtraction, then converted to the selected
scalar type and multiplied by `dt` for trace decay. FP32 workloads retain FP32
weights, traces, and neuron arithmetic. Each of the three per-synapse timestamp
fields remains four bytes.

This replaces the [FP32 tolerance and gap guard](MORRISON-TICK-TOLERANCE-20260920.md).
There is no fractional-tick gap to test in the new representation. Assertions
instead reject missing triggering timestamps, future timestamps, and clock
overflow. The numerical record and generated-kernel evidence are retained under
`copilot/tmp/workload_integer_ticks_20260920/`.

## Why the FP32 difference was normal rounding

The previous generated GeNN code performs these operations:

```cpp
// runner.cc, each global step
const float t = timestep * 0.1f;
// neuronUpdate.cc, when a neuron spikes
storedSpikeTime = t;
// synapseUpdate.cc, zero configured delay
const float arrivalTime = storedSpikeTime + 0.1f;
```

The global step count is already an integer in GeNN. The first expression
converts it to FP32 before multiplication. This is not repeated `t += dt`
accumulation. The arrival expression follows a different rounding path because
the spike time was stored as an already rounded float. For the delayed control,
the postsynaptic learning path instead adds the FP32 constant `3.1f`.

Binary32 has 24 significant binary digits, including its implicit leading bit.
At 5.2 ms, adjacent FP32 values are separated by
`2^(2 - 23) = 4.76837158203125e-7 ms`. The measured discrepancy is **one ulp**
(one representable spacing), about `9.17e-8` relative to 5.2 ms, or
**0.477 nanoseconds**. It is not an unusually large FP32 arithmetic error.

The stored value of `0.1f` is `0.10000000149011612`. Explicitly rounding each
operation gives:

| Operation | Rounded FP32 value (ms) | Rounding error relative to exact operation on its stored operands |
|---|---:|---:|
| `52 * 0.1f` | 5.200000286102295 | +0.4375 ulp |
| `51 * 0.1f`, stored at emission | 5.099999904632568 | −0.359375 ulp |
| Stored emission time `+ 0.1f` | 5.199999809265137 | −0.203125 ulp |

Every operation is within half an ulp of its exact result on the stored inputs.
The two final values have adjacent encodings, `0x40a66667` and `0x40a66666`.
Their difference is precisely `4.76837158203125e-7 ms`. Both represent the same
scheduled tick; using their difference to decide exact event coincidence was
the implementation defect. The small timestamp discrepancy selected a different
STDP update branch, causing the much larger approximately 0.461 pA weight error.

The same mechanism explains the later failure: near 131,072 ms, one FP32 ulp is
0.015625 ms, inside the previously requested (0.01, 0.09) ms forbidden interval.
At still longer times, even the integer-to-FP32 conversion loses tick identity:
tick 20,000,001 rounds to 20,000,000 before multiplication. An absolute-time
tolerance cannot recover the lost information.

The independent calculation in
`copilot/tmp/workload_integer_ticks_20260920/fp32_arithmetic.py` records exact
rational intermediates, rounded values, bit patterns, and operation errors in
`fp32_arithmetic.json`. It matches the old generated `runner.cc`,
`neuronUpdate.cc`, and `synapseUpdate.cc` retained in the preceding guard tests.

## Integer implementation

Each neuron increments an integer counter once per simulation step, including
refractory steps. Its spike reset stores the counter in `lastSpikeTick`. The
counter starts at zero and the first emitted spike has timestamp one, reserving
zero for "no previous spike". GeNN automatically buffers this referenced neuron
variable through the configured axonal and postsynaptic delay queues.

An emission on global tick `n` stores `n + 1`. Adding the configured integer
delay `d` gives delivery index `n + d + 1`, accounting for GeNN's synapse-before-
neuron update order. Each event handler obtains its current delivery tick from
the triggering spike's buffered integer timestamp. It compares those indices
and updates `lastTraceTick`, `lastPreUpdateTick`, and `lastPostUpdateTick` as
integers. Only `(currentTick - lastTraceTick) * dt` enters floating-point trace
decay. There is no conversion from a rounded absolute float back to an integer.

GeNN still computes its floating `t` argument for its general runtime interface;
the current Brunel neuron/STDP snippets do not use it to drive state or plastic
event timing. No GeNN vendor-source patch or per-step host/device clock transfer
is needed. Additional clock state scales with the neuron count, while the
per-synapse timestamp storage is unchanged from the FP32 version.

The counter guard reserves enough room for the largest configured delay before
`uint32` overflow. At `dt = 0.1 ms`, the representable duration is approximately
429,496.7 simulated seconds, or 4.97 days. The process aborts before wrapping.
Moving GeNN's public `model.timestep` independently does not move these neuron
clocks; the regression setup explicitly synchronizes the counters when testing
large absolute-time offsets. Normal continuous runs advance them together.

## Validation

The pre-change CPU suite passed 34 tests. After the change, the relevant CPU
suite passed **37 tests**, and all **11 timing tests passed on CUDA**. The
existing generated zero-delay current-delivery smoke test also passed.

For each backend, 208 generated-kernel schedules cover both additive and
Morrison rules, configured delays 0 and 1.5 ms, simultaneous and adjacent-tick
pairs, an existing presynaptic trace, and 13 absolute offsets. Offsets include
the former 131-second failure, 200/500/2,000 seconds, the FP32 integer precision
boundary, signed-32-bit boundary, and timestamps near `uint32` exhaustion.

The same event schedule produces **bitwise identical final weights at every
tested offset** on a given backend. Empty-history simultaneous pairs leave
weights exactly unchanged. Other updates agree with independent analytic STDP
values within 0.00001 pA. Tests also verify exact stored integer update indices
and advancement of both E and I neuron clocks through refractory periods.

Ten subprocess cases verify hard failures for missing or future timestamps,
trace/update timestamps later than the triggering event, and clock/arrival
overflow. Both CPU and CUDA checks use optimized generated code. Logs are
`tests_before.log`, `tests_cpu.log`, and `tests_cuda.log` under the artifact root;
the pre-change and final source snapshots are preserved alongside them.

```sh
python -m pytest -q brunel/tests workloads/test_workloads.py \
  reimpl/tests/test_genn_normalization.py
GENN_TIMING_TEST_BACKEND=cuda python -m pytest -q brunel/tests/test_genn_timing.py
```

## Full Morrison controls

Jobs 551713 (zero configured delay) and 551714 (1.5 ms) use separate 80 GB H800
allocations and a fresh immutable source snapshot. Both retain the full
90,000 E / 22,500 I network, 810 million plastic and 1.265625 billion total
recurrent synapses, seed/state seed 20260724, and the prior controls' parameters.
Only the timestamp representation and its checks changed. Source hashes were
matched before submission. The source base is Git revision
`9ad791ebc67086cb28047b82a884bceb55a565d3` plus the recorded worktree changes.

Each run saves per-second population and enqueued synaptic-visit counts, and a
fixed 100,000-edge weight sample every ten seconds. The cap is 500 seconds with
the same 100 Hz population-rate guard. Settling requires two consecutive checks,
from at least 200 seconds, over a trailing 100-second window: absolute fitted SD
drift below 1%, SD range below 2%, absolute fitted mean drift below 0.1 pA, and
endpoint KS distance below 0.02. No parameter retuning accompanies this change.

Both initial controls have finished:

| Configured delay | Outcome | Simulated duration | Final sampled mean | Final sampled SD |
|---|---|---:|---:|---:|
| 0 ms | Duration cap; not settled | 500 s | 44.737274 pA | 3.505538 pA |
| 1.5 ms | Population-rate guard | 63 s | 92.766260 pA | 77.502081 pA |

The zero-delay 400–500-second SD rises from 3.380148 to 3.505538 pA; fitted
relative drift is 3.6175% per 100 seconds. All seven settling checks fail.
The delayed endpoint has E/I rates of 203.767 / 155.180 Hz, and 23.946% of the
sampled weights exceed 100 pA. Neither endpoint is an equilibrium measurement.
The integer timing fix therefore does not by itself make the delayed port
reproduce the paper. This remains a comparison within the current GeNN port,
with next-tick transport and its existing STDP update scheduling, rather than
an exact reproduction of the paper's historical simulator.

The initial controls' logs, samples, manifests and generated source were
downloaded and **94 files matched remote SHA-256 hashes**. Their Slurm records,
verification record and exact commands are retained under the artifact root.

The user's request for a stabilized endpoint also launched job **551753**, a
fresh zero-delay control with a 2,000-second cap and the same stopping criteria.
It uses the same immutable integer-timestamp source snapshot. The 500-second
run was already active with a fixed cap; sampled weight files do not contain
enough runtime state for exact continuation, so the extended control starts
from the recorded seed rather than pretending to resume from a weight sample.

## Stabilized zero-delay result

The extended run **passed the predeclared settling test at 1,250 simulated
seconds**. Checks at 1,200 and 1,250 seconds both passed all four requirements.
This is an operational finite-window settling criterion, not a proof of
asymptotic stationarity. No stopping threshold was relaxed after launch.

| Measurement | Zero delay, 1,250 s | Paper, Figure 4B | Difference |
|---|---:|---:|---:|
| Mean E–E weight | **44.450353 pA** | **45.65 pA** | −1.199647 pA (−2.6279%) |
| Weight population SD | **3.986298 pA** | **3.99 pA** | −0.003702 pA (−0.0928%) |

Over 1,150–1,250 seconds, SD changes from **3.956305 to 3.986298 pA**, an
endpoint increase of **0.7581%**. The fitted SD drift is **0.7307% per
100 seconds**, SD range/mean is 0.7554%, fitted mean drift is −0.01214 pA,
and endpoint KS distance is 0.00477. The preceding passing window,
1,100–1,200 seconds, has fitted SD drift 0.8307%. The measured changes are
small but nonzero; the full trajectories remain available for judging whether
a stricter settling rule is appropriate.

The endpoint distribution is unimodal and approximately bell-shaped, with
skewness 0.220 and excess kurtosis 0.097. Its width is close to the published
value, while its center is lower. Mean E/I firing rates over the final
100 seconds are **2.2153 / 3.1709 Hz**. The paper reports approximately 8.8 Hz
for its plastic network. Thus the near match in weight SD does not imply a
match in the full network dynamics.

Our endpoint statistics use 100,000 fixed valid E–E edges. Pooling the eleven
snapshots from the final 100 seconds gives mean **44.455899 pA** and SD
**3.970591 pA**; these repeated synapse samples are correlated. The paper uses
8.1 million synapses and averages ten histograms from the final 500 seconds of
a 2,000-second run. The sampling/window protocols therefore differ.
[Morrison et al., Section 4.1 and Figure 4](https://brainworks.biologie.uni-freiburg.de/2007/journal%20papers/morrison-neco-2007.pdf).

The delayed control did not stabilize: it ended with a broad, multimodal
distribution at the rate guard. Consequently, the two delay variants do **not**
support a shared equilibrium-distribution claim. The integer fix removes the
confirmed tick-coincidence defect; remaining differences from the historical
paper implementation and its dynamics have not been isolated by this control.

![Weight statistics, drift tests, firing rate and endpoint distributions](../copilot/tmp/workload_integer_ticks_20260920/morrison_integer_settling.png)

The dashed density is a Gaussian constructed from the paper's mean and SD,
not digitized paper data. Standalone [PDF](../copilot/tmp/workload_integer_ticks_20260920/morrison_integer_settling.pdf)
and [SVG](../copilot/tmp/workload_integer_ticks_20260920/morrison_integer_settling.svg)
are available. The complete numerical summary is
[integer-timestamps-results-20260920.json](integer-timestamps-results-20260920.json).

## Spike and synapse-update counts

All columns use the complete simulated interval indicated, including the
guard-triggering second of the delayed case. The extended zero-delay run
contains 1.40625 trillion neuron timesteps.

| Quantity | Zero delay, 500 s control | Zero delay, settled at 1,250 s | 1.5-ms control, 63 s |
|---|---:|---:|---:|
| E spikes | 142,757,491 | **300,514,404** | 96,619,942 |
| I spikes | 44,702,273 | **99,947,688** | 22,193,526 |
| E + I spikes | 187,459,764 | **400,462,092** | 118,813,468 |
| E–E presynaptic visits enqueued | 1,284,811,075,063 | **2,704,609,790,754** | 869,578,118,755 |
| E–E postsynaptic visits enqueued | 1,284,817,419,000 | **2,704,629,636,000** | 869,579,478,000 |
| Plastic pre + post visits | 2,569,628,494,063 | **5,409,239,426,754** | 1,739,157,596,755 |
| All recurrent presynaptic visits | 2,108,922,721,769 | **4,505,198,839,619** | 1,336,658,222,324 |
| Simulation wall time | 1,456.77 s | **3,122.33 s** | 945.13 s |

These count event-handler visits, including zero weight increments, rather
than distinct synapses changed. They exclude the implicit external Poisson
drive and include deliveries enqueued but still pending at the endpoint.
The first two count rows sum the per-second population counters; outgoing
visits use the actual per-neuron outdegree vectors. Postsynaptic E–E visits
use the fixed indegree of 9,000.

The extended process took 3,154.74 wall seconds including compilation,
diagnostics and output. Its initial measured GPU allocation was 28,642 MiB
on the allocated H800 80 GB, the same as the 500-second integer control;
the delayed control used 28,666 MiB. This is a measured process allocation,
not a separately sampled peak-memory estimate.

The extended run's **144 downloaded files matched remote SHA-256 hashes**,
bringing the total to 238 verified files across all three jobs. Slurm marked
the extended job completed successfully. At matched historical times, all
sample positions and four outdegree vectors match the old float controls;
complete connectivity target arrays were not compared. At 500 seconds, the
integer zero-delay sample has mean/SD 44.737274/3.505538 pA versus the old
44.759832/3.549163 pA, with KS distance 0.00591. These modest aggregate
differences do not invalidate the deterministic demonstration of the old
coincident-event timing defect.

The separate [longer image runs](FULL-IMAGE-THETA10-20260920.md) also use an
opt-in integer timestamp representation for three-trace learning. Their
neuron-sized timestamp history preserves 0.5-ms event identity beyond the
FP32 clock's precision limit. This does not alter the Morrison implementation
or the immutable source used by these three cluster jobs.
