# GeNN STDP tick tolerance and gap guard

This report preserves the float-tolerance implementation and its measurements.
The current implementation uses
[integer tick timestamps](MORRISON-INTEGER-TICKS-20260920.md), as subsequently
requested by the user.

The GeNN Brunel/Morrison STDP version tested here uses the user-requested
`tieTolerance = 0.1 * dt`. It aborts on an absolute timestamp difference strictly
between `0.1 * dt` and `0.9 * dt` in the comparisons that classify simultaneous
events. With `dt = 0.1 ms`, the tolerance is 0.01 ms and the forbidden open
interval is (0.01, 0.09) ms. Exact boundary values are permitted. The clock,
per-synapse timestamps, neuron state, and weights remain FP32.

On an exact tick grid, such a gap is impossible: distinct events are at least
one tick apart. A half-tick tolerance would distinguish zero from one tick
provided timestamp errors stay below half a tick. The requested narrower
tolerance and gap guard additionally detect when that assumption is becoming
unsafe. These checks classify near-zero gaps; they do not round timestamps or
test the fractional part of arbitrarily large elapsed times.

## Implementation and validation

`brunel/ports/genn_port.py` defines both bounds, passes them to the generated
plastic synapse model, and checks the relevant spike and last-update times
before modifying traces or weights. The generated CPU/CUDA assertion terminates
execution when a gap is invalid. On CUDA the host observes the device assertion
at its next error check or synchronization. The bounds and hard-error action
are also recorded in GeNN and Morrison manifests.

The existing relevant suite passed 26 tests before the change. After the change,
34 CPU tests and eight CUDA tests passed. The new generated-kernel tests run
96 event schedules per backend: additive and Morrison rules, both configured
delays (0 and 1.5 ms), six time offsets through 120 seconds, and simultaneous,
adjacent-tick, and existing-trace cases. Empty-history simultaneous pairs leave
weights exactly unchanged. Other updates agree with independently calculated
STDP values within 0.0002 pA, including small FP32 elapsed-time rounding.

Intentional invalid gaps of 0.2, 0.5, and 0.8 ticks terminate child processes.
Separate cases cover each guarded timestamp comparison and a naturally occurring
late-time FP32 discrepancy. Assertions were verified in optimized generated CPU
and CUDA builds. The initial test helper attempted to reload the same sparse
GeNN model object and crashed in its allocator; the successful helper instead
creates fresh model objects and reuses only compiled code. Failed diagnostic
artifacts are preserved alongside the final passing tests.

Tests and implementation snapshots are under
`copilot/tmp/workload_tick_tolerance_20260920/`, including
`tests_before.log`, `tests_after_v2.log`, `tests_cuda_v2.log`, and
`validation_manifest.json`. The source base is Git revision
`9ad791ebc67086cb28047b82a884bceb55a565d3` plus the recorded worktree changes;
these results are not measurements of that committed revision alone.

Reproduce the relevant suites in the repository's GeNN/Nix environment:

```sh
python -m pytest -q brunel/tests workloads/test_workloads.py \
  reimpl/tests/test_genn_normalization.py
GENN_TIMING_TEST_BACKEND=cuda python -m pytest -q brunel/tests/test_genn_timing.py
```

## FP32 duration limit

The earlier defect was a timing-classification error caused by different FP32
rounding paths. For example, at tick 52, global time can be 5.200000286102295 ms
while a reconstructed arrival timestamp is 5.199999809265137 ms. Their
0.000000476837158 ms difference exceeded the old 0.0000001 ms tolerance even
though they represented the same tick. That could create a spurious
approximately 0.461 pA Morrison update. The new tolerance accepts that tie.

At much later times the discrepancy exceeds even the new tolerance. At tick
1,310,721, near 131.0721 simulated seconds, the FP32 global time and a same-tick
reconstructed arrival can differ by 0.015625 ms. This is inside the requested
forbidden gap. Both generated CPU and CUDA late-time regression cases stop on
the assertion. No off-grid event was scheduled: the arithmetic representations
of the same scheduled tick disagree. The guard makes that failure visible; it
does not increase clock resolution.

## Full-size controls

Fresh paired runs were submitted as Slurm jobs 551613 (zero configured delay)
and 551614 (1.5 ms), each on an 80 GB H800. Both use 90,000 E plus 22,500 I
neurons, 810 million plastic E-to-E synapses, 1.265625 billion total recurrent
synapses, seed/state seed 20260724, and the same parameters and settling criteria
as the earlier controls. Only the tie tolerance and gap guard changed. Source
hashes were matched between the local files and the new immutable remote source
snapshot before submission; previous run directories were preserved.

Each run saves population counts and cumulative enqueued synaptic visits every
completed simulated second, and a fixed 100,000-edge weight sample every ten
seconds. These incremental files survive an abrupt CUDA assertion. Counts after
the last complete interval cannot be recovered from a failed GPU context and
must not be included in reported totals. Enqueued visits include pending
deliveries at the interval endpoint; they are not counts of distinct changed
weights or nonzero STDP increments.

Both runs have a 500-second cap. Settling requires two consecutive checks, at
least 200 simulated seconds into the run, over the previous 100 seconds:
absolute fitted standard-deviation drift below 1%, standard-deviation range
below 2%, absolute fitted mean drift below 0.1 pA, and endpoint KS distance
below 0.02. A population rate of 100 Hz also stops the run. The requested timing
guard can prevent the FP32 runs from reaching that settling assessment.

Both jobs finished at their intended guards. Slurm reports `FAILED` with exit
code 1 for each; these are explicit stop conditions, not completed 500-second
or settled runs. The complete numerical record is
[tick-tolerance-results-20260920.json](tick-tolerance-results-20260920.json).

| Configured delay | Outcome | Last preserved weight sample | Mean weight | SD | E rate in last complete second |
|---|---|---:|---:|---:|---:|
| 0 ms | Timing assertion during the 132nd second; 131 seconds complete | 130 s | 45.0974 pA | 2.6756 pA | 3.344 Hz |
| 1.5 ms | Population-rate guard at 66 seconds | 66 s | 110.2836 pA | 106.1212 pA | 165.628 Hz |

The zero-delay log identifies the `postSpikeGap` assertion. Its failure interval
agrees with the independently reproduced FP32 precision limit near 131.0721
seconds. The full-run log does not record the individual failing tick, so the
micro-test's exact tick should not be reported as a directly measured full-run
timestamp. Weight SD grows from 2.506004 to 2.675553 pA between 100 and 130 seconds,
an increase of **6.7657%**. No declared settling check could run before the abort.

The delayed run passes the earlier run's 46-second failure point, but then
becomes unstable: E rate increases to 69.63 Hz at 65 seconds and 165.63 Hz at
66 seconds; I rate reaches 132.76 Hz. Weight SD is 3.7085 pA at 60 seconds and
106.1212 pA at 66 seconds. Unlike the earlier control, this run saves a weight
sample at the rate guard. Its sampled maximum is 418.07 pA, and 32.356% of the
sample exceeds 100 pA. The raw sample is retained; the saved fixed 0–100 pA
histogram alone omits that tail and must not be treated as the full distribution.

The broader tolerance eliminates the demonstrated spurious simultaneous-pair
updates in the tested valid range, but does not remove delayed-network
instability in this seed. Neither corrected run establishes agreement with the
paper's settled distribution, or equality of the two delays' equilibrium
distributions. The timing guard prevents the zero-delay control from reaching
the declared settling window, and the delayed control becomes unstable first.

### Comparisons at equal elapsed times

The sample indices and all four outdegree vectors match the earlier controls.
Full target-index arrays were not copied or compared. The following comparisons
use the same elapsed times and sampling method; they are single-seed controls,
not a multiple-seed stability study.

| Delay and time | Previous mean / SD (pA) | Requested tolerance mean / SD (pA) |
|---|---:|---:|
| 0 ms, 40 s | 45.3493 / 1.9472 | 45.3334 / 1.9324 |
| 0 ms, 130 s | 45.1148 / 2.7090 | 45.0974 / 2.6756 |
| 1.5 ms, 40 s | 46.0312 / 3.4848 | 45.9645 / 3.3667 |

At 40 seconds, the delayed new/old weight-sample KS distance is 0.01403. The
similar early distributions do not predict similar stable endpoints: the new
delayed run still develops a very broad distribution at its rate guard.

![Matched controls and 40-second weight distributions](../copilot/tmp/workload_tick_tolerance_20260920/morrison_tick_tolerance.png)

The last panel compares all four runs at **40 seconds**, before either delayed
run's instability. The large excursion in the mean and SD panels is the new
delayed run's retained 66-second sample. Old-control curves are truncated to
the new run's duration where applicable.
[PDF](../copilot/tmp/workload_tick_tolerance_20260920/morrison_tick_tolerance.pdf)
and [SVG](../copilot/tmp/workload_tick_tolerance_20260920/morrison_tick_tolerance.svg)
versions are also retained.

### Spike and synaptic-visit totals

| Quantity | Zero delay, completed 131 s | Delayed, 66 s including rate-guard interval |
|---|---:|---:|
| Excitatory spikes | 48,855,250 | 107,473,582 |
| Inhibitory spikes | 14,094,956 | 24,327,887 |
| Total recurrent-neuron spikes | 62,950,206 | 131,801,469 |
| E–E presynaptic visits enqueued | 439,695,713,066 | 967,248,227,809 |
| E–E postsynaptic visits enqueued | 439,697,250,000 | 967,262,238,000 |
| Plastic pre + post visits enqueued | 879,392,963,066 | 1,934,510,465,809 |
| All recurrent presynaptic visits enqueued | 708,189,933,373 | 1,482,759,261,107 |
| Neuron time steps | 147,375,000,000 | 74,250,000,000 |
| Simulation wall time, completed intervals | 441.55 s | 914.10 s |
| Process wall time through result capture | 469.47 s | 940.59 s |

External Poisson events are excluded. Plastic visit counts include zero-change
updates and pending endpoint events; they do not count unique changed weights.
The incomplete zero-delay interval is excluded from both its counts and reported
simulation time. The delayed final sample and counts include the complete
interval that trips the rate guard. These guarded diagnostic timings are not a
replacement for the frozen performance benchmark.

Slurm elapsed times are 8 min 13 s and 16 min 1 s. All 53 downloaded run artifacts,
generated-source files, and logs matched remote SHA-256 values, recorded in
`remote_artifact_sha256.json`. The runner source manifest also matches all 27
corresponding local source files. Commands, submissions, Slurm accounting,
raw samples, per-second counts, implementation snapshots, and the analysis
script remain under `copilot/tmp/workload_tick_tolerance_20260920/`.
