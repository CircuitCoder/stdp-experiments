# STDP update locality and buffering report

This report is generated alongside the simulator so that the research basis,
model assumptions, exact trace provenance, validation, and results remain tied
to the implementation. The completed experiment tables and conclusions are
added after the matrix run.

## Scope and fixed assumptions

- Weight size: 4 bytes.
- Physical line sizes: 32, 64, and 128 bytes (8, 16, and 32 weights).
- Payload capacities: 128, 256, and 512 KiB, and 1, 2, and 4 MiB. Metadata
  is reported separately.
- Sparse allocation: both eager-read and deferred-read variants.
- Continuous depression: ordinary write-back and drain-on-pre variants.
- Masked writes are available. A write transaction therefore does not require
  untouched words, although a concrete update still requires the old touched
  value unless it was already cached or memory accepts an update operator.
- Same-tick reconstruction uses input (`X`) events before excitatory (`E`)
  events. The trace does not record Brian callback order, so ordering sensitivity
  is validated separately.
- The active uncommitted Rust conductance implementation is not an input to the
  simulator. Accesses come only from the Brian 1 reference traces identified
  below.

## Trace provenance

The input is the three validated `stdp-firing-trace-v1` JSONL files captured at
10,000, 20,000, and 30,000 accepted presentations from
`zero_delay_midpoint_v1/inhib150_full`. They use Brian 1.4.4 under Python 2.7.18,
real MNIST training data, 784 inputs, 400 excitatory neurons, zero feedforward
delay, 0.5 ms ticks, midpoint conductance integration, triplet STDP, and
normalization to 78 before every presentation. Each trace is a fresh plastic
branch from its matched `XeAe`/`theta_A` checkpoint and contains the same ten
samples with no retries. These are access traces, not accuracy evaluations and
not continuous segments of one training run.

| Checkpoint | Input spikes | E spikes | Logical updates | 32 B accesses | 64 B accesses | 128 B accesses |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 20,644 | 123 | 8,354,032 | 1,128,632 | 612,532 | 364,804 |
| 20,000 | 20,754 | 163 | 8,429,392 | 1,165,492 | 646,642 | 397,594 |
| 30,000 | 20,601 | 206 | 8,401,904 | 1,191,554 | 676,529 | 429,317 |

Every matrix row reproduces these independently calculated counts. Sparse
potentiation contributes 96,432, 127,792, and 161,504 logical updates; the
remaining updates are continuous row depression.

## Previous locality evidence

Earlier analysis used the reference matrix's 8-byte weights and excluded
normalization. Its 64-byte lines therefore contained eight weights, equivalent
to this report's 32-byte/4-byte-weight geometry. At the 30k checkpoint,
postsynaptic operations were only 1.92% of element updates but caused 13.55% of
cold line references and about 25% of estimated misses in an ideal 256 KiB
cache. The disproportion comes from row-major storage: presynaptic depression
walks contiguous 400-weight rows while postsynaptic potentiation walks a
400-weight stride.

Ideal 64-byte-cache simulations previously measured approximately 6-7% hit
rate at 32 KiB, 52-56% at 256 KiB, and 96% at 1 MiB. More than half of synapse
references reused a line within 16 ms and about 90% within 64 ms. Preserving the
actual event order improved the 256 KiB hit rate by roughly 23-25 percentage
points over a shuffled stream. A separate dense-handler profile attributed
98.55% of handler visits to presynaptic triggers. These observations motivated
retaining dirty lines and collecting the sparse column traffic, but also warn
that a model with balanced pre/post firing, such as a two-population Brunel
network, will make the scattered side more important.

Normalization was absent from those locality numbers and from the STDP traffic
matrix below. With 4-byte weights the matrix is 1,254,400 bytes. Even an ideal
single read and write of the complete matrix before each of the ten attempts
would add 75,264,000 bytes across the three traces, before accounting for the
reference's strided column traversal. The reported barrier flushes preserve the
materialization boundary but not this scan traffic.

## Framework review

### GeNN

GeNN supports a single-threaded CPU backend and CUDA. Synapse matrices can be
dense, sparse CSR, bitmask, procedural, or kernel-shared. For sparse
postsynaptic learning it constructs column-remapping data, then maps a column
entry back to the row-major synapse index and updates synaptic state directly.
Kernel fusion, procedural connectivity/weights, and narrow sparse indices can
reduce storage or launch overhead, but the reviewed implementation does not
buffer or combine scattered weight updates.

Source: [single-threaded backend](https://github.com/genn-team/genn/blob/563c45c531eb6adce53ad3ff3f46d614a19abdb2/src/genn/backends/single_threaded_cpu/backend.cc#L2034-L2097),
[matrix types](https://github.com/genn-team/genn/blob/563c45c531eb6adce53ad3ff3f46d614a19abdb2/include/genn/genn/synapseMatrixType.h#L27-L69).

### Brian2 and Brian2CUDA

Brian2's C++ standalone pathway consumes queued synapse IDs and immediately
executes generated vector code on flat synaptic arrays. Event-driven equations
avoid advancing traces on every timestep, but do not defer weight writes.
Brian2CUDA groups triggered synapse IDs by source neuron and executes the same
generated update at each global synapse index; it likewise does not expose a
write-combining buffer for STDP state.

Sources: [Brian2 C++ synapse template](https://github.com/brian-team/brian2/blob/1bfa1a9275bd9672b49f4bf61ffbaf6f7cb55fc9/brian2/devices/cpp_standalone/templates/synapses.cpp#L20-L46),
[Brian2CUDA synapse template](https://github.com/brian-team/brian2cuda/blob/825c0c58d2a0b2bf471af7fc97e184e724522845/brian2cuda/templates/synapses.cu#L58-L160).

### NEST and NEST GPU

NEST's CPU STDP implementation is the closest algorithmic precedent for lazy
materialization. Connections are organized by source. On the next presynaptic
send, a connection fetches postsynaptic spike history since its previous
presynaptic event, replays potentiation into its locally loaded weight, applies
current depression, and transmits using the resulting weight. It moves work
from postsynaptic column traversal to presynaptic connection traversal, but
stores/replays event history rather than algebraically combining arbitrary
weight-dependent updates. NEST GPU maintains sorted source/delay and reverse
connection structures and performs direct weight-dependent STDP updates.

Sources: [NEST `stdp_synapse`](https://github.com/nest/nest-simulator/blob/182eba446a8b89108f21cd2ad54aa4c667afd86a/models/stdp_synapse.h#L222-L275),
[NEST GPU STDP](https://github.com/nest/nest-gpu/blob/830b15ba1d9204346cd5e83eef21a96018daac69/src/stdp.h#L144-L174).

### CARLsim6

CARLsim6 explicitly accumulates event deltas in a separate per-synapse
`wtChange` array. A periodic full pass reads weights and accumulated changes,
applies and clamps them, then decays the change buffer. This reduces accesses to
the transmission weight array, but moves sparse traffic to another full-size
array and intentionally leaves transmission weights stale between update
passes. Its CUDA implementation explicitly identifies some `wtChange` access as
uncoalesced.

Source: [CARLsim6 GPU update code](https://github.com/UCI-CARL/CARLsim6/blob/d527c55afba76f488053fb4c36f6adeebd01a5fa/carlsim/kernel/src/gpu_module/snn_gpu_module.cu#L2443-L2549).

## Exact additive buffering

For a weight bounded by `[L,U]`, any ordered sequence of additive updates with
clamping after each update has an exact constant-size representation

`F(w) = clamp(w + s, a, b)`.

Initialize `(s,a,b)=(0,L,U)`. For each update `delta`, set
`s += delta`, `a = clamp(a + delta,L,U)`, and
`b = clamp(b + delta,L,U)`. This retains both saturation plateaus, which a net
sum loses. For example, on `[0,1]`, `+0.8` followed by `-0.8` produces
`clamp(w,0,0.2)`, not the identity.

The reference triplet rule is easier: deltas depend on traces but not on the
current weight. Potentiations accumulated between presynaptic events have one
sign and collapse to a single sum. At the next presynaptic materialization the
simulator can apply that sum, preserve event order, and then apply depression.
The one- and two-trace derived rules use fractional powers of the current weight;
their deltas cannot be calculated or collapsed without the weight. Exact lazy
execution would have to retain and replay the ordered events.

The traffic simulator assigns one 4-byte buffered value to one payload slot. An
exact mixed-sign `(s,a,b)` transform requires three such values, so its physical
line and capacity numbers should be scaled by three as agreed; an 8-byte value
likewise scales them by two. The access and collision behavior is otherwise the
same under uniform scaling.

Deferred allocation postpones a read; ordinary masked writes alone do not
eliminate the eventual need for the old touched value. The simulator therefore
reports deferred operators separately and materializes them on a later line
read. A forced eviction of an unmaterialized operator is reported as an
operator write, representing a memory-side read-modify-write/update command.

## Simulator contract

The simulator reconstructs a row-major `784 x 400` matrix. An input spike
updates the 400-weight row for that input; an excitatory spike updates the 784
weights in its column. A compact event consists of a physical line number and a
mask of updated 4-byte words. Input events precede excitatory events within a
record. Before every recorded attempt, a barrier writes dirty state and empties
the simulated structure. This enforces normalization as a materialization
boundary but deliberately excludes the normalization scan's own full-matrix
reads and writes, keeping the reported traffic specific to STDP.

The six structures are:

1. Direct memory, with one masked read/write transaction per logical weight
   update. The separate `accesses` counter retains the number of coalescible
   line events as a lower-bound view.
2. A conventional set-associative whole-line cache with true LRU replacement.
3. A fine-grained set-associative cache whose offset lanes have independent
   tags and LRU state.
4. A bounded Robin Hood table containing whole lines.
5. A bounded Robin Hood table containing individual weights. All offsets of a
   backing line have consecutive home slots, and its probe bound is the
   requested distance multiplied by weights per line.
6. A banked bounded Robin Hood table with one tag table per in-line offset.

During bounded Robin Hood insertion, entries with larger displacement take
priority. If the displaced entry cannot remain within its bound, its complete
backing line is evicted. Fine-grained set conflicts likewise choose an LRU lane
victim and then evict every resident word belonging to that victim's backing
line. A dirty coordinated eviction emits one masked line write containing all
resident words of that backing line, including clean words gathered at no extra
transaction cost.

Eager sparse allocation reads the backing line. The updated target word is
mandatory; clean siblings from that read fill only empty slots. Deferred sparse
allocation stores only the touched update operator and performs no allocation
read. `operator_writebacks` count cases where such an operator reaches memory
without first being materialized.

Write-back depression materializes missing/unknown words and retains dirty
state. Drain-on-pre depression materializes the continuously touched words,
combines all resident words for that backing line into one masked write, and
invalidates the complete line. Results report line reads, line writebacks,
words carried by writes, writeback cause, forced bounded evictions, probes,
occupancy, deferred operator occupancy, estimated metadata, and wall time.

## Experiment results

The matrix contains 1,452 configurations per checkpoint and 4,356 rows total:
three line sizes, six capacities, four ways or probe distances, two sparse
allocation policies, two depression policies, six structures including direct
baselines, and ten barrier-separated samples per trace. Direct-memory rows do
not vary by capacity.

### Principal result

For these traces, **deferred allocation plus ordinary write-back is consistently
best**. The expanded capacity sweep locates a sharp knee between 256 and 512
KiB: at 64 bytes, the best 128, 256, and 512 KiB organizations need 1,740,056,
631,908, and 288,171 transactions. The 512 KiB result is only 129 transactions
(0.045%) above the 288,042 unconstrained result. This is consistent with the
largest observed resident working set of 79,422 four-byte payloads, or about
310 KiB, rather than the full 1.196 MiB matrix.

Drain-on-pre invalidates useful row data so frequently that it needs more than
eleven times as many transactions as retained write-back state. At 64 bytes and
1 MiB, a practical 8-way whole-line cache needs 294,416 line transactions
across all three traces, 99.41% fewer than the literal per-weight direct
baseline. The best fine/hash layouts need 288,042, only 2.2% fewer than the
whole-line cache, while their estimated tag metadata is 5.5-9.4 times larger.

The following table fixes 64-byte lines, 1 MiB payload, distance/ways 8, and
aggregates all three traces. `Mode transactions` treats an evicted deferred
operator as one memory-side update command. `Ordinary RMW` adds one read for
each such writeback, representing memory that accepts only concrete masked
data.

| Structure/mode | Reads | Writes | Mode transactions | Ordinary RMW | Traffic bytes | Forced evictions | Metadata |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct, deferred | 24,799,600 | 25,185,328 | 49,984,928 | 50,370,656 | 1,687,915,712 | 0 | 0 |
| Set whole-line, deferred WB | 100,478 | 193,938 | 294,416 | 387,876 | 13,292,396 | 0 | 475,136 |
| Set fine-grain, deferred WB | 98,824 | 190,107 | 288,931 | 380,214 | 13,071,764 | 0 | 4,456,448 |
| Hash whole-line, deferred WB | 98,562 | 189,736 | 288,298 | 379,472 | 13,037,724 | 165 | 475,136 |
| Hash individual, deferred WB | 98,450 | 189,592 | 288,042 | 379,184 | 13,023,260 | 0 | 2,621,440 |
| Hash per-offset, deferred WB | 98,450 | 189,592 | 288,042 | 379,184 | 13,023,260 | 0 | 2,621,440 |
| Hash per-offset, deferred drain | 1,549,975 | 1,645,407 | 3,195,382 | 3,343,941 | 198,836,260 | 0 | 2,621,440 |

The deferred whole-line cache has 93,460 operator writebacks. Adding their
required reads gives exactly the eager whole-line total of 387,876
transactions. Deferred execution still reduces write payload from 3,103,008
words under eager allocation to 1,715,451 words because clean siblings are not
allocated and later written unnecessarily. Thus deferred allocation's large
transaction advantage requires a memory-side update operation; masked absolute
writes alone preserve the payload advantage but not that transaction advantage.

Drain-on-pre performs 1,549,975 coalesced continuous line reads and writes about
24.9 million words. Its 3,195,382 transactions and 198,836,260 transferred bytes
are far below the literal per-weight direct baseline because the continuous row
is still grouped into masked line operations, but far above retained
write-back. It is a poor fit because presynaptic spikes are overwhelmingly
common and row accesses already have high temporal locality. It could become
useful only in a workload where sparse traffic dominates, continuous sweeps are
rare, or the drained data is required elsewhere immediately.

### Line size

The best update-command configurations at each line size are:

| Line | Reads | Writes | Transactions | Ordinary RMW | Traffic bytes |
|---:|---:|---:|---:|---:|---:|
| 32 B | 196,900 | 291,279 | 488,179 | 582,558 | 13,023,260 |
| 64 B | 98,450 | 189,592 | 288,042 | 379,184 | 13,023,260 |
| 128 B | 49,607 | 131,181 | 180,788 | 262,362 | 13,120,136 |

Larger lines reduce transaction count almost exactly in proportion to line
width, but not transferred bytes. The 128-byte case transfers slightly more
because a 400-weight row occupies twelve full lines plus two half-line boundary
cases across alternating rows. Choosing line size is therefore mainly a
transaction/implementation tradeoff under masked writes, not a bandwidth win.

### Capacity, associativity, and bounded probing

For a 64-byte deferred whole-line cache, total transactions are:

| Payload | 1 way | 2 ways | 4 ways | 8 ways |
|---:|---:|---:|---:|---:|
| 128 KiB | 2,073,424 | 1,923,041 | 1,839,053 | 1,798,419 |
| 256 KiB | 1,485,605 | 1,216,161 | 1,041,974 | 932,695 |
| 512 KiB | 1,003,305 | 693,913 | 503,533 | 402,202 |
| 1 MiB | 685,662 | 429,413 | 321,518 | 294,416 |
| 2 MiB | 500,702 | 331,140 | 291,482 | 288,161 |
| 4 MiB | 398,345 | 300,183 | 288,244 | 288,042 |

For the corresponding whole-line bounded Robin Hood table:

| Payload | distance 1 | distance 2 | distance 4 | distance 8 |
|---:|---:|---:|---:|---:|
| 128 KiB | 1,905,918 | 1,897,663 | 1,900,333 | 1,897,675 |
| 256 KiB | 1,344,436 | 1,218,951 | 1,146,659 | 1,096,414 |
| 512 KiB | 912,283 | 638,085 | 486,016 | 421,432 |
| 1 MiB | 633,949 | 371,228 | 298,429 | 288,298 |
| 2 MiB | 471,937 | 305,495 | 288,472 | 288,042 |
| 4 MiB | 383,070 | 291,666 | 288,044 | 288,042 |

At 128 KiB the table remains overloaded across the entire tested probe range,
so a longer bound cannot recover locality. At 1 MiB, increasing distance from
one to eight reduces forced backing-line evictions from 238,288 to 165. At 2
MiB, distance eight eliminates them. Associativity and probe reach matter more
than additional payload once the active per-attempt working set fits; every
attempt barrier also prevents capacity from retaining state across samples.

The best layout at each normalized 64-byte capacity is:

| Payload | Layout | Transactions | Relative to 288,042 | Bounded forced evictions |
|---:|---|---:|---:|---:|
| 128 KiB | 8-way fine set | 1,740,056 | 6.041x | 0 |
| 256 KiB | weight hash, distance 8 | 631,908 | 2.194x | 397,406 |
| 512 KiB | weight hash, distance 8 | 288,171 | 1.00045x | 93 |
| 1 MiB | weight/per-offset hash, distance 8 | 288,042 | 1.000x | 0 |
| 2 MiB | whole-line hash, distance 8 | 288,042 | 1.000x | 0 |
| 4 MiB | whole-line hash, distance 8 | 288,042 | 1.000x | 0 |

The 256 KiB individual-weight table issues many coordinated evictions, but
retains enough useful words to beat every whole-line organization at that
capacity. At 512 KiB it has effectively reached the floor. These payload and
line sizes are normalized to one 4-byte buffered value per weight. An exact
three-value saturating transform maps the 512 KiB/64-byte point to 1.5 MiB of
payload and 192-byte physical lines; 8-byte values map it to 1 MiB and 128-byte
physical lines.

### Ordering sensitivity

Reversing same-record order from X-before-E to E-before-X on representative 30k
configurations changes transaction counts by at most 0.024%. The 8-distance
per-offset write-back case has identical reads and writes in both orders; the
drain case changes writes by 30 out of about 553,000. This demonstrates traffic
robustness only. The transcript cannot establish Brian 1 callback ordering or
numerical equivalence when both populations fire in one tick.

### Simulator wall time

| Checkpoint | Trace bytes | Configurations | Matrix wall time |
|---:|---:|---:|---:|
| 10,000 | 781,638 | 1,452 | 299.186 s |
| 20,000 | 783,898 | 1,452 | 292.158 s |
| 30,000 | 784,100 | 1,452 | 317.695 s |

Total matrix wall time was 909.039 seconds (15 minutes 9 seconds) on one core of
the recorded WSL2 x86-64 host with Clang 21.1.7. Parsing each trace took roughly
8.8-17.6 ms and compact event construction 0.8-10.2 ms, depending on line size
and concurrent host load.
Per-configuration
`simulation_seconds` includes structure initialization, event replay, barrier
flushes, and final flush, but excludes JSON parsing/event construction.

Summed over all 4,356 rows, direct, whole-line set, fine set, whole-line hash,
individual-weight hash, and per-offset hash simulation time was 0.061, 10.929,
100.326, 12.189, 641.445, and 141.825 seconds respectively. These are simulator
implementation costs, not predictions of hardware performance. The individual
layout is expensive to model because its maximum probe distance is multiplied
by 8/16/32 weights per line as specified. Coordinated fine-grained line
operations use a simulator-only reverse locator; it is excluded from modeled
capacity/metadata and never serves a simulated access or changes probe counts.

## Validation and artifacts

The C++ self-test covers all six structures, per-weight direct accounting,
coordinated fine-grained eviction and drain, multiplied individual-weight probe
bounds, set conflicts, barrier/final flushing, and synthetic row/column
reconstruction. Address and undefined-behavior sanitizers pass the self-test and
a full 10k distance-1 hash trace. `validate_results.py` verifies all 4,356 rows
against the exact requested configuration set, known logical counts,
allocation-specific direct reads/writes, hit/occupancy bounds, and
writeback-cause reconciliation.

The expanded 4,356-row matrix was regenerated completely with one current
binary. All 2,196 overlapping 1/2/4 MiB rows match the archived result set in
every non-timing field. The prior result set is preserved under
`results/archive_1m_2m_4m_20260723/`. Raw per-checkpoint CSVs, the three-trace
aggregate, ordering sensitivity, wall-time records, and the run manifest are in
`results/`. The manifest records trace, simulator-source, and executable hashes,
compiler, platform, event order, weight size, capacities, and barrier policy.
Third-party source revisions are pinned in `../3rdparty/README.md`.
