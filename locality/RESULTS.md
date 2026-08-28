# Synapse-update locality results

## Conclusion

The five selected workloads do not support one common description of synapse
locality.

- MNIST input traffic is strongly spatially local: the accumulated 28x28 input
  spike maps have four-neighbor Moran's I of about 0.978, and the busiest 10%
  of pixels account for about 36.7% of feedforward traversal mass.
- That locality applies directly to triplet weight-update attempts, because the
  triplet rule assigns the weight on both pre- and postsynaptic event paths. It
  does not apply directly to one-trace weight assignments: the one-trace rule
  transmits continuously on presynaptic events but assigns the weight only on
  postsynaptic spikes.
- Brunel E-E updates are temporally bursty at the 0.1 ms simulation tick but
  occur in every 100 ms interval. Their frequency graph is almost completely
  explained by neuron firing counts. No firing-trajectory community structure
  emerged beyond the directed configuration-model null in either STDP rule.
- Extending Brunel from 1 s to 10 s makes the full-run edge-frequency graph
  more uniform, not more modular. The absolute sampled weight change is only
  weakly to moderately rank-correlated with update frequency, so event count is
  not a proxy for plastic effect.

The negative Brunel community result is also a property of the proposed
observable. For every structural E-E edge `(i, j)`, both rules execute a weight
assignment on each delayed presynaptic arrival and each postsynaptic spike, so

```text
update_frequency(i, j) = delayed_pre_spikes(i) + post_spikes(j).
```

Consequently, an update-frequency graph can reveal firing-rate heterogeneity
and fixed connectivity, but it cannot by itself reveal synapse-specific learned
assemblies. A follow-up aimed at spontaneous learned structure should construct
graphs from signed or absolute `delta_w` in short windows, or from lagged spike
coactivity, and compare those graphs with the update-frequency null measured
here.

## Event definitions

The analysis keeps three quantities separate:

1. A **synapse traversal** is execution of a pre- or postsynaptic handler.
2. A **weight-update attempt** is execution of code that assigns the plastic
   weight, including assignments that produce zero numerical change because a
   trace is zero or a bound is active.
3. **Net weight change** is the before/after difference. MNIST was snapshotted
   at every 500 ms presentation attempt. Brunel net changes were measured on a
   deterministic sample of 200,000 of the 81,000,000 E-E edges.

The triplet MNIST and both Brunel rules assign weights on pre and post paths, so
their traversal and update-attempt counts are equal. The one-trace MNIST rule
uses its pre path only for transmission and trace maintenance; weight assignment
is post-driven.

Temporal histograms use the native simulation timestep, 0.5 ms for MNIST and
0.1 ms for Brunel. The raw artifacts also contain 1, 10, and 100 ms aggregates
and sampled per-edge inter-update gaps.

## Protocol and provenance

All measurements used commit `0016f27144bd056ea31a3cd7da274bb89bb0e0ad`
plus the uncommitted locality instrumentation listed below. They ran with
PyGeNN 5.4.0 in FP32 on an NVIDIA RTX 3090 with driver 596.49.

MNIST used real MNIST from `data/mnist`, sequential training images starting at
index 10,000, and the three committed 10,000-accepted-sample checkpoints. Each
case measured 100 additional accepted samples with plasticity enabled, including
all retries, 350 ms stimulus, and 150 ms rest. Seed 0 and the selected GeNN
parallelism settings match `genn-sweep`: triplet/dense and one-trace/dense use
PostSpan x1; one-trace/12.5%-sparse uses PreSpan x32.

| MNIST case | Checkpoint SHA-256 | Attempts / retries | Simulated time |
|---|---|---:|---:|
| Triplet dense | `e4cef93ef2ad8c8e93b7d3d1b93b28276d62616f1e3b59b4dbc0db99149becce` | 102 / 2 | 51.0 s |
| One-trace dense | `eab026d59dad20f1dd979f800e6a37e3f8a3e2b0386febb3f9b1fe672e44d289` | 101 / 1 | 50.5 s |
| One-trace sparse 0.125 | `c32bf0269d8dd33879a7ecfadc096e078cb4e6967d0d58215a1fe061253cc82d` | 100 / 0 | 50.0 s |

Brunel used the scale-1 network (9,000 E, 2,250 I, 81 million plastic
E-E edges), seed and state seed 20260724, 100 ms presimulation, 1.5 ms
presynaptic delay, arrival-timed STDP, and the NEST causal boundary convention.
Both 1 s and 10 s measurement windows start from a fresh seeded state. A 100 Hz
excitatory-rate guard was checked every 100 ms; all four runs completed their
requested duration.

Commands, excluding the Nix/CUDA environment wrapper documented in
`locality/README.md`, were:

```sh
python locality/run.py \
  --output locality/runs/selected_1s_20260826_a

python locality/run.py \
  --cases brunel_additive brunel_morrison \
  --brunel-sim-ms 10000 \
  --output locality/runs/brunel_10s_20260826_a

python locality/run.py \
  --cases brunel_morrison \
  --brunel-sim-ms 10000 \
  --output locality/runs/brunel_morrison_10s_20260827_b
```

The first 10 s Morrison process completed its simulation but filled the disk
while writing `summary.json`. Its incomplete directory is retained at
`locality/runs/brunel_10s_20260826_a/brunel_morrison`; no result is taken from
it. The successful fresh rerun is the third command above. Its first-1-second
metrics exactly reproduce the separate selected 1 s Morrison run. The additive
10 s run likewise exactly reproduces its separate selected 1 s run in its first
window.

Instrumentation source hashes at execution time:

| File | SHA-256 |
|---|---|
| `locality/analysis.py` | `e51498d213959e2cc7b0068c0645d7638fd38b8223065c3790ad99727a299792` |
| `locality/plotting.py` | `9f40c015265cf772e11a3b38bd288170426403571087cd08229d467f2be099a0` |
| `locality/run.py` | `1da1ba059b2ac1109cd52dd4e8bd121dc9f8e5d137e63528f3723a638422e0a0` |
| `reimpl/backends/genn_backend.py` | `9d65788c2a7de4c53c8e8230175b88d8ea8298edd4a43fbfa48169449fa4e4a4` |

No classification evaluation was run. These are locality measurements during
plastic training, not accuracy measurements.

## Temporal locality

| Workload | Traversals | Weight attempts | Attempts / traversals | Native-tick idle | Native-tick CV | 10 ms idle | 100 ms idle |
|---|---:|---:|---:|---:|---:|---:|---:|
| MNIST triplet dense | 97,613,504 | 97,613,504 | 100% | 33.94% | 1.013 | 29.90% | 20.00% |
| MNIST one-trace dense | 96,821,408 | 1,087,408 | 1.123% | 98.87% | 11.318 | 80.73% | 20.59% |
| MNIST one-trace sparse 0.125 | 12,124,013 | 165,556 | 1.366% | 98.40% | 8.143 | 70.94% | 20.20% |
| Brunel additive, 1 s | 746,718,146 | 746,718,146 | 100% | 36.54% | 2.434 | 1.00% | 0% |
| Brunel Morrison, 1 s | 987,509,929 | 987,509,929 | 100% | 57.50% | 2.473 | 7.00% | 0% |
| Brunel additive, 10 s | 4,603,375,649 | 4,603,375,649 | 100% | 45.45% | 2.442 | 2.60% | 0% |
| Brunel Morrison, 10 s | 7,113,695,805 | 7,113,695,805 | 100% | 66.39% | 2.597 | 10.80% | 0% |

`Native-tick idle` and `Native-tick CV` refer to weight-update attempts. The
MNIST traversal streams themselves have about 33.9% idle 0.5 ms ticks for all
three cases. Thus the input-driven transmission workload is relatively dense
during presentations, but the one-trace weight-write workload is not
continuous: more than 98% of native ticks contain no writes. At 100 ms,
approximately 20% of MNIST bins remain empty because the protocol contains a
150 ms zero-input rest after every 350 ms stimulus.

Brunel is bursty on the native tick and strongly autocorrelated: native-tick
lag-1 correlation ranges from 0.935 to 0.950. Nevertheless every 100 ms window
contains updates. Morrison generates more updates overall but also a larger
fraction of empty 0.1 ms ticks, reflecting larger synchronized bursts separated
by silence rather than a smoother stream.

## MNIST spatial locality

The traversal graph weights each structural input-to-E edge by pre plus post
event count. The update graph uses the same weights for triplet, but only post
count for one-trace.

| Workload | Input Moran's I | Traversal Gini | Traversal top-10% edge mass | Top-10% pixels' traversal mass | Update Gini | Top-10% pixels' update mass |
|---|---:|---:|---:|---:|---:|---:|
| Triplet dense | 0.97797 | 0.65796 | 36.49% | 36.71% | 0.65796 | 36.71% |
| One-trace dense | 0.97816 | 0.65788 | 36.47% | 36.70% | 0.69188 | 10.08% |
| One-trace sparse 0.125 | 0.97808 | 0.65316 | 35.96% | 36.78% | 0.70667 | 10.87% |

The input spike maps clearly preserve image geometry. Input spike count versus
outgoing traversal strength has Spearman rho 1.000 for both dense cases and
0.972 for sparse connectivity. In contrast, input spike count versus outgoing
one-trace update strength is undefined for dense connectivity because every
input row has the same strength, and only 0.087 for sparse connectivity. Sparse
topology adds random degree variation; it does not transfer image locality into
post-driven weight-write locations.

At the excitatory endpoint, firing count versus incident update strength has
rho 1.000 for both dense cases and 0.992 for sparse. This is mostly an algebraic
consequence of the event rule and fixed degree, rather than independent evidence
of a learned assembly.

Net plastic effect is much sparser than attempt count. Across the 100 accepted
samples, 66,973 triplet synapses, 183,126 dense one-trace synapses, and 20,275
sparse one-trace synapses had a nonzero net change in at least one presentation
attempt. Mean accumulated absolute net change over structural edges was 0.00146,
0.000610, and 0.00641 respectively. These are sums of per-attempt absolute net
changes, so opposite updates within one 500 ms attempt may still cancel.

## Brunel spatial locality

The graph scan covers all 81 million plastic E-E edges, including multapses.
Firing communities are k-means clusters of normalized 20 ms-binned firing
trajectories. Weighted directed modularity compares update mass within those
communities against a directed strength-preserving configuration null.

| Rule / window | E spikes | Update attempts | Edge Gini | Top-10% edge mass | Firing/out-strength rho | Top-10% firing-neuron incident mass | Modularity Q |
|---|---:|---:|---:|---:|---:|---:|---:|
| Additive, 1 s | 41,417 | 746,718,146 | 0.1480 | 14.80% | 0.9844 | 25.06% | -0.000336 |
| Morrison, 1 s | 54,856 | 987,509,929 | 0.1328 | 14.26% | 0.9887 | 24.49% | -0.000369 |
| Additive, 10 s | 255,675 | 4,603,375,649 | 0.0584 | 11.85% | 0.9887 | 21.36% | +0.000085 |
| Morrison, 10 s | 395,344 | 7,113,695,805 | 0.0451 | 11.42% | 0.9813 | 20.81% | -0.000173 |

Incoming-strength correlations are even higher: 0.9844 and 0.9888 at 1 s,
then 0.9973 and 0.9981 at 10 s for additive and Morrison. At 10 s, 45.88% of
additive edges and 46.82% of Morrison edges are required to carry half of all
update mass. This is close to uniform and moves farther from a concentrated
edge graph as the observation window grows.

The 10 s whole-window result does not hide a late modular community. Additive Q
moves from -0.000336 in the first second to -0.003723 in the last; Morrison
moves from -0.000369 to -0.001208. Late-window edge-frequency Gini rises to
0.2536 for additive and 0.1798 for Morrison as firing declines and becomes more
heterogeneous, but the mass remains no more community-local than its endpoint
strengths predict.

Update count also only partly predicts learning magnitude. On the deterministic
200,000-edge sample, Spearman rho between update frequency and absolute net
weight change is 0.208 for additive and 0.279 for Morrison over 10 s. Mean
absolute changes are 2.04 pA and 2.26 pA. Spike timing, trace state, update sign,
weight dependence, clipping, and cancellation therefore matter substantially.

## Artifacts

Each successful case directory contains a complete `manifest.json`, structured
`summary.json`, compressed raw arrays in `locality.npz`, and `temporal.png` and
`spatial.png` figures:

- `locality/runs/selected_1s_20260826_a/`: all three MNIST cases and both 1 s
  Brunel cases.
- `locality/runs/brunel_10s_20260826_a/brunel_additive/`: additive 10 s.
- `locality/runs/brunel_morrison_10s_20260827_b/brunel_morrison/`: Morrison
  10 s successful rerun.

The raw arrays retain native-tick traversal/update series, pre and post series,
endpoint firing counts, graph strengths and communities, frequency histograms,
sampled weight trajectories, and sampled inter-update gaps. Generated run
directories are ignored by Git but remain in the workspace for inspection.

This is one MNIST segment and one Brunel seed. The formulas and deterministic
first-window reproductions support the mechanistic conclusions, but population
statistics should be repeated over seeds before interpreting small differences
between additive and Morrison as general effects.
