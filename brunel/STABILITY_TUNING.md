# Scale-1 Morrison stability tuning

Run date: 2026-07-24 to 2026-07-25 UTC

## Question and conclusion

These experiments tested whether the scale-1 NEST HPC workload can avoid the
Morrison-rule runaway by changing recurrent inhibition, external drive, and
the LTD/LTP balance.

Yes, immediate runaway can be avoided at scale 1. The tested candidate uses
`g = 8`, `eta = 3.2`, and `alpha = 0.1026` (twice the benchmark depression
coefficient). It completed a 10 s simulation at 5.859 Hz, with a maximum
100 ms block rate of 8.58 Hz. The E-to-E sample remained unimodal and entirely
interior. The same parameters completed 2 s under two additional seeds at
5.566 and 6.063 Hz.

This does **not** recover Morrison et al.'s validated equilibrium. The 10 s
population-count Fano factor was 50.39, still far above the reported large
network value near 8.5, and the sampled weight standard deviation continued
to grow from zero to 1.260 pA. Morrison et al. report approximately
`45.65 +/- 3.99 pA` after much longer equilibration. The current result is a
stable low-rate scale-1 pilot, not proof of a stationary asynchronous-irregular
state.

## Meaning of scale

In NEST's benchmark, `scale` multiplies the population size:

```text
NE = 9000 * scale
NI = 2250 * scale
```

It is not the average fan-out. The excitatory and inhibitory indegrees remain
9,000 and 2,250 as `scale` changes, so the connection probability and shared
input correlations fall as the network grows. NEST says the original network
corresponds to `scale >= 100`, warns that the desktop `scale = 1` limit can
synchronize, and recommends `scale > 10` with a final rate below 10 Hz for
meaningful benchmark use. Thus `> 10` is the documented lower validation
guideline, not an experimentally sharp minimum.

The local runner exposes population and indegree scaling independently as
`--network-scale` and `--indegree-scale`. All runs below used both at 1, giving
9,000 E neurons, 2,250 I neurons, and 81 million plastic E-to-E synapses.

## Protocol

The runner was extended with a non-plastic E-to-E control and independent
parameters for `g`, `eta`, and Morrison `alpha`. A 100 Hz rate cutoff stops a
run after a 100 ms reporting block while preserving its actual duration,
steps, timing, weights, and termination reason. Static controls and plastic
pilots used NEST `3.9.0-post0.dev14`, eight threads, 0.1 ms resolution, 1.5 ms
delay, a 100 ms presimulation, and seed `20260724` unless stated otherwise.
The runner SHA-256 is
`7fe2589d02addcf1cfd8cb6689351f2e68a9b3c2102b8e1884b2d58d49872305`.

The host concurrently ran two single-threaded Brian 1 training jobs. The two
additional-seed runs were also concurrent with each other. Rates and dynamics
remain comparable, but their wall timings are load-dependent.

## Static controls

Static E-to-E weights isolate the network dynamics before plastic feedback.

| `g` | `eta` | Mean rate | Max 100 ms rate | Fano, 3 ms | Mean ISI CV |
|---:|---:|---:|---:|---:|---:|
| 5.0 | 1.685 | 9.927 Hz | 16.93 Hz | 90.92 | 0.706 |
| 5.5 | 1.850 | 8.115 Hz | 9.90 Hz | 80.28 | 0.691 |
| 6.0 | 1.685 | 5.785 Hz | 8.07 Hz | 99.80 | 0.602 |
| 6.0 | 1.850 | 5.670 Hz | 6.86 Hz | 51.24 | 0.651 |
| 6.0 | 2.100 | 7.038 Hz | 9.14 Hz | 63.18 | 0.660 |
| 8.0 | 3.200 | 6.014 Hz | 6.78 Hz | 50.43 | 0.671 |
| 8.0 | 4.000 | 10.319 Hz | 17.69 Hz | 187.13 | 0.744 |

Increasing inhibition and compensating with external drive can set a low mean
rate, but it did not remove the excessive population correlations at scale 1.
The best tested static Fano factors remained around 50 rather than 8.5.

## Plastic sweep

The stock Morrison value is `alpha = 0.0513`. `Completed` means the requested
duration was reached without triggering the 100 Hz block-rate cutoff. The
first `alpha = 0.053865` run predates the cutoff but completed in a clearly
runaway state.

| `g` | `eta` | `alpha` | Requested / actual | Completed | Mean rate | Max block rate | Final weight mean +/- std | Fano |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 5 | 1.685 | 0.053865 | 1.0 / 1.0 s | yes | 267.974 Hz | 936.47 Hz | 88.095 +/- 83.796 pA | 2997.83 |
| 5 | 1.685 | 0.056430 | 1.0 / 0.4 s | no | 112.628 Hz | 377.79 Hz | 79.162 +/- 8.033 pA | 3225.43 |
| 5 | 1.685 | 0.061560 | 1.0 / 0.6 s | no | 134.615 Hz | 722.07 Hz | 56.285 +/- 43.340 pA | 3280.31 |
| 8 | 3.200 | 0.061560 | 10.0 / 2.2 s | no | 46.819 Hz | 688.55 Hz | 62.600 +/- 38.322 pA | 2655.10 |
| 8 | 3.200 | 0.076950 | 2.0 / 1.5 s | no | 23.126 Hz | 120.08 Hz | 64.743 +/- 1.615 pA | 1639.11 |
| 8 | 3.200 | 0.102600 | 10.0 / 10.0 s | yes | 5.859 Hz | 8.58 Hz | 45.519 +/- 1.260 pA | 50.39 |

The `g = 8`, `eta = 3.2`, `alpha = 0.06156` case looked healthy at 1 s
(5.949 Hz) but ran away after 2 s. This is direct evidence that the original
250 ms Morrison failure was not simply too short to settle, and also that a
quiet 1 s pilot is insufficient to establish stability.

For this seed and drive/inhibition pair, 1.5 times the stock `alpha` failed,
while twice the stock value survived 10 s. The sweep brackets a practical
short-run threshold; it does not determine a universal critical value.

## Seed check and performance

| Seed | Duration | Mean rate | Max block rate | Fano | Weight mean +/- std | Simulation wall/step |
|---:|---:|---:|---:|---:|---:|---:|
| 20260724 | 2 s | 5.938 Hz | 8.10 Hz | 56.92 | 45.562 +/- 0.585 pA | 0.598 ms |
| 20260725 | 2 s | 5.566 Hz | 6.72 Hz | 38.89 | 45.502 +/- 0.553 pA | 0.757 ms |
| 20260726 | 2 s | 6.063 Hz | 9.91 Hz | 64.33 | 45.587 +/- 0.597 pA | 0.818 ms |
| 20260724 | 10 s | 5.859 Hz | 8.58 Hz | 50.39 | 45.519 +/- 1.260 pA | 0.622 ms |

All four samples remained a single interior histogram mode with zero sampled
boundary mass. The 10 s measured NEST calls took 62.183 s for 100,000 steps;
including the 100 ms presimulation, all `Run` calls took 62.563 s for 101,000
steps. Construction took 23.556 s and total process wall time was 121.265 s.

## Interpretation and next test

The tested ratios can suppress the positive-feedback failure at scale 1, but
they define a different operating point: inhibition is 60% stronger, external
drive is about 90% stronger, and depression is doubled. Stronger inhibition
alone is not enough, because the higher external drive needed to restore the
rate can also increase correlations. Depression must be retuned against the
resulting spike statistics.

A defensible equilibrium claim would require at least a 200 s run, several
seeds, a plateau in weight mean and variance, and stable blockwise rate/Fano
statistics. A scale sweep would answer the separate minimum-scale question,
but the documented `scale > 10` lower guideline already exceeds this
container's practical memory budget: recurrent connection count is about
1.392 billion at scale 11 with fixed indegree.

The 10 s candidate command was:

```sh
bash brunel/run_with_nest_env.sh --rule morrison \
  --output brunel/runs/stability_morrison_g8_eta3200_alpha102600_10s_seed20260724_a \
  --threads 8 --seed 20260724 --network-scale 1 --indegree-scale 1 \
  --inhibitory-weight-ratio 8 --external-drive-eta 3.2 \
  --morrison-alpha 0.1026 --abort-rate-hz 100 \
  --record-neurons 1000 --weight-sample-size 100000 \
  --presim-ms 100 --sim-ms 10000 --chunk-ms 100
```

All artifacts are under `brunel/runs/stability_*`; matching stdout logs are
under `brunel/logs/`. Each run contains the complete manifest, periodic and
final statistics, timings, and deterministic initial/final weight samples.
