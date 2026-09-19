# Dense MNIST ActiveN / GPU comparison

[Plot (PNG)](mnist_dense_violin.png) · [Plot (SVG)](mnist_dense_violin.svg) ·
[Results](report.md) · [Summary JSON](summary.json) · [CSV](comparison.csv)

The [33 W power-normalized comparison](power/README.md) adds measured GPU
running power, sampled five times at five-second intervals per GPU/rule.

The four violins show measured ActiveN timestep durations for 1-trace,
2-trace, 3-trace, and all three rules combined. ActiveN runs at an assumed
1 GHz: one cycle is one nanosecond. Each violin includes presentation and rest.
Colored horizontal bars and dots show each GPU's arithmetic mean timestep
duration, averaged over all five recorded run repetitions for the corresponding
rule. The combined GPU marker averages all 15 repetition means. ActiveN
diamonds mark arithmetic means; the vertical center bars span the 25th to
75th percentiles. Duration labels are microseconds, on a linear axis starting
at zero. Each absolute GPU point also shows its mean duration divided by the
ActiveN mean for that group, rounded to one decimal place (e.g. `7.3×`). These
are ratios of means, distinct from the mean sample ratios reported below.

The supplied hardware file has 1,200 samples per rule: 840 presentation and
360 rest, giving 3,600 samples combined. GPU repetitions cover 100 accepted
images, including retries: 102,000 steps for 1-trace and 103,000 for both
2-trace and 3-trace. No GPU timestep distributions or phase-specific GPU
timings are available. GPU bars represent aggregate means and have no
per-timestep spread or confidence interval attached.

## Equal weighting and speedup definitions

Let `H[r,i]` be the hardware duration of sample `i` for rule `r`, and let
`G[r,j]` be GPU repetition `j`'s elapsed workload time divided by its executed
simulation steps. All quantities use the same time unit. First compute
`Gmean[r] = mean_j(G[r,j])`, including all five repetitions without trimming.

The confirmed primary statistic is the arithmetic mean of sample speedups:

```text
S[gpu,r] = mean_i(Gmean[gpu,r] / H[r,i])
S[gpu,combined] = sum_r(n[r] * S[gpu,r]) / sum_r(n[r])
S[all GPUs,combined] = mean_gpu(S[gpu,combined])
```

This equals averaging every `G[r,j] / H[r,i]` within the same rule, then
weighting each hardware sample equally across rules. The current inputs have
equal hardware sample counts and five GPU repetitions everywhere: each of
the 72,000 hardware-sample / GPU-repetition combinations receives equal weight.
These are comparisons with run averages, not paired measurements of the same
individual timestep. Presentation and rest retain their measured 70:30 mixture;
they are not reweighted to 50:50. Combined duplicates none of the underlying
samples in the overall speedup calculation.

The report also includes a distinct ratio of mean durations:

```text
R[gpu,r] = Gmean[gpu,r] / mean_i(H[r,i])
R[gpu,combined] = weighted_mean_r(Gmean[gpu,r], n[r]) / mean_all_samples(H)
R[all GPUs,combined] = mean_gpu(R[gpu,combined])
```

For these data the primary statistic is **6.967685×** across all GPUs; the
ratio of overall mean durations is **5.971012×**. Averaging ratios emphasizes
speedups at shorter hardware timesteps, so the two results differ. The primary
statistic is not an end-to-end total-runtime speedup. No geometric means,
median substitution, outlier removal, or duration-based weighting are used.

## Inputs and measurement scope

- ActiveN: [`../../mnist_dense_results.json`](../../mnist_dense_results.json).
  The user identifies this as densely connected MNIST with 400 excitatory
  neurons and all three STDP rules. The JSON contains no hardware source hash,
  checkpoint identity, seed, sampling schedule, or timing-boundary metadata.
- GPUs: [`../baseline-20260907-matrix.json`](../baseline-20260907-matrix.json),
  only `mnist_1trace_dense`, `mnist_2trace_dense`, and `mnist_3trace_dense`.
  RTX 3090, A100 PCIe 40 GB, A800 SXM4 80 GB, and H800 use the same recorded
  GeNN 5.4.0 FP32 simulation sources and workloads. Full machine configurations,
  exact commands, diagnostics, and source hashes are in that baseline artifact;
  its SHA-256 and all selected timing commands are copied to `manifest.json`.
- GPU protocol: real MNIST training data from `data/mnist`, 784 inputs,
  400 excitatory and 400 inhibitory neurons, 0.5 ms simulation timestep,
  700 presentation plus 300 rest steps per attempt, learning enabled,
  zero configured GeNN delay steps, GeNN seed 20260724. Each case branches from
  a 10,000-image weight/threshold checkpoint and accepts 100 additional images.
  See [`../../BASELINE.md`](../../BASELINE.md) for checkpoint origins and the
  full protocol. All four GPUs have matching recorded population spike totals.
- GPU elapsed workload time includes normalization, transfers, retry handling,
  presentation, rest, and a final completion read. Compilation, allocation,
  dataset loading, and result writing are outside the timer. Per-timestep GeNN
  event timing and profilers were disabled. Hardware uses supplied cycle counts;
  matching phase proportions do not establish identical timing boundaries,
  checkpoints, numerical precision, or spike trajectories across hardware and GPUs.
- This compares the recorded complete GPU/host/software configurations with
  the supplied hardware timing samples. It is not a kernel-only comparison or
  a claim about intrinsic GPU architectural limits. No new benchmark runs or
  accuracy evaluations were performed for this historical timing graph; the
  separate power graph uses fresh runs collected alongside power readings.

## JSON exports

`data/hardware.json` preserves the input rows and their order. Each of
`data/rtx3090.json`, `data/a100.json`, `data/a800.json`, and `data/h800.json`
has 15 rows: rules 1, 2, 3 in order, each with repetitions 1 through 5 in the
original order. They retain the same three field names:

```json
{"class": "1-trace dense", "presenting": null, "cycles": 26473.034411787477}
```

For GPUs, `cycles` stores nanoseconds per simulated timestep, equivalently
cycles at a hypothetical 1 GHz, **not physical GPU clock cycles**.
`presenting: null` extends the hardware boolean field to indicate an aggregate
covering both phases; neither phase is fabricated. One GPU row is one run's
average, not one timed simulation step. `manifest.json` records this distinction.
`summary.json` contains full-precision arithmetic statistics; `comparison.csv`
and `report.md` make them easier to inspect.

## Reproduce

From the repository root, using Python with the pinned plotting dependencies:

```sh
python3 -m pip install -r genn-sweep/hardware-comparison/requirements.txt
python3 genn-sweep/hardware-comparison/compare.py
```

Alternatively, Nix can supply the plotting environment (package versions may
differ from the pinned versions above and are recorded in each output manifest):

```sh
nix-shell -p 'python3.withPackages (p: [ p.numpy p.matplotlib ])' \
  --run 'python3 genn-sweep/hardware-comparison/compare.py'
```

The script validates the input schema, timing denominators, selected GPU
manifests, and recorded cross-GPU population totals. It overwrites only its
named derived artifacts in the output directory. Use another output directory
to retain a separate rendering:

```sh
python3 genn-sweep/hardware-comparison/compare.py \
  --scale log --output copilot/tmp/mnist-dense-comparison-log
```

The default linear axis starts at zero, showing the absolute duration gap;
the ActiveN violins consequently appear compact. KDE is calculated in linear microseconds with Scott's
bandwidth on 512 points between the observed minimum and maximum. Violins have
equal maximum width; their width does not encode sample count. Only the display
axis changes when `--scale log` is requested. No hardware samples are downsampled or discarded.

If future hardware files have unequal sample counts per rule, combined GPU
means and speedups use those counts to preserve equal weight per hardware
sample. The supplied file has equal counts, so all three rules have equal weight.

## Validation of the supplied results

- Independent standard-library calculations recomputed all 72,000 ratios
  directly from GPU wall seconds / simulation steps and hardware cycles.
  Per-rule and combined statistics agree within relative tolerance `1e-14`.
- All 3,600 hardware rows retain their original values and order. All 60 GPU
  exports match the five source repetitions for each GPU/rule. Phase counts,
  the 20-row CSV, all artifact hashes, and all nine baseline simulation source
  hashes were checked. The SVG contains exactly four violin bodies.
- The rendered PNG was visually inspected for readable labels and correct
  group/color mapping.
- Existing baseline, reimplementation, and Brunel unit suites passed both
  before and after this work: **27 passed, 3 skipped**. The skips are the GeNN
  event-order tests, since PyGeNN is not installed in the plotting environment.
  Tests ran with `--import-mode=importlib -p no:cacheprovider`, with `reimpl`
  and `brunel` on `PYTHONPATH`. Simulation source files were not changed.
