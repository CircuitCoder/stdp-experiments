# GPU absolute performance and performance adjusted to 33 W

[Results](report.md) · [PNG](mnist_dense_33w.png) · [SVG](mnist_dense_33w.svg) ·
[Raw readings](power_samples.csv) · [Comparison CSV](comparison.csv)

ActiveN's assumed power and clock are **33 W at 1 GHz**. GPU power is measured running
board power from `nvidia-smi`, with no TDP substitution or idle subtraction.
The measurement includes GPU memory/board overhead but excludes host CPU power.

The graph overlays **solid GPU bars for absolute timestep duration** and
**dashed GPU bars for duration adjusted to 33 W**, using the same GPU colors
and a shared linear time axis starting at zero. Each pair uses timing and power
collected together in the selected measurement run. Within each group, the GPU
columns follow the legend order; each GPU's solid and dashed bars align.
RTX 3090's adjusted markers sit in a labeled strip above the axis, with their
actual microsecond values retained. They are excluded from the axis range;
their vertical placement in the strip does not encode duration. Each absolute
GPU point also shows its mean duration divided by the ActiveN mean for that
group, rounded to one decimal place (e.g. `7.3×`). This ratio of mean durations
is distinct from the mean sample energy advantage reported below.
The four ActiveN violins include vertical center bars spanning the 25th to
75th percentiles and diamonds at their arithmetic means. The linear scale
makes the ActiveN distributions compact. The original standalone timing graph
continues to use the historical five repetitions per GPU/rule.

## Protocol

Each GPU runs each of the three dense MNIST STDP cases once. The simulator
loads the same immutable 10,000-image checkpoints and retains the recorded
FP32 GeNN model, seed 20260724, sequential real-MNIST order, 400 excitatory
neurons, 700 presentation / 300 rest steps per attempt, normalization and retry
behavior. All first-100-image attempt counts and input/E/I spike totals must
match the saved baseline before measurement begins.

After at least 10 seconds of training warmup, a background sampler requests
**exactly five power readings at 5, 10, 15, 20 and 25 seconds**. The training
loop continues while `nvidia-smi` runs. Query start/end timestamps, sensor
timestamps, GPU UUID, watts, utilization, SM clock and temperature are retained.
Power queries are never issued per timestep. Their actual scheduling jitter
and the five-second intervals are checked by the report script.

After the fifth query, the current presentation/rest attempt finishes and a
completion read synchronizes the final rest before timing stops. The duration
is divided by all executed steps in that window, including retry attempts.
Build, allocation, loading, warmup and final diagnostics are outside the timer.
Per-timestep GeNN event timing and spike recording are disabled. **No periodic
or final checkpoint is written.** Inputs are only loaded; outputs are telemetry,
timing, diagnostics, manifests, compiler output and logs.

These fresh timings come from a longer continuation after warmup. Their image
ranges and endpoints differ from the historical 100-image timing runs and can
differ across GPUs. They retain the same workload mechanics, but are not a new
strictly paired image-count timing baseline. A separate table applies the new
power estimates to the old timings for continuity with the original comparison.

## Computation

For GPU `g`, rule `r`, hardware timestep `i`, and power reading `j`:

```text
T[g,r] = measured window seconds / executed simulation steps
P[g,r] = arithmetic mean of its five measured watts
E[g,r] = P[g,r] × T[g,r]
T33[g,r] = E[g,r] / 33 W
Sample advantage[g,r] = mean_i(T33[g,r] / hardware_duration[r,i])
```

Every power reading and hardware timestep receives equal weight within its
rule. The combined result weights rules by their hardware sample counts,
which are equal here. GPUs receive equal weight. Presentation and rest keep
their measured 70:30 mix. Combined mean energy is the mean of each rule's
`P × T`; multiplying the pooled mean power by pooled mean duration would lose
the association between power and workload.

The primary sample advantage uses the same arithmetic mean of individual
ratios as the original comparison. The ratio of overall mean energies is also
reported as a separate statistic. ActiveN mean energy is 33 W times mean
duration: **113.813 µJ per step** across all three rules.

This is a performance-per-watt normalization and coarse energy estimate.
It does not model actual GPU performance under a physical 33 W power cap.
Five sensor readings do not provide a continuous energy integral. The supplied
33 W ActiveN design figure also has not been independently measured here.
NVIDIA's `power.draw` semantics vary by architecture/driver; see its
[power-reading documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html#gpu-power-readings).

## Artifacts and reproducibility

- `experiment.json` records the plan and initial launches before training.
  `service_commands.json` records the four named local services; remote
  commands use the existing user Slurm allocations for A800/H800. The services
  finish their own commands without terminating those allocations.
- `manifest.json` preserves each selected measurement manifest, exact commands,
  compiler/dependency versions, GPU identity, source/input hashes, and output
  hashes. `reference.json` records the expected simulation hashes and first
  100-image counters. Logs and build products remain under `copilot/tmp/`.
- `data/*_measurements.json` preserves the selected measurements and diagnostics.
  `data/*_33w.json` retains the original three fields: `class`, `presenting`,
  `cycles`. Here `cycles` means **equivalent nanoseconds at 33 W**, and
  `presenting: null` means both phases. These rows are derived from five power
  readings, not five GPU timing repetitions or individual GPU timesteps.
- The initial H800 attempt reached the five-reading stage, then failed an
  extra final `1.02 × wmax` guard in the measurement harness. The dense model's
  existing normalization validator explicitly has no such tolerance: column
  normalization is not clipped to the STDP event cap. The corrected harness
  uses that existing validator and records weight extrema. The model and
  training loop were unchanged. The failed attempt's five readings and logs
  are preserved and excluded; only the rerun's five readings per rule enter
  the comparison. See `h800_rerun.json` and `h800_service_command.json`.

The completed H800 rerun's final one-trace maximum weight was 1.023003, with
finite neuron state and column sums 77.874–78.000. This is consistent with
the small normalization overshoot that the dense model permits. The other
H800 rules finished at maximum weights 1.000000 and 0.782716.

Validation independently recomputed all 72,000 sample-ratio comparisons from
the 60 selected power readings and source timings. Observed query intervals
were 4.990–5.009 seconds. All 12 first-100-image baseline checks passed, all
recorded simulation source hashes matched, and no checkpoint/array output
files were created. The existing unit suite passed all 30 tests. The plot was
visually inspected and its SVG contains exactly four violins. See
[`validation.json`](validation.json).

To regenerate the report from the selected local run artifacts, from the
repository root with the parent directory's plotting dependencies installed:

```sh
python3 genn-sweep/hardware-comparison/power_report.py \
  --run H800 copilot/tmp/power_dense_20260910_v2_h800
```

Both plotting scripts default to `--scale linear`; `--scale log` remains
available for an alternate rendering without changing the data or statistics.

To make a fresh measurement, use `measure_power.py` in the recorded GPU/CUDA
environment with a **new output directory**. It refuses to reuse a run directory
or a GPU with another compute process. The five-sample count and five-second
interval are fixed in the script. GeNN input checkpoints are read-only inputs;
there is no checkpoint-save call in the runner.
