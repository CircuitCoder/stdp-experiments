# Zero-delay three-trace reimplementations

This directory contains independent ports of the validated Brian 1
`zero_delay_midpoint_v1` network and selected derived one-trace networks to
current simulators. The common contract is implemented in `zd3/`;
simulator-specific code lives below `backends/`.

The shared network model is fixed to:

- real MNIST in `data/mnist`, in sequential training-set order;
- 784 Poisson inputs, 400 excitatory neurons, and 400 inhibitory neurons;
- a 0.5 ms timestep, 350 ms stimulus, and 150 ms rest per attempt;
- exact exponential conductance decay and the frozen midpoint-conductance
  voltage update documented in `../network.md`;
- zero-delay reference semantics, normalization of each feedforward column
  to 78 before every attempt, and retry at intensity increments of one until at
  least five excitatory spikes occur;
- the Brian 1 spike-counter window, which is advanced after stimulus rather
  than after rest, so the next attempt includes spikes from the preceding rest;
- a selected feedforward rule and topology from the variant table below;
- lateral inhibition 25.5 during training and 17 during inference; and
- an adaptive excitatory threshold initialized to 20 mV, incremented by
  0.05 mV per spike, with a `1e7 ms` decay constant.

The runners accept these `--variant` values:

| Variant | Feedforward rule | Topology | `wmax` | Potentiation rate |
|---|---|---|---:|---:|
| `triplet-dense` | Reference three-trace | Dense | 1 | 0.01 |
| `one-trace-dense` | Power one-trace | Dense | 1 | 0.0005 |
| `one-trace-bernoulli-0125` | Power one-trace | Seeded Bernoulli 12.5% | 8 | 0.0026390158 |

The sparse mask is generated once with NumPy `RandomState(20260723)` and has
39,001 structural input-to-excitatory synapses. Absent pairs remain absent;
normalization and learning operate only on present pairs. This reproduces the
Brian 1 sparse experiment definition in `../sparse-networks.md`. After every
normalization, the sparse runners also enforce the reference safety bound
`max(weight) <= 1.02 * wmax`; crossing it aborts before the next simulation
attempt because the fractional-power update is not defined above `wmax`.

## Checkpoints

Portable checkpoints are NumPy `.npz` files containing float64 feedforward
weights in row-major `[input, excitatory]` order, float64 excitatory theta, the
accepted sample count, and a canonical JSON manifest. They intentionally have
the same scope as the Brian 1 checkpoints: they support fresh plasticity-off
inference and branched training, but are not exact resumable runtime snapshots.

Each backend may additionally write a native runtime checkpoint. Native files
must not be presented as portable checkpoints.

`convert_brian1_checkpoint.py` converts an immutable Brian 1 weight/theta pair
to this format. Brian 1 theta is converted from volts to millivolts. The input
files and their SHA-256 hashes are retained in the portable manifest.

## Runners

The CPU implementations use the same CLI shape:

```text
python reimpl/run_brian2.py train --output RUN --samples 1000
python reimpl/run_genn.py train --backend single_threaded_cpu --output RUN --samples 1000
python reimpl/run_nest.py train --output RUN --samples 100 --threads 8

python reimpl/run_brian2.py evaluate --output EVAL --checkpoint CHECKPOINT
python reimpl/run_genn.py evaluate --output EVAL --checkpoint CHECKPOINT
python reimpl/run_nest.py evaluate --output EVAL --checkpoint CHECKPOINT --threads 8
```

All run and evaluation directories are created with `exist_ok=False`. Reusing a
tag fails instead of overwriting an older artifact.

The GeNN source is backend-neutral. `--backend cuda` selects its tested CUDA
backend, while `--backend single_threaded_cpu` selects the CPU backend. GeNN
also accepts `--parallelism postsynaptic|presynaptic` and
`--num-threads-per-spike N`; the latter is passed to the feedforward synapse
group when presynaptic parallelism is selected.

`run_brian2cuda_codegen.py` instantiates the same Brian2 model and emits CUDA
standalone source for a stimulus/rest attempt. Pass `--variant` to select the
three-trace, dense one-trace, or sparse one-trace graph, and pass `--run` to
compile and execute that attempt on CUDA. It does not implement the
presentation-level Python controller inside CUDA standalone, so it is a
runtime validation rather than an exact training runner: normalization, retry
decisions, periodic reporting, and checkpoint actions require host interaction
between attempts while preserving all hidden synapse, neuron, delay, and RNG
state.

The NEST implementation is an external C++ extension under `nest_module/`.
Build it against the same NEST installation used by PyNEST:

```text
bash reimpl/nest_module/build_module.sh /path/to/nest-config BUILD_DIRECTORY
```

Put `BUILD_DIRECTORY` on `LD_LIBRARY_PATH` before invoking `run_nest.py`.
With the source-built PyNEST on `PYTHONPATH`, validate the module with:

```text
python reimpl/tests/nest_module_probe.py
```

The probe checks that NEST's one-resolution transport changes target
conductance on the immediately following integration cycle, checks an
independently calculated midpoint state step, verifies that postsynaptic spikes
preceding the first real pre-event cannot cause triplet potentiation, and
forces a pre/post/post/pre sequence whose slow trace must produce a positive
weight update.

## Measurement contract

`simulation_wall_seconds` excludes import, compilation, dataset loading,
checkpoint serialization, and evaluation. It includes normalization, simulator
calls, retry decisions, and the stimulus/rest dynamics. The primary cycle
metric is:

```text
seconds_per_timestep_cycle = simulation_wall_seconds / simulated_ticks
```

where `simulated_ticks` includes every stimulus and rest tick for accepted and
retried attempts. Build time and evaluation time are reported separately.

Checkpoint accuracy uses a fresh plasticity-off inference pass over the first
1,000 real-MNIST test images. The same activity assigns neurons and scores the
predictions. This is the optimistic simple-demo probe used by the reference,
not the paper protocol.

## Backend limitations

NEST transports spikes with a minimum delay of one simulator resolution. Its
port assigns every feedforward and recurrent synapse the fixed latency 0.5 ms,
asserts that value after construction, and records it explicitly in every
manifest. The event reaches the target's immediately following integration
cycle. CUDA runtime validation and measurements are recorded in `RESULTS.md`.

`run_nestgpu_mnist_validation.py` provides checkpoint-only inference for the
dense and 12.5% sparse one-trace variants. It reconstructs the trained topology,
weights, theta, retry protocol, and one-step latency, but uses NEST-GPU's
`aeif_cond_beta` dynamics as a conductance approximation. Its results are
validation controls, not an exact simulator port or a trainable one-trace
implementation.

The vendored NEST-GPU revision routes both forward and reverse spike updates
through `SynapseUpdate(syn_group, weight*, delta_t)` in `src/get_spike.h` and
`src/rev_spike.h`. Its `conn12b` and `conn16b` records have only target metadata
and one float weight. The callback therefore has neither a connection index for
auxiliary trace storage nor an accumulated source trace for the one-trace power
rule; it also lacks the target's previous-post time needed by `post2before` in
the three-trace rule. Exact training requires a core connection/state ABI
change at this revision and cannot be supplied as a registered custom synapse
model.
