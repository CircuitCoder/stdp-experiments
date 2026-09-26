# Proposed GeNN STDP timestamp correction

This records the original FP64 proposal, which was not implemented. The user
instead requested an FP32 tie tolerance of `0.1 * dt` and a hard error for gaps
strictly between `0.1 * dt` and `0.9 * dt`. That change is implemented and tested;
see [MORRISON-TICK-TOLERANCE-20260920.md](MORRISON-TICK-TOLERANCE-20260920.md).
No simulator source was changed during the earlier settling/homeostasis
investigation described below.

## Confirmed defect

`brunel/ports/genn_port.py` selects `nest_causal_boundary` to process coincident
events using the old pre/post traces. A single coincident pair with initially
empty traces must therefore leave its synaptic weight unchanged.

GeNN computes global time as `timestep * dt`, while a delayed spike timestamp
is reconstructed by adding the delivery interval to its stored emission time.
Those expressions can round differently in FP32. The comparison tolerance
at the time of that investigation was only `1e-7 ms`. At an event time of 5.2 ms, the
rounding difference can exceed that tolerance. The pre handler misses the
coincident post event, then the post handler includes the newly added pre
trace. A 45.61 pA synapse incorrectly gains approximately 0.460918 pA.

Tests extracted the actual plastic model definition from the live source and
ran it with two scheduled neurons and one synapse. In both the CPU and CUDA
generated kernels, five of twelve selected starting offsets fail in FP32,
for each of the zero- and 1.5-ms delay cases. All twelve offsets pass in FP64
for each delay and backend. The offsets cover early times and times near
1, 10, 100, 200, and 500 seconds; these are diagnostic selections, not an
estimate of the fraction of events affected in a full network.

Valid reproduction artifacts:

- `copilot/tmp/workload_settling_homeostasis_20260920/morrison_tie_probe_v2/`
- `copilot/tmp/workload_settling_homeostasis_20260920/morrison_tie_probe_cuda_v2/`

Each contains its exact inline command, generated kernel code, and result JSON.
The earlier `morrison_tie_probe/` failed on a diagnostic array-shape assumption.
The earlier CUDA probe omitted the initial device-to-host weight copy; use the
`cuda_v2` result for initial/final weight comparisons. Those artifacts remain
preserved.

## Original implementation plan (not implemented)

1. Set `time_precision="double"` on the shared `GeNNBrunel` model.
2. Change `lastTraceTime`, `lastPostUpdateTime`, and `lastPreUpdateTime` from
   `scalar` to GeNN's `timepoint` type. Weights, traces, neuron state, and their
   arithmetic retain the selected scalar precision, including FP32 workloads.
3. Record time precision explicitly in manifests. Preserve the existing
   plasticity equations, event-order convention, network parameters, and
   configured delays.
4. Add deterministic regressions for coincident and adjacent-tick events at
   early and late absolute times, with both delays, on CPU and CUDA. Check the
   shared additive rule as well as Morrison's rule. Run the existing relevant
   unit suite before and after the source change.
5. Rerun the full-size zero-delay and delayed settling controls from fresh,
   matched initial configurations on 80 GB GPUs, using the same declared
   settling criteria. Save final counts and a final weight sample even when a
   stopping guard fires.

Three additional four-byte timestamp halves cost 9.72 GB for 810 million valid
plastic synapses, plus sparse-storage padding. This fits the available 80 GB
devices given the approximately 28 GiB prior FP32 allocation, but allocation
and runtime must be measured again. This corrects shared Brunel/Morrison code,
so previous performance timings cannot be assumed unchanged.

The defect is established independently of network dynamics. Its contribution
to the delayed network's instability, and any remaining paper mismatch after
correction, require the controlled reruns. The current measurements alone do
not establish causality.
