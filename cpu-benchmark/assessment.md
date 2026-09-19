# CPU simulator and energy assessment

Use optimized single-threaded GeNN CPU as the current complete-suite CPU latency baseline. It has now
executed all 23 GeNN-CUDA workload definitions using the same model source,
checkpoints, masks and evaluation boundaries. This establishes a working
baseline, not a universal fastest-CPU-simulator claim. The custom NEST ports now
also cover the 21 MNIST and two Brunel definitions, with deterministic mechanics
tests and repeated latency measurements. Brian2 and the Rust implementation
still require work to cover the current contract.

| Option | Coverage in this repository | Concurrency | Assessment |
|---|---|---|---|
| GeNN 5.4 CPU | All 21 MNIST and both current Brunel cases tested | One simulation thread per network | Complete-suite latency baseline, one pinned network at a time with optimized native code. |
| NEST | Separate custom ports cover all 21 MNIST and both current Brunel cases | Native OpenMP within one network; the custom MNIST extension requires one MPI process | MNIST uses 16 threads after a 1/2/4/8/16/32-thread pilot. Brunel has 16- and 32-thread measurements. Native RNG and accumulation order differ from GeNN; short-run mechanics validation does not establish long-training accuracy alignment. |
| Brian2 C++ standalone | Existing implementations need a current-contract audit and port; this full matrix was not timed | Optional OpenMP within one network | Candidate for a later optimized CPU comparison. MNIST normalization and activity-dependent retry control must be supported inside the compiled execution path. |
| Repository Rust application | A different MNIST training implementation; no equivalent 23-case runner | Requires implementation work for this benchmark | Historical accuracy or throughput is not evidence for this workload matrix. |

GeNN's installed source lists `single_threaded_cpu`, CUDA and HIP backends in
[`genn_model.py`](../3rdparty/genn/pygenn/genn_model.py). Its CPU backend does
not provide OpenMP simulation threading. The first experiment used 32
independent replicas, which answered a throughput question instead of the
requested latency question. Those aggregate values must not be used as CPU
timestep latency. The corrected runner rejects more than one network and
reports its own elapsed time per executed timestep.

The upstream ISPC branch and open PR #710 were inspected on September 10,
2026. Both emit SIMD `foreach` loops; neither has task launches, OpenMP regions
or a CPU thread pool for multicore execution of this suite. SIMD support is
not evidence of multicore execution. No usable multicore GeNN backend was
enabled, so the requested single-core fallback was used. Implementing a new
backend would require thread ownership for RNG, spike lists and synaptic
current accumulation, and barriers preserving STDP/neuron event ordering.
Those synchronization costs must be included in any future latency result.
[ISPC development report](https://genn-team.github.io/posts/developing-an-ispc-backend-for-genn-bridging-gpu-and-cpu-performance-for-neural-network-simulations.html),
[ISPC pull request](https://github.com/genn-team/genn/pull/710).

NEST requires `local_num_threads` to be set before creating the network;
`OMP_NUM_THREADS` does not select its simulation thread count. Its default is
one thread. Our runners explicitly select the simulation thread count. The
earlier Brunel runs use passive OpenMP waiting. The MNIST matrix uses active
waiting and spreads 16 threads over the 16 reported physical cores on this
contended host.
[NEST parallel computing documentation](https://nest-simulator.readthedocs.io/en/main/hpc/parallel_computing.html).

The existing repository MNIST NEST CLI already exposes `--threads`, defaulting
to 1; the historical Brunel CLI defaults to 8. Thus NEST did not need a new
parallel execution engine. The custom ports implement and check the current
Brunel and MNIST dynamics and plasticity contracts. The MNIST port reuses all
21 frozen checkpoints and masks, covering one-, two- and three-trace learning
with dense, Bernoulli and fixed-fan-out connectivity. Target-owned plastic
weights avoid delayed NEST history updates crossing a normalization boundary;
native NEST spike transport and per-tick synchronization remain in the timed
path. One input spike is broadcast to its fan-out, with shared input traces.

The [NEST MNIST report](results-nest-mnist-20260910/report.md) records the full
100-image, five-repetition matrix and its thread pilot. Sixteen-thread NEST
took 39.498–99.334 µs per timestep, or 3.16–5.80 times the single-core GeNN
elapsed time across the 21 cases. NEST excitatory spikes per timestep ranged
from 0.955 to 1.072 times the corresponding GeNN rate. All 105 timing runs
exactly matched their NEST diagnostic final weights, thresholds and event
counts. No normalization cap overshoot occurred in these diagnostics; cap
overshoots are nevertheless recorded without aborting, as requested.
These results were collected with other host processes running. The older
`reimpl/run_nest.py` training measurements use another implementation and
protocol; they are not interchangeable with these timings.

Brian2 supports `prefs.devices.cpp_standalone.openmp_threads`, but standalone
execution restricts Python control and access to live state. Enabling this
setting alone does not port the current MNIST host-controlled retry loop.
[Brian2 computation documentation](https://brian2.readthedocs.io/en/stable/user/computation.html).
The existing CPU runners select Brian2's Cython or NumPy runtime target, so
using standalone OpenMP also requires changing their execution path.

The 7950X exposes 16 physical cores and 32 logical CPUs to this guest. Using
both CCDs is appropriate when one simulation can use them. CCD
communication, memory bandwidth, synchronization and SMT can still affect
scaling; the CPU model alone does not establish that 32 threads are optimal.
The measurements retain the tested NEST thread counts rather than assuming
scaling. Eight threads were modestly faster than sixteen in the short MNIST
pilot, and thirty-two were slower. Sixteen were selected for the full matrix
to exercise all physical cores as requested. This is not a claim that maximum
hardware concurrency minimizes latency or energy.

## Quick x86 package-energy measurement

On native Linux with supported hardware and access to the counters, use RAPL
package energy. Linux exposes `energy_uj` and `max_energy_range_uj` through
powercap. Despite its name, the `intel_rapl_msr` driver also handles AMD RAPL
registers. This support exists in upstream Linux 6.6; WSL exposing the same
kernel version does not establish access to the physical host registers.
[Linux powercap documentation](https://www.kernel.org/doc/html/latest/power/powercap/powercap.html),
[Linux 6.6 AMD RAPL driver handling](https://github.com/torvalds/linux/blob/v6.6/drivers/powercap/intel_rapl_msr.c).

A quick native-Linux check is:

```sh
sudo modprobe intel_rapl_msr
sudo perf stat -a -e power/energy-pkg/ -- sleep 5
```

The perf event, when exposed, reports joules over the interval; divide by wall
seconds for average watts. It is a system-wide counter, so running perf around
a command does not attribute that energy exclusively to that process.
[Linux RAPL perf implementation](https://github.com/torvalds/linux/blob/v6.6/arch/x86/events/rapl.c).

For a future power measurement, sample **each package once** around one
network's timed interval. With executed steps `S`, package energy `E` joules,
and wall time `T` seconds:

- Average package power: `E / T` watts.
- Energy cost: `E / S` joules per simulation step.
- Energy efficiency: `S / E` steps per joule, reported alongside latency.

Use the total package counter directly; do not add core or uncore counters to
it. It already represents the package boundary. Package overhead is charged
once to the network even when its computation spans multiple cores. Keep results per workload because MNIST
and Brunel steps represent different amounts of biological time and work.

This container currently exposes no package powercap counter, no RAPL power
PMU, and no accessible physical-CPU MSR device. Thus the completed timing runs
have **null energy and power**, and no performance-per-watt ranking is
available. Native Linux with accessible counters, or a synchronized sensor
log from the Windows host, is needed for the energy measurement. An external
meter would instead measure the whole system and must be labeled accordingly.

Unrelated host processes contribute to both contention and package energy.
Also, the recorded CUDA baseline runs one network per GPU and GPU board power
includes device memory but excludes host CPU power. A future
CPU-versus-GPU energy comparison should use one network on each device
and explicitly chosen hardware boundaries.

See the [corrected latency measurements](results-latency-20260910/report.md) and
[runner instructions](README.md) for commands, validation and raw artifacts.
