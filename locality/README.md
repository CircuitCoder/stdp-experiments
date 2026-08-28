# Synapse update locality

This directory captures temporal and graph locality for the five workloads in
`genn-sweep`. The capture is deliberately separate from the timing sweep: spike
recording, weight snapshots, connectivity reads, graph reductions, and plots
are diagnostic overhead and must not be included in performance measurements.

Two event definitions are reported. A **synapse traversal** executes a pre- or
postsynaptic handler. A **weight-update attempt** executes code assigning the
plastic weight, even when a zero trace or bound leaves the value unchanged.
The MNIST one-trace rule therefore has pre-driven transmission traversals but
only post-driven weight updates. The triplet MNIST and both Brunel rules update
weights on both event paths.

The default command reproduces the selected FP32 protocols: real MNIST, the
three committed 10,000-sample checkpoints, 100 accepted training samples, and
the scale-1 Brunel cases with seed 20260724, 100 ms presimulation, and 1,000 ms
measurement.

```sh
nix-shell \
  -p python313Packages.numpy python313Packages.scipy \
     python313Packages.matplotlib python313Packages.psutil \
  --run 'PYTHONPATH="$PWD/3rdparty/genn:$PYTHONPATH" \
    CUDA_PATH=/tmp/stdp-locality-cuda13-patched \
    CUDAHOSTCXX=/nix/store/5p9jf97rva6v7w56a372as90sgvw82jl-gcc-wrapper-13.4.0/bin/g++ \
    LD_LIBRARY_PATH=/nix/store/22qr7rzm4ba3z8dpai056lcxq6wh6w4a-stdp-cuda-toolkit-genn-12.8-complete/lib:/tmp/stdp-locality-cuda13-patched/lib:/usr/lib/wsl/lib \
    python locality/run.py --output locality/runs/selected_1s'
```

On this container, `/tmp/stdp-locality-cuda13-patched` is a writable copy of
the retained CUDA 13 toolchain with the `rsqrt` and `rsqrtf` declarations in
`include/crt/math_functions.h` disabled for glibc 2.42. CUDA 12.8's runtime is
also kept on `LD_LIBRARY_PATH` because the retained PyGeNN extension was built
against `libcudart.so.12`.

Each case directory contains `manifest.json`, `summary.json`, compressed raw
arrays in `locality.npz`, and `temporal.png`/`spatial.png`. Build products stay
inside that case directory. Output directories must be new.

An extended Brunel capture uses the same model but measures 10 seconds. It has
a 100 Hz excitatory-rate guard checked every 100 ms.

```sh
python locality/run.py \
  --cases brunel_additive brunel_morrison \
  --brunel-sim-ms 10000 \
  --output locality/runs/brunel_10s
```

The completed measurements and their interpretation are in `RESULTS.md`.
Generated run directories are excluded from Git by the repository `.gitignore`;
their manifests, raw arrays, and plots remain available in the workspace.

Run the deterministic analysis tests with:

```sh
pytest -q locality/tests
```
