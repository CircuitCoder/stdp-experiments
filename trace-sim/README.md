# STDP trace memory simulator

This directory reconstructs feedforward STDP weight accesses from the validated
`zero_delay_midpoint_v1` firing traces. A weight is always 4 bytes. Capacity is
payload capacity; tag, control, recency, and deferred-operator metadata are
reported separately. The matrix covers 128/256/512 KiB and 1/2/4 MiB payload
capacities.

The simulator compares direct memory, whole-line and fine-grained set-associative
caches, and three bounded Robin Hood hash-table layouts. It evaluates eager and
deferred sparse allocation, ordinary write-back depression and drain-on-pre
depression, and 32/64/128-byte lines.

Build and test:

```sh
nix develop path:./trace-sim -c cmake -S trace-sim -B trace-sim/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
nix develop path:./trace-sim -c cmake --build trace-sim/build
nix develop path:./trace-sim -c ctest --test-dir trace-sim/build --output-on-failure
```

Run one configuration:

```sh
trace-sim/build/trace-sim run \
  --trace ref/zero_delay_midpoint_v1/traces/checkpoint_010000/runtime/results/firing-trace.jsonl \
  --structure set-line --line-size 64 --capacity 1MiB --ways 4 \
  --allocation deferred --depression write-back --output result.csv
```

Run the complete matrix on all three traces and regenerate the summary tables:

```sh
nix develop path:./trace-sim -c python3 trace-sim/scripts/run_matrix.py
nix develop path:./trace-sim -c python3 trace-sim/scripts/summarize.py
nix develop path:./trace-sim -c python3 trace-sim/scripts/validate_results.py
nix develop path:./trace-sim -c python3 trace-sim/scripts/run_order_sensitivity.py
```

`refresh_direct_baselines.py` is a narrow maintenance helper for regenerating
only the direct-memory rows after changes confined to that baseline. Normal
reproduction should use the complete matrix command above.

See [report.md](report.md) for the model contract, research notes, and results.
