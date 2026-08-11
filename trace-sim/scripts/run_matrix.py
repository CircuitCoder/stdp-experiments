#!/usr/bin/env python3
"""Run the fixed experiment matrix without touching reference run directories."""

from pathlib import Path
import csv
import datetime
import hashlib
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "trace-sim" / "build" / "trace-sim"
SIM_SOURCE = ROOT / "trace-sim" / "src" / "main.cpp"
OUT = ROOT / "trace-sim" / "results"
TRACES = [
    ROOT / "ref" / "zero_delay_midpoint_v1" / "traces"
    / f"checkpoint_{checkpoint:06d}" / "runtime" / "results" / "firing-trace.jsonl"
    for checkpoint in (10000, 20000, 30000)
]


def main():
    if not SIM.exists():
        raise SystemExit(f"build the simulator first: {SIM}")
    OUT.mkdir(parents=True, exist_ok=True)
    timings = []
    for trace in TRACES:
        checkpoint = trace.parents[2].name
        output = OUT / f"{checkpoint}.csv"
        temporary = output.with_suffix(".csv.tmp")
        start = time.perf_counter()
        completed = subprocess.run(
            [str(SIM), "matrix", "--trace", str(trace), "--output", str(temporary)],
            check=True,
            text=True,
            capture_output=True,
        )
        elapsed = time.perf_counter() - start
        print(completed.stderr.strip())
        with temporary.open() as handle:
            rows = sum(1 for _ in handle) - 1
        temporary.replace(output)
        timings.append({
            "checkpoint": checkpoint.removeprefix("checkpoint_"),
            "trace_bytes": trace.stat().st_size,
            "configurations": rows,
            "matrix_wall_seconds": f"{elapsed:.9f}",
        })

    with (OUT / "matrix-wall-times.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=timings[0].keys())
        writer.writeheader()
        writer.writerows(timings)

    manifest = [
        "schema=stdp-trace-sim-results-v1",
        f"created_utc={datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"platform={platform.platform()}",
        "weight_bytes=4",
        "capacity_bytes=131072,262144,524288,1048576,2097152,4194304",
        "barrier_policy=flush before every attempt; full normalization traffic excluded",
        "same_tick_order=X then E",
        "command=python3 trace-sim/scripts/run_matrix.py",
        f"simulator_source_sha256={sha256(SIM_SOURCE)}",
        f"simulator_sha256={sha256(SIM)}",
        "compiler=" + subprocess.run(
            ["clang++", "--version"], check=True, text=True, capture_output=True
        ).stdout.splitlines()[0],
    ]
    for trace in TRACES:
        manifest.append(f"trace_{trace.parents[2].name}_sha256={sha256(trace)}")
    for output in sorted(OUT.glob("checkpoint_*.csv")):
        manifest.append(f"result_{output.stem}_sha256={sha256(output)}")
    (OUT / "run-manifest.txt").write_text("\n".join(manifest) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
