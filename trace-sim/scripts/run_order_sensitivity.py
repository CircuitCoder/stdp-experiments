#!/usr/bin/env python3
"""Compare X-before-E and E-before-X for representative 30k configurations."""

from pathlib import Path
import csv
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "trace-sim" / "build" / "trace-sim"
TRACE = (ROOT / "ref" / "zero_delay_midpoint_v1" / "traces" /
         "checkpoint_030000" / "runtime" / "results" / "firing-trace.jsonl")
OUTPUT = ROOT / "trace-sim" / "results" / "order-sensitivity.csv"
CONFIGS = [
    ("set-line", "--ways", "8", "write-back"),
    ("set-fine", "--ways", "8", "write-back"),
    ("hash-offset", "--distance", "8", "write-back"),
    ("hash-offset", "--distance", "8", "drain"),
]
FIELDS = [
    "event_order", "structure", "depression", "reads", "writebacks",
    "written_words", "forced_evictions", "hits", "simulation_seconds",
]


def main():
    rows = []
    with tempfile.TemporaryDirectory(prefix="stdp-order-") as temp:
        for order in ("x-e", "e-x"):
            for index, (structure, parameter_name, parameter, depression) in enumerate(CONFIGS):
                output = Path(temp) / f"{order}-{index}.csv"
                subprocess.run([
                    str(SIM), "run", "--trace", str(TRACE), "--output", str(output),
                    "--event-order", order, "--structure", structure, "--line-size", "64",
                    "--capacity", "1MiB", parameter_name, parameter,
                    "--allocation", "deferred", "--depression", depression,
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with output.open(newline="") as handle:
                    result = next(csv.DictReader(handle))
                rows.append({field: result.get(field, order) for field in FIELDS})
                rows[-1]["event_order"] = order
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
