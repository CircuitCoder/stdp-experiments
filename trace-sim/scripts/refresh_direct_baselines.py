#!/usr/bin/env python3
"""Refresh only direct-memory rows after changing baseline accounting."""

from pathlib import Path
import csv
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / "trace-sim" / "build" / "trace-sim"
RESULTS = ROOT / "trace-sim" / "results"


def main():
    refreshed = 0
    with tempfile.TemporaryDirectory(prefix="stdp-direct-") as temp:
        for checkpoint in (10000, 20000, 30000):
            trace = (ROOT / "ref" / "zero_delay_midpoint_v1" / "traces" /
                     f"checkpoint_{checkpoint:06d}" / "runtime" / "results" /
                     "firing-trace.jsonl")
            result_path = RESULTS / f"checkpoint_{checkpoint:06d}.csv"
            with result_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames
                rows = list(reader)
            for index, row in enumerate(rows):
                if row["structure"] != "direct":
                    continue
                output = Path(temp) / f"{checkpoint}-{index}.csv"
                subprocess.run([
                    str(SIM), "run", "--trace", str(trace), "--output", str(output),
                    "--structure", "direct", "--line-size", row["line_bytes"],
                    "--capacity", "0", "--ways", "0", "--allocation",
                    row["allocation"], "--depression", row["depression"],
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                with output.open(newline="") as handle:
                    rows[index] = next(csv.DictReader(handle))
                refreshed += 1
            temporary = result_path.with_suffix(".csv.tmp")
            with temporary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            temporary.replace(result_path)
    print(f"refreshed {refreshed} direct-memory rows")


if __name__ == "__main__":
    main()
