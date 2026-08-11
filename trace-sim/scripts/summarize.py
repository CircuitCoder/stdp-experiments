#!/usr/bin/env python3
"""Create compact aggregate tables consumed by report.md."""

from collections import defaultdict
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "trace-sim" / "results"
SUMMARY = RESULTS / "summary.csv"


def main():
    rows = []
    for path in sorted(RESULTS.glob("checkpoint_*.csv")):
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit("no checkpoint result CSV files found")

    groups = defaultdict(list)
    for row in rows:
        key = (
            row["structure"], row["allocation"], row["depression"],
            row["line_bytes"], row["capacity_bytes"], row["parameter"],
        )
        groups[key].append(row)

    fields = [
        "structure", "allocation", "depression", "line_bytes",
        "capacity_bytes", "parameter", "trace_count", "reads",
        "writebacks", "memory_transactions", "written_words",
        "operator_writebacks", "ordinary_rmw_transactions", "traffic_bytes",
        "forced_evictions", "access_hit_rate", "simulation_seconds",
        "peak_resident_words", "peak_deferred_words", "metadata_bytes",
    ]
    with SUMMARY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, values in sorted(groups.items()):
            reads = sum(int(v["reads"]) for v in values)
            writes = sum(int(v["writebacks"]) for v in values)
            accesses = sum(int(v["accesses"]) for v in values)
            hits = sum(int(v["hits"]) for v in values)
            writer.writerow(dict(
                zip(fields[:6], key),
                trace_count=len(values),
                reads=reads,
                writebacks=writes,
                memory_transactions=reads + writes,
                written_words=sum(int(v["written_words"]) for v in values),
                operator_writebacks=sum(int(v["operator_writebacks"]) for v in values),
                ordinary_rmw_transactions=reads + writes + sum(
                    int(v["operator_writebacks"]) for v in values
                ),
                traffic_bytes=reads * int(values[0]["line_bytes"]) + 4 * sum(
                    int(v["written_words"]) for v in values
                ),
                forced_evictions=sum(int(v["forced_evictions"]) for v in values),
                access_hit_rate=f"{hits / accesses:.9f}" if accesses else "0",
                simulation_seconds=f"{sum(float(v['simulation_seconds']) for v in values):.9f}",
                peak_resident_words=max(int(v["peak_resident_words"]) for v in values),
                peak_deferred_words=max(int(v["peak_deferred_words"]) for v in values),
                metadata_bytes=int(values[0]["metadata_bytes"]),
            ))


if __name__ == "__main__":
    main()
