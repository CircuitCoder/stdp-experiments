#!/usr/bin/env python3
"""Validate the complete matrix and trace reconstruction invariants."""

from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "trace-sim" / "results"
EXPECTED = {
    10000: (8354032, {32: 1128632, 64: 612532, 128: 364804}),
    20000: (8429392, {32: 1165492, 64: 646642, 128: 397594}),
    30000: (8401904, {32: 1191554, 64: 676529, 128: 429317}),
}


def expected_configurations():
    result = set()
    for line_bytes in (32, 64, 128):
        for allocation in ("eager", "deferred"):
            for depression in ("write-back", "drain"):
                result.add(("direct", allocation, depression, str(line_bytes), "0", "0"))
        for capacity in (131072, 262144, 524288, 1048576, 2097152, 4194304):
            for allocation in ("eager", "deferred"):
                for depression in ("write-back", "drain"):
                    for ways in (1, 2, 4, 8):
                        for structure in ("set-line", "set-fine"):
                            result.add((structure, allocation, depression,
                                        str(line_bytes), str(capacity), str(ways)))
                    for distance in (1, 2, 4, 8):
                        for structure in ("hash-line", "hash-weight", "hash-offset"):
                            result.add((structure, allocation, depression,
                                        str(line_bytes), str(capacity), str(distance)))
    return result


def main():
    for checkpoint, (updates, accesses) in EXPECTED.items():
        path = RESULTS / f"checkpoint_{checkpoint:06d}.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        expected = expected_configurations()
        assert len(rows) == len(expected), (path, len(rows), len(expected))
        keys = {
            (r["structure"], r["allocation"], r["depression"], r["line_bytes"],
             r["capacity_bytes"], r["parameter"])
            for r in rows
        }
        assert keys == expected
        for row in rows:
            integer = {key: int(row[key]) for key in (
                "accesses", "hits", "logical_updates", "writebacks",
                "eviction_writebacks", "drain_writebacks", "barrier_writebacks",
                "final_writebacks", "peak_resident_words", "peak_deferred_words",
                "capacity_bytes",
            )}
            assert integer["logical_updates"] == updates
            assert integer["accesses"] == accesses[int(row["line_bytes"])]
            assert integer["hits"] <= integer["accesses"]
            assert integer["peak_deferred_words"] <= integer["peak_resident_words"]
            assert integer["writebacks"] == sum(integer[key] for key in (
                "eviction_writebacks", "drain_writebacks", "barrier_writebacks",
                "final_writebacks",
            ))
            if row["structure"] != "direct":
                assert 4 * integer["peak_resident_words"] <= integer["capacity_bytes"]
            else:
                assert integer["writebacks"] == integer["logical_updates"]
                assert int(row["written_words"]) == integer["logical_updates"]
                if row["allocation"] == "eager":
                    assert int(row["reads"]) == integer["logical_updates"]
                    assert int(row["operator_writebacks"]) == 0
                else:
                    assert int(row["reads"]) == int(row["continuous_updates"])
                    assert int(row["operator_writebacks"]) == int(row["sparse_updates"])
        print(f"checkpoint {checkpoint}: {len(expected)} configurations passed")


if __name__ == "__main__":
    main()
