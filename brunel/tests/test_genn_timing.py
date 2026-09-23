"""Exercise actual GeNN kernels, isolating intentional hard errors in children."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


PROBE = Path(__file__).with_name("genn_timing_probe.py")
BACKEND = os.environ.get("GENN_TIMING_TEST_BACKEND", "single_threaded_cpu")


def run_probe(tmp_path, case):
    pytest.importorskip("pygenn")
    out = tmp_path / "probe"
    result = subprocess.run(
        [sys.executable, str(PROBE), "--backend", BACKEND, "--output", str(out),
         "--case", case],
        capture_output=True, text=True, timeout=180)
    (tmp_path / "stdout.log").write_text(result.stdout)
    (tmp_path / "stderr.log").write_text(result.stderr)
    return result, out


def test_generated_ties_adjacent_ticks_and_existing_traces(tmp_path):
    result, out = run_probe(tmp_path, "pairs")
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads((out / "result.json").read_text())
    assert len(rows) == 208
    assert max(row["start_tick"] for row in rows) > (1 << 31)


@pytest.mark.parametrize("case", ["missing-spike", "future-spike", "future-trace-pre",
                                  "future-post-pre", "future-trace-post", "future-post-post",
                                  "future-pre-post", "clock-overflow", "clock-overflow-delayed",
                                  "arrival-overflow"])
def test_generated_invalid_integer_timestamps_hard_fail(tmp_path, case):
    result, _ = run_probe(tmp_path, case)
    assert result.returncode != 0
    assert "assert" in result.stderr.lower(), result.stdout + result.stderr
    assert "tick" in result.stderr.lower(), result.stdout + result.stderr
    assert "Timing guard failed" not in result.stderr
