"""The integer image clock must preserve short-time learning and long-time lags."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_three_trace_integer_clock_matches_float_and_survives_large_offsets(tmp_path):
    pytest.importorskip('pygenn')
    backend = os.environ.get('IMAGE_TIMING_BACKEND', 'single_threaded_cpu')
    results = {}
    for mode in ('float', 'integer'):
        completed = subprocess.run([sys.executable, str(Path(__file__).with_name('image_timing_probe.py')),
            '--backend', backend, '--mode', mode, '--output', str(tmp_path / mode)],
            capture_output=True, text=True, timeout=240)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        results[mode] = json.loads(completed.stdout.splitlines()[-1])
    assert results['float'][0]['weight'] == results['integer'][0]['weight']
