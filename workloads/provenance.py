from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip()}


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def manifest(args):
    sources = [p for directory in ('workloads', 'reimpl/zd3', 'reimpl/backends',
                                   'brunel/ports', 'cpu-benchmark/mnist_module')
               for p in (ROOT / directory).rglob('*')
               if p.suffix in ('.py', '.h', '.cpp')]
    sources += [ROOT / 'cpu-benchmark/run_nest_mnist.py',
                ROOT / 'genn-sweep/baseline_cases.py']
    return {'schema': 'new-workload-pilot-v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'command': sys.argv, 'cwd': os.getcwd(),
            'configuration': {k: str(v) if isinstance(v, Path) else v
                              for k, v in vars(args).items()},
            'host': platform.uname()._asdict(),
            'python': sys.version,
            'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
            'git_revision': command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
            'environment': {key: os.environ.get(key) for key in (
                'CUDA_VISIBLE_DEVICES', 'CUDA_PATH', 'CUDAHOSTCXX', 'CXX',
                'NVCC_PREPEND_FLAGS', 'LD_LIBRARY_PATH', 'PYTHONPATH',
                'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'SLURM_JOB_ID',
                'OMP_PROC_BIND', 'OMP_PLACES', 'OMP_WAIT_POLICY')},
            'numpy': __import__('numpy').__version__}
