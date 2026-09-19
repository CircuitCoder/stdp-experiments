#!/usr/bin/env bash
set -euo pipefail
build_root=${1:?pass a fresh absolute build directory}
mkdir "${build_root}"
git -C 3rdparty/nest-simulator rev-parse HEAD > "${build_root}/source-commit.txt"
python - "${build_root}/environment.json" <<'PY'
import json, os, sys
from pathlib import Path
keys = ['PATH', 'LD_LIBRARY_PATH', 'CXX', 'NIX_CFLAGS_COMPILE',
        'NIX_ENFORCE_NO_NATIVE', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']
Path(sys.argv[1]).write_text(json.dumps({k: os.environ.get(k) for k in keys}, indent=2))
PY
export CXX=g++
cmake -S 3rdparty/nest-simulator -B "${build_root}/build" \
  -DCMAKE_INSTALL_PREFIX="${build_root}/install" -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_CXX_FLAGS_RELEASE=-O3 -march=native -DNDEBUG' \
  -Dwith-python=ON -Dwith-openmp=ON -Dwith-mpi=OFF -Dwith-readline=OFF
cmake --build "${build_root}/build" --parallel 8
cmake --install "${build_root}/build"
touch "${build_root}/COMPLETE"
