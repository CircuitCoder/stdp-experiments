#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 NEST_CONFIG BUILD_DIRECTORY" >&2
  exit 2
fi

nest_config=$1
build_directory=$2
cmake -S "$(dirname "$0")" -B "$build_directory" \
  -Dwith-nest="$nest_config" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_directory" --parallel
