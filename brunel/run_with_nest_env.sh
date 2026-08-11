#!/usr/bin/env bash
set -euo pipefail

repository=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
nest_python=/tmp/zd3-nest-source-install-c/lib64/python3.13/site-packages

if [[ ! -d "${nest_python}/nest" ]]; then
    echo "Source-built PyNEST not found at ${nest_python}" >&2
    exit 1
fi

exec nix shell --impure \
    --expr 'with import <nixpkgs> {}; python313.withPackages (ps: [ps.numpy ps.scipy])' \
    --command env PYTHONPATH="${nest_python}" \
    python "${repository}/brunel/nest_brunel_stdp.py" "$@"

