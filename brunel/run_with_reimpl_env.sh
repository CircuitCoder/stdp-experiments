#!/usr/bin/env bash
set -euo pipefail

repository=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python=/tmp/stdp-reimpl-venv/bin/python

if [[ ! -x "${python}" ]]; then
    echo "Framework environment not found at ${python}" >&2
    exit 1
fi

runtime_libraries=$(nix eval --raw --impure --expr \
    'with import <nixpkgs> {}; lib.makeLibraryPath [ stdenv.cc.cc.lib zlib ]')

export LD_LIBRARY_PATH="${runtime_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec nix shell nixpkgs#gnumake nixpkgs#gcc --command "${python}" "$@"
