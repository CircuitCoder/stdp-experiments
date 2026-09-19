#!/usr/bin/env bash
set -euo pipefail
out=${1:?fresh output directory required}
prefix=${2:?NEST installation prefix required}
module=${3:?MNIST module required}
if [[ -e "$out" ]]; then
  printf 'Output already exists: %s\n' "$out" >&2
  exit 1
fi
mkdir "$out"
for threads in 1 2 4 8 16 32; do
  OMP_WAIT_POLICY=ACTIVE GOMP_SPINCOUNT=300000 OMP_PROC_BIND=spread OMP_PLACES=cores \
    python cpu-benchmark/run_nest_mnist.py --nest-prefix "$prefix" --module "$module" \
      --output "$out/t$threads" --threads "$threads" --samples 5 --repetitions 2 \
      --cases mnist_1trace_dense mnist_2trace_dense mnist_3trace_dense \
      > "$out/t$threads.log" 2>&1
  printf 'Completed thread pilot: %s\n' "$threads"
done
printf 'COMPLETE\n' > "$out/COMPLETE"
