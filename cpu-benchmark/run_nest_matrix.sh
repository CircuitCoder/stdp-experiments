#!/usr/bin/env bash
set -euo pipefail
output_root=${1:?fresh output directory required}
nest_prefix=${2:?source-built NEST prefix required}
nest_module=${3:?compiled extension required}
mkdir "${output_root}"
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=0
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
for nest_threads in 16 32; do
  for nest_rule in additive morrison; do
    python cpu-benchmark/run_nest.py \
      --nest-prefix "${nest_prefix}" --module "${nest_module}" \
      --output "${output_root}/${nest_rule}-t${nest_threads}" \
      --threads "${nest_threads}" --repetitions 5 --rule "${nest_rule}" \
      > "${output_root}/${nest_rule}-t${nest_threads}.log" 2>&1
  done
done
touch "${output_root}/COMPLETE"
