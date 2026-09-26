#!/bin/bash -l
set -euo pipefail
baseline_root="$HOME/meow/stdp-experiments"
experiment_root="$baseline_root/workload-pilots-20260920"
source_root="$experiment_root/${WORKLOAD_SOURCE:-source_v3}"
cd "$source_root"
module load soft/anaconda3/config gpu/v12.8.1 compilers/gcc/v12.4.0
export TMPDIR="$baseline_root/tmp" PIP_CACHE_DIR="$baseline_root/cache/pip"
export XDG_CACHE_HOME="$baseline_root/cache" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$source_root/reimpl:$source_root/brunel:$baseline_root/3rdparty/genn"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_PATH=/apps/gpu/cuda/v12.8.1
export CUDAHOSTCXX=/apps/compilers/gcc/v12.4.0/bin/g++
export CC=/apps/compilers/gcc/v12.4.0/bin/gcc CXX="$CUDAHOSTCXX"
export NVCC_PREPEND_FLAGS="-I$CUDA_PATH/include -ccbin=$CUDAHOSTCXX"
export PATH="$baseline_root/.venv/bin:$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$baseline_root/3rdparty/genn/pygenn:$CUDA_PATH/lib64:/apps/compilers/gcc/v12.4.0/lib64:${LD_LIBRARY_PATH:-}"
date -u
hostname
scontrol show job -dd "$SLURM_JOB_ID"
nvidia-smi
srun --cpu-bind=cores python "$baseline_root/helpers/with_slurm_cpus.py" "$@"
