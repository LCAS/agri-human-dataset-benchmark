#!/bin/bash
#SBATCH --job-name=yolo_bench
#SBATCH --output=/home/users/%u/yolo_bench/logs/bench_%j.out
#SBATCH --time=0-12:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=multithread
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --qos=normal
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hvu@lincoln.ac.uk

#SBATCH --container-image=/home/users/hvu/yolo_bench/ultralytics.sqfs
#SBATCH --container-mounts=/home/users/hvu/yolo_bench:/workspace
#SBATCH --container-workdir=/workspace/repo/2d-detector-benchmark

module purge

mkdir -p /home/users/$USER/yolo_bench/logs
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

export YOLO_CONFIG_DIR=/workspace/.config/Ultralytics
export MPLCONFIGDIR=/workspace/.config/matplotlib

export REPO_ROOT=/workspace/repo/2d-detector-benchmark
export BENCH_ROOT=$REPO_ROOT/2d-detection/benchmarks/ultralytics
export BENCH_CONFIG=$BENCH_ROOT/configs/benchmark_ultralytics_fisheye.yaml
export BENCH_OUT=$REPO_ROOT/2d-detection/reports/benchmarks/ultralytics/summary_fisheye.csv

# Basic sanity check
srun bash -lc "python -V && nvidia-smi || true"

srun bash -lc '
set -e
cd $BENCH_ROOT
mkdir -p /workspace/.config/Ultralytics /workspace/.config/matplotlib
if [ ! -d /workspace/.venv ]; then
  python -m venv /workspace/.venv
  source /workspace/.venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt
else
  source /workspace/.venv/bin/activate
fi

python -c "import torch; print(\"CUDA:\", torch.cuda.is_available())"
'

srun bash -lc '
set -e
cd $REPO_ROOT
source /workspace/.venv/bin/activate

export RANK=-1
export LOCAL_RANK=-1
export WORLD_SIZE=1

python $BENCH_ROOT/src/run_benchmark.py --config $BENCH_CONFIG --out $BENCH_OUT
'
