#!/bin/bash
#SBATCH --job-name=norfair_track
#SBATCH --output=/home/users/%u/logs/norfair_track_%j.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus=1

set -euo pipefail

# TODO: Adjust these paths for your cluster environment.
export REPO_ROOT=/workspace/agri-human-dataset-benchmark
export BENCH_ROOT=$REPO_ROOT/2d-tracking/benchmarks/norfair
export CONFIG_FILE=${CONFIG_FILE:-$BENCH_ROOT/configs/tracking/default.yaml}

# TODO: Replace this with your environment activation method.
# Example:
# source /path/to/venv/bin/activate
# python -m pip install -r $BENCH_ROOT/requirements.txt

python $BENCH_ROOT/src/run_tracker.py --config $CONFIG_FILE
