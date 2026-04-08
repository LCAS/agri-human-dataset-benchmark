#!/bin/bash
#SBATCH --job-name=yolo_dump_detections
#SBATCH --output=/home/users/%u/logs/dump_detections_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=multithread
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hvu@lincoln.ac.uk

# Container Settings
#SBATCH --container-image=/home/users/hvu/container-images/ultralytics.sqfs
#SBATCH --container-mounts=/home/users/hvu:/workspace
#SBATCH --container-workdir=/workspace/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics

# Reset environment to prevent host-container conflicts
module purge

# Create local logs directory if it doesn't exist
mkdir -p /home/users/$USER/logs

# Environment Variables
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1

# Paths
export config_dir=/workspace/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/configs
export VENV_PATH=/workspace/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/.venv

# TODO: Set your predict config file
export config_file='predict_in_straw_3pick_diff_st_10_24_2024_5_a_label_zed_rgb.yaml'

srun bash -c "
  set -e
  echo '--- System Check ---'
  python -V
  nvidia-smi

  echo '--- Setting up Directories ---'
  mkdir -p \$config_dir

  echo '--- Virtual Environment Management ---'
  if [ ! -d \$VENV_PATH ]; then
    echo 'Creating new virtual environment...'
    python -m venv \$VENV_PATH
    source \$VENV_PATH/bin/activate
    pip install -U pip
    if [ -f requirements.txt ]; then
      pip install -r requirements.txt
    fi
  else
    echo 'Using existing virtual environment.'
    source \$VENV_PATH/bin/activate
  fi

  echo '--- Hardware Verification ---'
  python -c 'import torch; print(\"CUDA Available:\", torch.cuda.is_available()); print(\"Device Name:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\")'

  echo '--- Starting Detection Dump ---'
  export RANK=-1
  export LOCAL_RANK=-1
  export WORLD_SIZE=1

  python src/dump_detections.py \
    --config \$config_dir/\$config_file

  echo '--- Job Completed ---'
"