#!/bin/bash
#SBATCH --job-name=yolo_bench
#SBATCH --output=/home/users/%u/logs/bench_ultralytics_%j.out
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

# Container Settings
#SBATCH --container-image=/home/users/hvu/container-images/ultralytics.sqfs
#SBATCH --container-mounts=/home/users/hvu:/workspace
#SBATCH --container-workdir=/workspace/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics

# Reset environment to prevent host-container conflicts
module purge

# Create local logs directory if it doesn't exist
mkdir -p /home/users/$USER/logs

# Environment Variables for Python/Pip
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1

# Configuration and Output Paths (Inside the container)
export YOLO_CONFIG_DIR=/workspace/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/.config/Ultralytics
export MPLCONFIGDIR=/workspace/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/.config/matplotlib
export config_dir=/workspace/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/configs
export out_dir=/workspace/agri-human-dataset-benchmark/2d-detection/reports/benchmarks/summary/ultralytics
export VENV_PATH=/workspace/agri-human-dataset-benchmark/2d-detection/benchmarks/ultralytics/.venv

# TODO: Configure the specific benchmark settings
export config_file='benchmark_ultralytics_zedrgb_to_kitti_filtered.yaml'
export out_file='summary_ultralytics_zedrgb_to_kitti_filtered.csv'

# Single srun block to maintain environment consistency
srun bash -c "
  set -e
  echo '--- System Check ---'
  python -V
  nvidia-smi
  
  echo '--- Setting up Directories ---'
  mkdir -p \$YOLO_CONFIG_DIR \$MPLCONFIGDIR \$out_dir

  echo '--- Virtual Environment Management ---'
  # Create venv if it doesn't exist
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

  echo '--- Starting Benchmark ---'
  # Standard single-node/single-GPU flags
  export RANK=-1
  export LOCAL_RANK=-1
  export WORLD_SIZE=1

  python src/transfer_eval.py \
    --config \$config_dir/\$config_file \
    --out \$out_dir/\$out_file

  echo '--- Job Completed ---'
"