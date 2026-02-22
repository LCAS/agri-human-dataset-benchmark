#!/bin/bash
#SBATCH --job-name=mmdet_frcnn_dcn
#SBATCH --output=/home/users/%u/mmdet_frcnn_dcn/logs/bench_%j.out
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

#SBATCH --container-image=/home/users/hvu/mmdet_frcnn_dcn/mmdet_pytorch_cu118.sqfs
#SBATCH --container-mounts=/home/users/hvu/mmdet_frcnn_dcn:/workspace
#SBATCH --container-workdir=/workspace/repo/mmdetection

module purge
mkdir -p /home/users/$USER/mmdet_frcnn_dcn/logs

# Environment setup
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29520

# MMDet config
export MMDET_CONFIG=configs/faster_rcnn/faster-rcnn_r50_fpn_dcn_fisheye.py
export WORK_DIR=work_dirs/faster-rcnn_r50_fpn_dcn_fisheye

# Sanity check
srun bash -lc "python -V && nvidia-smi || true"

# Setup venv and install dependencies
srun bash -lc '
set -e

# CRITICAL: Redirect ALL cache/home writes to writable workspace
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export MIM_CACHE_DIR=/workspace/.cache/mim
export TORCH_HOME=/workspace/.cache/torch

# Create cache dirs
mkdir -p /workspace/.cache/{pip,torch,mim}
mkdir -p /workspace/.config

# Create venv WITHOUT system-site-packages to avoid conflicts
if [ ! -d /workspace/.venv_mmdet ]; then
  python -m venv /workspace/.venv_mmdet
  source /workspace/.venv_mmdet/bin/activate
  
  # Upgrade pip and install compatible setuptools
  pip install -U pip
  pip install "setuptools>=60.2.0,<70"
  pip install wheel
  
  # Install torch FIRST (needed for mmdet setup.py)
  pip install "torch==2.1.2+cu118" "torchvision==0.16.2+cu118" --index-url https://download.pytorch.org/whl/cu118

  
  echo "PyTorch installed:"
  python -c "import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())"
  
  # Install OpenMMLab stack (use prebuilt wheels)
  pip install -U openmim
  pip install mmengine
  pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html


  # Install mmdet requirements BEFORE editable install
  pip install -r requirements.txt

  # Install mmdet with --no-build-isolation (uses already-installed torch)
  pip install --no-build-isolation -v .
else
  source /workspace/.venv_mmdet/bin/activate
  python -m pip uninstall -y opencv-python opencv-python-headless numpy
  python -m pip install "numpy<2"
  python -m pip install "opencv-python-headless==4.11.0.86"
fi

# Verify installation
python -c "import numpy; print(numpy.__version__, numpy.__file__)"
python -c "import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())"
python -c "import mmcv, mmengine, mmdet; print(\"mmcv:\", mmcv.__version__, \"mmengine:\", mmengine.__version__, \"mmdet:\", mmdet.__version__)"
'

# Train
srun bash -lc '
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/.cache/torch
source /workspace/.venv_mmdet/bin/activate

pip install -U certifi
export SSL_CERT_FILE="$(python -c "import certifi; print(certifi.where())")"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
python tools/train.py $MMDET_CONFIG --work-dir $WORK_DIR
'

