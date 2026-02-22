#!/bin/bash
#SBATCH --job-name=mmdet_frcnn_dcn
#SBATCH --output=/home/users/%u/mmdet_frcnn_dcn/logs/bench_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=multithread
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --qos=long
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
export BENCHMARK_CONFIG=configs/benchmarks/faster-rcnn_r50_fpn_dcn_fisheye_benchmark.yaml
export BENCHMARK_OUT=runs/benchmarks/summary_fisheye_mmdet.csv

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

# Create venv and install all dependencies
if [ ! -d /workspace/.venv_mmdet ]; then
  echo "=== Creating new virtual environment ==="
  python -m venv /workspace/.venv_mmdet
  source /workspace/.venv_mmdet/bin/activate
  
  # Upgrade pip and install compatible setuptools
  pip install -U pip
  pip install "setuptools>=60.2.0,<70"
  pip install wheel
  
  # Install numpy FIRST with version constraint (critical for mmcv compatibility)
  pip install "numpy<2"
  
  # Install torch with exact versions matching your CUDA
  pip install "torch==2.1.2+cu118" "torchvision==0.16.2+cu118" --index-url https://download.pytorch.org/whl/cu118
  
  echo "=== PyTorch installed ==="
  python -c "import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())"
  
  # CRITICAL: Install opencv-python-headless BEFORE mmcv to prevent full opencv
  pip install "opencv-python-headless==4.11.0.86"
  
  # Install OpenMMLab stack with prebuilt wheels
  pip install -U openmim
  pip install mmengine
  pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
  
  # Remove any full opencv that might have been pulled in and reinstall headless
  pip uninstall -y opencv-python opencv-contrib-python 2>/dev/null || true
  pip install --force-reinstall --no-deps "opencv-python-headless==4.11.0.86"
  
  echo "=== mmcv installed ==="
  python -c "import cv2; print(\"OpenCV:\", cv2.__version__)"
  python -c "import mmcv; print(\"mmcv:\", mmcv.__version__)"
  
  # Install mmdet requirements (modify to use headless opencv)
  if [ -f requirements.txt ]; then
    # Install requirements but skip opencv-python if present, edit to remote it from the file if needed
    pip install -r requirements.txt
  fi
  
  # Install mmdet itself with --no-build-isolation
  pip install --no-build-isolation -v .
  
  # Install certifi for SSL
  pip install -U certifi
  
  echo "=== Installation complete ==="
else
  echo "=== Using existing virtual environment ==="
  source /workspace/.venv_mmdet/bin/activate
fi

# Check numpy version and fix if needed
  NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "missing")
  if [[ "$NUMPY_VERSION" == "missing" ]] || [[ "$NUMPY_VERSION" == 2.* ]]; then
    echo "Fixing numpy version (current: $NUMPY_VERSION)..."
    pip uninstall -y numpy 2>/dev/null || true
    pip install "numpy<2"
  fi
  
# Check opencv and fix if needed
python -c "import cv2" 2>/dev/null || {
  echo "Installing opencv-python-headless..."
  pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
  pip install "opencv-python-headless==4.11.0.86"
}

# Verify final installation
echo "=== Verifying installation ==="
python -c "import numpy; print(\"numpy:\", numpy.__version__, numpy.__file__)"
python -c "import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())"
python -c "import cv2; print(\"OpenCV:\", cv2.__version__)"
python -c "import mmcv, mmengine, mmdet; print(\"mmcv:\", mmcv.__version__, \"mmengine:\", mmengine.__version__, \"mmdet:\", mmdet.__version__)"
echo "=== All checks passed ==="
'

# # Train
# srun bash -lc '
# export HOME=/workspace
# export XDG_CACHE_HOME=/workspace/.cache
# export TORCH_HOME=/workspace/.cache/torch
# source /workspace/.venv_mmdet/bin/activate

# # Set SSL certificate paths
# export SSL_CERT_FILE="$(python -c "import certifi; print(certifi.where())")"
# export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# export RANK=0
# export LOCAL_RANK=0
# export WORLD_SIZE=1

# echo "=== Starting training ==="
# python tools/train.py $MMDET_CONFIG --work-dir $WORK_DIR --resume
# '

# # Test
# srun bash -lc '
# export HOME=/workspace
# export XDG_CACHE_HOME=/workspace/.cache
# export TORCH_HOME=/workspace/.cache/torch
# source /workspace/.venv_mmdet/bin/activate

# # Set SSL certificate paths
# export SSL_CERT_FILE="$(python -c "import certifi; print(certifi.where())")"
# export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# echo "=== Starting testing ==="
# python tools/test.py $MMDET_CONFIG $WORK_DIR/epoch_18.pth
# '

# srun bash -lc '

# export HOME=/workspace
# export XDG_CACHE_HOME=/workspace/.cache
# export TORCH_HOME=/workspace/.cache/torch
# source /workspace/.venv_mmdet/bin/activate

# # Set SSL certificate paths
# export SSL_CERT_FILE="$(python -c "import certifi; print(certifi.where())")"
# export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# echo "=== Starting testing ==="

# for ckpt in $WORK_DIR/epoch_*.pth; do
#   echo "== $ckpt =="
#   python tools/test.py $MMDET_CONFIG $ckpt | tail -n 20
# done
# '

srun bash -lc '
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/.cache/torch
source /workspace/.venv_mmdet/bin/activate

# Set SSL certificate paths
export SSL_CERT_FILE="$(python -c "import certifi; print(certifi.where())")"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

echo "=== Starting benchmark (train + eval + summary export) ==="
python tools/run_benchmark_summary.py \
  --config $BENCHMARK_CONFIG \
  --out $BENCHMARK_OUT
'