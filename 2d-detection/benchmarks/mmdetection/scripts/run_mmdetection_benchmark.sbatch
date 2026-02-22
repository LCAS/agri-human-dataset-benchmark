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
#SBATCH --container-workdir=/workspace/repo/2d-detector-benchmark

module purge
mkdir -p /home/users/$USER/mmdet_frcnn_dcn/logs

export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29520

export REPO_ROOT=/workspace/repo/2d-detector-benchmark
export MMDET_ROOT=$REPO_ROOT/2d-detection/third_party/mmdetection
export BENCH_ROOT=$REPO_ROOT/2d-detection/benchmarks/mmdetection
export BENCH_CONFIG=$BENCH_ROOT/configs/benchmark_mmdetection_fisheye.yaml
export BENCH_OUT=$REPO_ROOT/2d-detection/reports/benchmarks/mmdetection/summary_fisheye_mmdet.csv

srun bash -lc "python -V && nvidia-smi || true"

srun bash -lc '
set -e

export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export MIM_CACHE_DIR=/workspace/.cache/mim
export TORCH_HOME=/workspace/.cache/torch

mkdir -p /workspace/.cache/{pip,torch,mim}
mkdir -p /workspace/.config

if [ ! -d /workspace/.venv_mmdet ]; then
  python -m venv /workspace/.venv_mmdet
fi
source /workspace/.venv_mmdet/bin/activate

pip install -U pip
pip install "setuptools>=60.2.0,<70" wheel
pip install "numpy<2"
pip install "opencv-python-headless==4.11.0.86"
pip install "torch==2.1.2+cu118" "torchvision==0.16.2+cu118" --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
pip install mmengine
pip install "mmcv==2.1.0" -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install -r $MMDET_ROOT/requirements.txt
pip install --no-build-isolation -v -e $MMDET_ROOT
pip install -U certifi

python -c "import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())"
python -c "import mmcv, mmengine, mmdet; print(\"mmcv:\", mmcv.__version__, \"mmengine:\", mmengine.__version__, \"mmdet:\", mmdet.__version__)"
'

srun bash -lc '
set -e
export HOME=/workspace
export XDG_CACHE_HOME=/workspace/.cache
export TORCH_HOME=/workspace/.cache/torch
source /workspace/.venv_mmdet/bin/activate

export SSL_CERT_FILE="$(python -c "import certifi; print(certifi.where())")"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

echo "=== Starting MMDetection benchmark (train + eval + summary export) ==="
python $BENCH_ROOT/tools/run_benchmark_summary.py \
  --config $BENCH_CONFIG \
  --out $BENCH_OUT
'
