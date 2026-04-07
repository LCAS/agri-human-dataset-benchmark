#!/bin/bash
#SBATCH --job-name=mmdet_bench
#SBATCH --output=/home/users/%u/logs/bench_mmdetection_%j.out
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --qos=long
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=hvu@lincoln.ac.uk

# Container Settings
#SBATCH --container-image=/home/users/hvu/mmdet_frcnn_dcn/mmdet_pytorch_cu118.sqfs
#SBATCH --container-mounts=/home/users/hvu:/workspace
#SBATCH --container-workdir=/workspace/agri-human-dataset-benchmark/2d-detection/benchmarks/mmdetection

module purge

# Environment Variables (Inherited by Container)
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29520

# Path Configuration
export REPO_ROOT=/workspace/agri-human-dataset-benchmark
export BENCH_ROOT=$REPO_ROOT/2d-detection/benchmarks/mmdetection
export VENV_PATH=$BENCH_ROOT/.venv
export CACHE_ROOT=$BENCH_ROOT/.cache
export OUT_DIR=$REPO_ROOT/2d-detection/reports/benchmarks/summary/mmdetection

# TODO: Benchmark Settings
export CONFIG_FILE='benchmark_mmdetection_fieldsafepedestrian_to_zedrgb.yaml'
export OUT_FILE='summary_mmdetection_fieldsafepedestrian_to_zedrgb.csv'

srun bash -c "
    set -e
    
    echo '--- Setting up Directories ---'
    mkdir -p \$OUT_DIR \$CACHE_ROOT/pip \$CACHE_ROOT/torch \$CACHE_ROOT/mim \$BENCH_ROOT/.config

    # Re-route home to the writeable workspace
    export HOME=/workspace
    export XDG_CACHE_HOME=\$CACHE_ROOT
    export MIM_CACHE_DIR=\$CACHE_ROOT/mim
    export TORCH_HOME=\$CACHE_ROOT/torch

    echo '--- Virtual Environment Management ---'
    if [ ! -d \$VENV_PATH ]; then
        python -m venv \$VENV_PATH
        source \$VENV_PATH/bin/activate
        pip install -U pip setuptools wheel
        pip install 'numpy<2' 'opencv-python-headless==4.11.0.86'
        pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
        pip install -U openmim
        pip install 'mmcv==2.1.0' -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
        pip install -r \$BENCH_ROOT/requirements.txt
        pip install certifi
    else
        source \$VENV_PATH/bin/activate
    fi

    # Check numpy version and fix if needed
    _NP_VER=\$(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'missing')
    if [[ \"\$_NP_VER\" == 2.* ]] || [[ \"\$_NP_VER\" == \"missing\" ]]; then
        echo \"Fixing/Installing numpy - current: \$_NP_VER\"
        pip install 'numpy<2'
    fi

    # Check opencv and fix if needed
    if ! python -c 'import cv2' 2>/dev/null; then
        echo \"Installing opencv-python-headless...\"
        # Uninstall both to prevent conflicts between 'headless' and standard 'opencv-python'
        pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
        pip install \"opencv-python-headless==4.11.0.86\"
    fi

    echo '--- Verification ---'
    export SSL_CERT_FILE=\$(python -c 'import certifi; print(certifi.where())')
    export REQUESTS_CA_BUNDLE=\$SSL_CERT_FILE
    
    python -c 'import torch, mmdet; print(f\"PyTorch: {torch.__version__} | MMDetection: {mmdet.__version__}\")'

    echo '--- Starting Benchmark ---'
    python src/run_benchmark.py \
        --config \$BENCH_ROOT/configs/\$CONFIG_FILE \
        --out \$OUT_DIR/\$OUT_FILE
"
