#!/bin/bash
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd $PROJECT_DIR

echo "=========================================="
echo "Work Dir: $PROJECT_DIR"
echo "Config: configs/hpc_a800.yaml"
echo "Time: $(date)"
echo "=========================================="

export HF_ENDPOINT=https://hf-mirror.com
mkdir -p logs outputs

source ~/.bashrc 2>/dev/null
conda activate cafa6 2>/dev/null || source activate cafa6 2>/dev/null

# 重新开始训练（不resume）
python -u src/training/train_hpc.py \
    --config configs/hpc_a800.yaml \
    2>&1 | tee logs/a800_train.log
