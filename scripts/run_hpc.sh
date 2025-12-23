#!/bin/bash
#SBATCH --job-name=cafa6_train
#SBATCH --output=logs/cafa6_%j.out
#SBATCH --error=logs/cafa6_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Location: scripts/run_hpc.sh
# Usage: sbatch scripts/run_hpc.sh [config_name]
# Example: sbatch scripts/run_hpc.sh hpc_650m

# =============================================================================
# Configuration
# =============================================================================
CONFIG_NAME=${1:-hpc_650m}  # Default to hpc_650m if not specified
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
CONFIG_PATH="${PROJECT_DIR}/configs/${CONFIG_NAME}.yaml"

echo "========================================"
echo "CAFA6 Training Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Config: $CONFIG_PATH"
echo "Start time: $(date)"
echo "========================================"

# =============================================================================
# Environment Setup
# =============================================================================
cd $PROJECT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate virtual environment (adjust path as needed)
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated venv"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Activated .venv"
else
    # Try conda
    source ~/.bashrc
    conda activate cafa6 2>/dev/null || echo "No conda environment 'cafa6' found"
fi

# Print Python and CUDA info
echo ""
echo "Environment Info:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# =============================================================================
# Training
# =============================================================================
echo "Starting training..."
echo ""

python src/training/train_hpc.py \
    --config "$CONFIG_PATH"

echo ""
echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"

