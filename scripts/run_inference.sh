#!/bin/bash
#SBATCH --job-name=cafa6_infer
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Location: scripts/run_inference.sh
# Usage: sbatch scripts/run_inference.sh [experiment_dir]
# Example: sbatch scripts/run_inference.sh outputs/hpc_650m

# =============================================================================
# Configuration
# =============================================================================
EXPERIMENT_DIR=${1:-outputs/hpc_650m}
PROJECT_DIR=$(dirname $(dirname $(realpath $0)))
CHECKPOINT="${EXPERIMENT_DIR}/best_model.pt"
PROCESSOR="${EXPERIMENT_DIR}/go_processor.pkl"
TEST_FASTA="data/Test/testsuperset.fasta"
OUTPUT="${EXPERIMENT_DIR}/submission.tsv"

echo "========================================"
echo "CAFA6 Inference Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment: $EXPERIMENT_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"
echo "Start time: $(date)"
echo "========================================"

# =============================================================================
# Environment Setup
# =============================================================================
cd $PROJECT_DIR
mkdir -p logs

# Activate environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    source ~/.bashrc
    conda activate cafa6 2>/dev/null
fi

echo ""
echo "Environment Info:"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# =============================================================================
# Inference
# =============================================================================
echo "Starting inference..."
echo ""

python src/inference.py \
    --checkpoint "$CHECKPOINT" \
    --processor "$PROCESSOR" \
    --test_fasta "$TEST_FASTA" \
    --output "$OUTPUT" \
    --batch_size 16 \
    --threshold 0.01 \
    --max_terms 1500 \
    --propagate \
    --use_amp

echo ""
echo "========================================"
echo "Inference completed at: $(date)"
echo "Output: $OUTPUT"
echo "========================================"

# Show file stats
wc -l "$OUTPUT"
head -20 "$OUTPUT"

