#!/bin/bash
#SBATCH --job-name=eval_transfer
#SBATCH --output=logs/eval_transfer_%j.out
#SBATCH --error=logs/eval_transfer_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Load required modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/12.0

# Activate your Python environment (adjust path as needed)
# source /path/to/your/venv/bin/activate
# OR
# conda activate your_env_name

# Install requirements if needed
echo "Installing requirements..."
pip install -q -r requirements.txt

# Print Python and PyTorch info
echo ""
echo "=========================================="
echo "Environment Information"
echo "=========================================="
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""

# Run the evaluation script
echo "=========================================="
echo "Starting Transfer Learning Evaluation"
echo "=========================================="
python evaluate_pretrained_transfer.py \
    --model_dir resnet18_transfer_learning \
    --num_models 5 \
    --output pretrained_transfer_results.csv

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Evaluation failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Evaluation completed successfully!"
echo "Job finished at: $(date)"
echo "=========================================="