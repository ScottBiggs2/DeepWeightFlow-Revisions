#!/bin/bash
#SBATCH --job-name=resnet18_cifar_training
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job information
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Load required modules (adjust based on your cluster)
# module load python/3.9
# module load cuda/12.0

# Activate your Python environment (adjust path as needed)
# source /path/to/your/venv/bin/activate
# OR
# conda activate your_env_name

# Install requirements
echo "================================"
echo "Installing requirements..."
echo "================================"
pip install -r requirements.txt
INSTALL_EXIT_CODE=$?

if [ $INSTALL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Failed to install requirements with exit code $INSTALL_EXIT_CODE"
    exit $INSTALL_EXIT_CODE
fi

# Print Python and PyTorch info
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

echo "================================"
echo "Starting ResNet18 training..."
echo "================================"
python generateModelsCode/generateResnet18.py
RESNET_EXIT_CODE=$?

if [ $RESNET_EXIT_CODE -ne 0 ]; then
    echo "ERROR: ResNet18 training failed with exit code $RESNET_EXIT_CODE"
    exit $RESNET_EXIT_CODE
fi

echo "================================"
echo "ResNet18 training completed!"
echo "Job finished at: $(date)"
echo "================================"