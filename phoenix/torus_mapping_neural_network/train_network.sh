#!/bin/bash


echo ">>> Activating Conda environment 'phoenix'..."
# Initialize conda for the script (required for 'conda activate' in bash scripts)
eval "$(conda shell.bash hook)"
conda activate phoenix

echo ">>> Running setup_neural_network_multi_gpu.py..."
python setup_neural_network_multi_gpu.py

echo ">>> Pipeline finished successfully!"