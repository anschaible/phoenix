#!/bin/bash

echo ">>> Script started at: $(date)"

echo ">>> Scanning for the 4 GPUs with the most free VRAM..."
# 1. Query nvidia-smi for free memory and GPU index
# 2. Sort numerically in descending order (highest free RAM at the top)
# 3. Take the top 4 lines
# 4. Extract just the GPU index
# 5. Join them with commas (e.g., "1,4,5,7")
FREE_GPUS=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -n 4 | awk '{print $2}' | paste -sd ",")

echo ">>> Reserving GPUs: $FREE_GPUS"

# Tell JAX and CUDA to ONLY look at these specific GPUs
export CUDA_VISIBLE_DEVICES=$FREE_GPUS

echo ">>> Activating Conda environment 'phoenix'..."
# Initialize conda for the script
eval "$(conda shell.bash hook)"
conda activate phoenix

echo ">>> Running setup_neural_network_multi_gpu.py..."
python setup_neural_network_multi_gpu.py 

echo ">>> Pipeline finished successfully!"
echo ">>> Script ended at: $(date)"