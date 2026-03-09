#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tgl
export LD_LIBRARY_PATH=/home/rl_gaming/miniconda3/envs/tgl/lib:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=122
export PYTHONNOUSERSITE=1

echo "Starting IR evaluation..."
CUDA_VISIBLE_DEVICES=7 python train_raw_llm.py -d IR -o test > test_ir_raw.log 2>&1

echo "Starting IS evaluation..."
CUDA_VISIBLE_DEVICES=7 python train_raw_llm.py -d IS -o test > test_is_raw.log 2>&1

echo "Evaluations completed."
