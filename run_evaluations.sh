#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tgl

export CUDA_VISIBLE_DEVICES=4

echo "Testing TGL-LLM models" > evaluation_results.txt
echo "=====================" >> evaluation_results.txt

for ds in IR IS EG; do
    for k in 3 5 9; do
        if [ -d "./checkpoints/LLM/${ds}/model_final_${k}" ]; then
            echo "Running test: Dataset=${ds}, K=${k}" | tee -a evaluation_results.txt
            python3 train_llm.py -d ${ds} -o test -k ${k} -p ./checkpoints/LLM/${ds}/model_final_${k} 2>&1 | tee -a evaluation_results.txt
            echo "----------------------------------------" | tee -a evaluation_results.txt
        else
            echo "Checkpoint not found: ${ds} K=${k}" | tee -a evaluation_results.txt
        fi
    done
done
