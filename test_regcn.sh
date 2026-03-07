#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tgl

export CUDA_VISIBLE_DEVICES=4

echo "Testing REGCN Models" > regcn_results.txt
echo "=====================" >> regcn_results.txt

for ds in IR IS EG; do
    echo "Running test for Dataset=${ds}" | tee -a regcn_results.txt
    python3 train.py -d ${ds} -m REGCN -g "4" 2>&1 | tee -a regcn_results.txt
    echo "----------------------------------------" | tee -a regcn_results.txt
done
