#!/bin/bash
# final_I (10 tasks): CoT Training — TimeOmni, EP=4
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

MODEL="anton-hugging/TimeOmni-1-7B"
LR=5e-5; EP=4; R=32; A=64; BS=1; GA=16; ML=4096; NG=2

FINAL="final_I"
FINAL_DATA="data/${FINAL}"
CKPT="/home/seunghan.lee/nfs-fin/seunghan.lee/time-omni/ckpts/ckpts/${FINAL}_r${R}_lr${LR}/cot"

if [ ! -f "${FINAL_DATA}/train_cot.json" ]; then
    echo "[ERROR] Data not found. Run 01_zs.sh first."
    exit 1
fi

if [ ! -f "${CKPT}/lora/adapter_config.json" ]; then
    echo "=== CoT Training (TimeOmni, 10 tasks) ==="
    accelerate launch --num_processes ${NG} --mixed_precision bf16 \
        ideas/fin_tsr/train_lora_v2.py \
        --model_dir "${MODEL}" --train_file "${FINAL_DATA}/train_cot.json" \
        --output_dir "${CKPT}/lora" \
        --lora_r ${R} --lora_alpha ${A} --batch_size ${BS} --grad_accum ${GA} \
        --max_length ${ML} --num_epochs ${EP} --lr ${LR} --class_weighted
else
    echo "[SKIP] TimeOmni CoT training already done"
fi
