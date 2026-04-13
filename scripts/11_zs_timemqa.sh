#!/bin/bash
# final_I: Zero-shot — TimeMQA models (merge + eval, 10 tasks)
# Usage:
#   bash experiments_final/final_I/11_zs_timemqa.sh <GPU_ID> <VARIANT>
#
# VARIANT options:
#   qwen        → TimeMQA-Qwen (base: Qwen/Qwen2.5-7B)
#   mistral     → TimeMQA-Mistral (base: Mistral-7B-v0.3)
#   llama       → TimeMQA-Llama (base: Meta-Llama-3-8B)
#   all         → all of the above
#
# Examples:
#   bash experiments_final/final_I/11_zs_timemqa.sh 0 qwen
#   bash experiments_final/final_I/11_zs_timemqa.sh 0 llama
#   bash experiments_final/final_I/11_zs_timemqa.sh 0 all
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"
source ideas/fin_tsr/smart_eval_auto.sh

GPU_MEM_UTIL=0.85
GPU_ID="${1:-0}"
VARIANT="${2:-all}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

FINAL="final_I"
R=32; LR=5e-5
FINAL_DATA="data/${FINAL}"
RESULT_BASE="results/${FINAL}_r${R}_lr${LR}"
TASKS="F1_drawdown F2_correlation F3_event F4_momentum F5_volatility F6_trend F7_breakout F8_drawdown_recovery F9_volatility_forecast F10_pair_convergence"
MERGED_BASE="models/timemqa"

declare -A VARIANTS
VARIANTS[qwen]="timemqa_qwen_zs"
VARIANTS[mistral]="timemqa_mistral_zs"
VARIANTS[llama]="timemqa_llama_zs"

if [ "$VARIANT" = "all" ]; then
    SELECTED=(qwen mistral llama)
else
    if [ -z "${VARIANTS[$VARIANT]+x}" ]; then
        echo "ERROR: Unknown variant '${VARIANT}'"
        echo "Available: qwen, mistral, llama, all"
        exit 1
    fi
    SELECTED=("$VARIANT")
fi

echo "Using GPU: ${GPU_ID} | Variants: ${SELECTED[*]} | Dataset: ${FINAL}"

# Step 1: Merge if needed
for v in "${SELECTED[@]}"; do
    MERGED_DIR="${MERGED_BASE}/timemqa_${v}"
    if [ -f "${MERGED_DIR}/config.json" ]; then
        echo "[SKIP] ${v} already merged"
    else
        echo "Merging ${v}..."
        python3 ideas/fin_tsr/merge_timemqa.py --variant "$v" --output_base "${MERGED_BASE}"
    fi
done

# Step 2: Eval
TEST_SPLITS=(
    "test_sft.json:test_a"
    "test_b_ood_stock.json:test_b"
    "test_c_ood_stock_period.json:test_c"
)

for entry in "${TEST_SPLITS[@]}"; do
    IFS=':' read -r TEST_FILENAME TAG <<< "$entry"
    TEST_FILE="${FINAL_DATA}/${TEST_FILENAME}"
    [ ! -f "$TEST_FILE" ] && continue

    for v in "${SELECTED[@]}"; do
        MERGED_DIR="${MERGED_BASE}/timemqa_${v}"
        [ ! -f "${MERGED_DIR}/config.json" ] && continue
        RESULT_PREFIX="${VARIANTS[$v]}"
        OUTPUT_DIR="${RESULT_BASE}/${RESULT_PREFIX}_${TAG}"
        echo ""
        echo "--- ${FINAL} / ${TAG} / TimeMQA-${v} ---"
        smart_eval_auto "${MERGED_DIR}" "${TEST_FILE}" "${OUTPUT_DIR}" "${TASKS}" "${GPU_MEM_UTIL}"
    done
done

echo ""
echo "=== ${FINAL} TimeMQA ZS Done ==="
