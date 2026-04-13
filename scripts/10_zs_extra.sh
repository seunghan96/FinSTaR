#!/bin/bash
# final_I: Zero-shot — Additional open-source models (10 tasks)
# Usage:
#   bash experiments_final/final_I/10_zs_extra.sh <GPU_ID> <MODEL_FILTER>
#
# MODEL_FILTER options:
#   llama       → Llama-3.1-8B-Instruct
#   mistral     → Mistral-7B-Instruct-v0.3
#   gemma       → Gemma-2-9B-it
#   phi         → Phi-3.5-mini-instruct
#   all         → all of the above
#
# Examples:
#   bash experiments_final/final_I/10_zs_extra.sh 0 llama
#   bash experiments_final/final_I/10_zs_extra.sh 0 gemma
#   bash experiments_final/final_I/10_zs_extra.sh 0 all
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"
source ideas/fin_tsr/smart_eval_auto.sh

GPU_MEM_UTIL=0.85
GPU_ID="${1:-0}"
MODEL_FILTER="${2:-all}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

FINAL="final_I"
R=32; LR=5e-5
FINAL_DATA="data/${FINAL}"
RESULT_BASE="results/${FINAL}_r${R}_lr${LR}"
TASKS="F1_drawdown F2_correlation F3_event F4_momentum F5_volatility F6_trend F7_breakout F8_drawdown_recovery F9_volatility_forecast F10_pair_convergence"

MODELS_7B=(
    "llama31_8b_zs:meta-llama/Llama-3.1-8B-Instruct"
    "mistral7b_zs:mistralai/Mistral-7B-Instruct-v0.3"
    "gemma2_9b_zs:google/gemma-2-9b-it"
    "phi35_zs:microsoft/Phi-3.5-mini-instruct"
)

MODELS=()
for m in "${MODELS_7B[@]}"; do
    IFS=':' read -r MNAME MID <<< "$m"
    if [ "$MODEL_FILTER" = "all" ] || [[ "$MNAME" == *"$MODEL_FILTER"* ]]; then
        MODELS+=("$m")
    fi
done

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "ERROR: No models matched '${MODEL_FILTER}'"
    echo "Available: llama, mistral, gemma, phi, all"
    exit 1
fi

echo "Using GPU: ${GPU_ID} | Models: ${#MODELS[@]} | Dataset: ${FINAL}"

TEST_SPLITS=(
    "test_sft.json:test_a"
    "test_b_ood_stock.json:test_b"
    "test_c_ood_stock_period.json:test_c"
)

for entry in "${TEST_SPLITS[@]}"; do
    IFS=':' read -r TEST_FILENAME TAG <<< "$entry"
    TEST_FILE="${FINAL_DATA}/${TEST_FILENAME}"
    [ ! -f "$TEST_FILE" ] && continue

    for model_entry in "${MODELS[@]}"; do
        IFS=':' read -r MNAME MID <<< "$model_entry"
        OUTPUT_DIR="${RESULT_BASE}/${MNAME}_${TAG}"
        echo ""
        echo "--- ${FINAL} / ${TAG} / ${MNAME} ---"
        smart_eval_auto "${MID}" "${TEST_FILE}" "${OUTPUT_DIR}" "${TASKS}" "${GPU_MEM_UTIL}"
    done
done

echo ""
echo "=== ${FINAL} Extra ZS Done ==="
