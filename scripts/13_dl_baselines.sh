#!/bin/bash
# final_I: DL forecasting baselines (prediction tasks, zero-shot)
# Chronos 1 (T5) and Chronos 2 (Bolt) + lightweight methods.
#
# Usage:
#   bash experiments_final/final_I/13_dl_baselines.sh <GPU_ID> [METHOD]
#
# METHOD options (all zero-shot, no training):
#   dlinear, nlinear, seasonal_naive       (CPU, instant)
#   chronos1_tiny, chronos1_small          (GPU, Chronos 1 / T5)
#   chronos2_tiny, chronos2_small          (GPU, Chronos 2 / Bolt)
#   all                                    (all of the above)
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

GPU_ID="${1:-0}"
METHOD="${2:-all}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

FINAL="final_I"
RESULT_BASE="results/${FINAL}_r32_lr5e-5/dl_baselines"

if [ "$METHOD" = "all" ]; then
    METHODS="dlinear nlinear seasonal_naive chronos1_tiny chronos1_small chronos2_tiny chronos2_small"
else
    METHODS="$METHOD"
fi

echo "=== DL Baselines for ${FINAL} ==="
echo "GPU: ${GPU_ID} | Methods: ${METHODS}"

for test_tag in "test_sft.json:test_a" "test_b_ood_stock.json:test_b" "test_c_ood_stock_period.json:test_c"; do
    IFS=':' read -r TEST_FILE TAG <<< "$test_tag"
    FULL_PATH="data/${FINAL}/${TEST_FILE}"
    [ ! -f "$FULL_PATH" ] && continue

    echo ""
    echo "--- ${TAG} ---"
    python3 ideas/fin_tsr/eval_dl_baselines.py \
        --test_file "${FULL_PATH}" \
        --output_dir "${RESULT_BASE}/${TAG}" \
        --methods ${METHODS} \
        --gpu_id ${GPU_ID}
done

echo ""
echo "=== Done ==="
