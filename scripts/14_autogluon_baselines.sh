#!/bin/bash
# final_I: AutoGluon DL forecasting baselines (prediction tasks)
# Includes: DeepAR, PatchTST, TFT, TiDE, Chronos (via AutoGluon)
#
# Usage:
#   bash experiments_final/final_I/14_autogluon_baselines.sh <GPU_ID> [METHOD]
#
# Examples:
#   bash experiments_final/final_I/14_autogluon_baselines.sh 0                # All methods
#   bash experiments_final/final_I/14_autogluon_baselines.sh 0 deepar         # DeepAR only
#   bash experiments_final/final_I/14_autogluon_baselines.sh 0 patchtst       # PatchTST only
#   bash experiments_final/final_I/14_autogluon_baselines.sh 0 chronos_ag     # Chronos via AG
#
# METHOD options:
#   chronos_ag  (zero-shot, fast)
#   deepar      (global, train on corpus)
#   patchtst    (global, train on corpus)
#   tft         (global, train on corpus)
#   tide        (global, train on corpus)
#   all         (all of the above)
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

# ── Auto-setup isolated virtualenv for AutoGluon (per-server, in /tmp) ──
AG_VENV="/tmp/venv_ag_$(whoami)"
PYTHON="${AG_VENV}/bin/python3"

if ! ${PYTHON} -c "import autogluon.timeseries" 2>/dev/null; then
    echo "=== Setting up AutoGluon environment at ${AG_VENV} ==="

    # Install virtualenv if not available
    python3 -m virtualenv --version 2>/dev/null || {
        echo "  Installing virtualenv..."
        pip install -q virtualenv
    }

    # Create clean venv (always fresh to avoid path issues)
    rm -rf "${AG_VENV}"
    echo "  Creating virtualenv..."
    python3 -m virtualenv "${AG_VENV}" --python=python3

    # Install AutoGluon inside venv
    echo "  Installing autogluon.timeseries (this may take a few minutes)..."
    "${AG_VENV}/bin/pip" install -q autogluon.timeseries
    echo "  Done!"
fi

${PYTHON} -c "import autogluon.timeseries; print('AutoGluon OK')"

GPU_ID="${1:-0}"
METHOD="${2:-all}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

FINAL="final_I"
RESULT_BASE="results/${FINAL}_r32_lr5e-5/ag_baselines"
TRAIN_FILE="data/${FINAL}/train_sft.json"

if [ "$METHOD" = "all" ]; then
    METHODS="chronos_ag deepar patchtst tft tide"
else
    METHODS="$METHOD"
fi

echo "=== AutoGluon Baselines for ${FINAL} ==="
echo "GPU: ${GPU_ID} | Methods: ${METHODS}"

for test_tag in "test_sft.json:test_a" "test_b_ood_stock.json:test_b" "test_c_ood_stock_period.json:test_c"; do
    IFS=':' read -r TEST_FILE TAG <<< "$test_tag"
    FULL_PATH="data/${FINAL}/${TEST_FILE}"
    [ ! -f "$FULL_PATH" ] && continue

    echo ""
    echo "--- ${TAG} ---"
    ${PYTHON} ideas/fin_tsr/eval_autogluon_baselines.py \
        --test_file "${FULL_PATH}" \
        --train_file "${TRAIN_FILE}" \
        --output_dir "${RESULT_BASE}/${TAG}" \
        --methods ${METHODS} \
        --gpu_id 0 \
        --max_train_series 2000 \
        --time_limit 600
done

echo ""
echo "=== Done ==="
