#!/bin/bash
# final_I: Statistical/ML baselines for prediction tasks
# No GPU needed. Runs traditional forecasting → classification.
#
# Methods: last_value, linear_trend, moving_avg, exp_smoothing, drift, momentum
# Optional (if statsmodels installed): arima, ets
#
# Usage:
#   bash experiments_final/final_I/12_statistical_baselines.sh
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"

FINAL="final_I"
RESULT_BASE="results/${FINAL}_r32_lr5e-5/statistical_baselines"

# Base methods (numpy only, no extra packages)
METHODS="last_value linear_trend moving_avg exp_smoothing drift momentum"

# Add statsmodels methods if available
python3 -c "import statsmodels" 2>/dev/null && METHODS="$METHODS arima ets" && echo "statsmodels found: adding ARIMA, ETS"

echo "=== Statistical Baselines for ${FINAL} ==="
echo "Methods: ${METHODS}"
echo ""

# Run on all 3 test splits
for test_tag in "test_sft.json:test_a" "test_b_ood_stock.json:test_b" "test_c_ood_stock_period.json:test_c"; do
    IFS=':' read -r TEST_FILE TAG <<< "$test_tag"
    FULL_PATH="data/${FINAL}/${TEST_FILE}"
    [ ! -f "$FULL_PATH" ] && continue

    echo "--- ${TAG} ---"
    python3 ideas/fin_tsr/eval_statistical_baselines.py \
        --test_file "${FULL_PATH}" \
        --output_dir "${RESULT_BASE}/${TAG}" \
        --methods ${METHODS}
    echo ""
done

echo "=== Done ==="
