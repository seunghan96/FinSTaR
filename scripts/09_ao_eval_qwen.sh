#!/bin/bash
# final_I (10 tasks): AO Eval — Qwen, EP 1-8, Test A/B/C
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"
source ideas/fin_tsr/smart_eval.sh

MODEL="Qwen/Qwen2.5-7B-Instruct"
GPU_MEM_UTIL=0.85
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using GPU: ${GPU_ID}"

FINAL="final_I"
R=32; LR=5e-5
FINAL_DATA="data/${FINAL}"
CKPT="/home/seunghan.lee/nfs-fin/seunghan.lee/time-omni/ckpts/ckpts/${FINAL}_r${R}_lr${LR}/ao_qwen"
RESULT_BASE="results/${FINAL}_r${R}_lr${LR}"
TASKS="F1_drawdown F2_correlation F3_event F4_momentum F5_volatility F6_trend F7_breakout F8_drawdown_recovery F9_volatility_forecast F10_pair_convergence"

for EVAL_EP in 1 2 3 4; do
    LORA_DIR="${CKPT}/lora/epoch_${EVAL_EP}"
    MERGED_DIR="${CKPT}/merged_ep${EVAL_EP}"
    if [ -d "${LORA_DIR}" ] && [ ! -d "${MERGED_DIR}" ]; then
        echo "=== Merging Qwen AO epoch ${EVAL_EP} ==="
        python3 -c "import sys;sys.path.insert(0,'ideas/fin_tsr');from train_lora_v2 import merge_lora;merge_lora('${MODEL}','${LORA_DIR}','${MERGED_DIR}')"
    fi
    if [ -d "${MERGED_DIR}" ]; then
        for test_tag in "test_sft:test_a" "test_b_ood_stock:test_b" "test_c_ood_stock_period:test_c"; do
            IFS=':' read -r TF TAG <<< "$test_tag"
            TEST_FILE="${FINAL_DATA}/${TF}.json"
            [ ! -f "$TEST_FILE" ] && continue
            TMPDIR="${FINAL_DATA}/eval_${TAG}"
            mkdir -p "${TMPDIR}"; ln -sf "$(realpath "${TEST_FILE}")" "${TMPDIR}/test_sft.json"
            RES="${RESULT_BASE}/ao_qwen_ep${EVAL_EP}_${TAG}"
            smart_eval "${MERGED_DIR}" "${TMPDIR}" "${RES}" "${TASKS}" "${GPU_MEM_UTIL}"
        done
    else
        echo "[WARN] Qwen AO epoch ${EVAL_EP} not found"
    fi
done

echo ""
echo "=== ${FINAL} Qwen AO Eval Done ==="
python3 ideas/fin_tsr/show_results.py "${RESULT_BASE}"
