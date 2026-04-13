#!/bin/bash
# final_I (10 tasks = final_E + Uncertainty-Aware CoT for prediction tasks)
# Same raw data as final_E, but CoT uses prepare_idea5_raw_v9_scenario (uncertainty-aware)
# Step 1: Reuse final_E's new task data (or generate if not exists)
# Step 2: Merge with final_A data (reuse final_E_raw if exists)
# Step 3: Prepare Uncertainty-Aware CoT/AO training data (v6)
# Step 4: ZS evaluation
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_DIR"
source ideas/fin_tsr/smart_eval.sh

GPU_MEM_UTIL=0.85
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using GPU: ${GPU_ID}"

FINAL="final_I"
FINAL_A_RAW="data/final_A_raw"
# Reuse final_E's raw data (same questions, different CoT only)
NEW_TASKS_DIR="data/final_E_new_tasks"
RAW_DATA="data/final_E_raw"
FINAL_DATA="data/${FINAL}"
R=32; LR=5e-5
RESULT_BASE="results/${FINAL}_r${R}_lr${LR}"
TASKS="F1_drawdown F2_correlation F3_event F4_momentum F5_volatility F6_trend F7_breakout F8_drawdown_recovery F9_volatility_forecast F10_pair_convergence"

# ── Step 1: Generate F8/F9/F10 data from existing Final_A data ──
if [ ! -f "${NEW_TASKS_DIR}/test_sft.json" ]; then
    echo "=== Generating F8/F9/F10 from existing Final_A data ==="
    python3 ideas/fin_tsr/generate_qa_3new_from_existing.py \
        --existing_dir "${FINAL_A_RAW}" \
        --output_dir "${NEW_TASKS_DIR}" \
        --cap_per_task 3500 \
        --test_cap 1000
fi

# ── Step 2: Merge final_A raw + new tasks → final_I raw ──
if [ ! -f "${RAW_DATA}/test_sft.json" ]; then
    echo "=== Merging final_A + new tasks ==="
    mkdir -p "${RAW_DATA}"
    python3 -c "
import json, os
for split in ['train_sft.json', 'test_sft.json', 'test_b_ood_stock.json', 'test_c_ood_stock_period.json']:
    a = json.load(open('${FINAL_A_RAW}/' + split))
    new = json.load(open('${NEW_TASKS_DIR}/' + split))
    merged = a + new
    json.dump(merged, open('${RAW_DATA}/' + split, 'w'), indent=2, ensure_ascii=False)
    from collections import Counter
    tc = Counter(d['task'] for d in merged)
    print(f'{split}: {len(merged)} samples')
    for t in sorted(tc): print(f'  {t}: {tc[t]}')
"
fi

# ── Step 3: Prepare CoT/AO data ──
if [ ! -f "${FINAL_DATA}/test_sft.json" ]; then
    echo "=== Preparing data (10 tasks with CoT) ==="
    python3 ideas/fin_tsr/prepare_idea5_raw_v9_scenario.py \
        --raw_dir "${RAW_DATA}" --output_dir "${FINAL_DATA}"
fi

# ── Step 4: ZS Baselines ──
echo "=== ZS Baselines ==="
for test_tag in "test_sft:test_a" "test_b_ood_stock:test_b" "test_c_ood_stock_period:test_c"; do
    IFS=':' read -r TF TAG <<< "$test_tag"
    TEST_FILE="${FINAL_DATA}/${TF}.json"
    [ ! -f "$TEST_FILE" ] && continue
    TMPDIR="${FINAL_DATA}/eval_${TAG}"
    mkdir -p "${TMPDIR}"; ln -sf "$(realpath "${TEST_FILE}")" "${TMPDIR}/test_sft.json"

    for bm in "timeomni1_zs:anton-hugging/TimeOmni-1-7B" "qwen25_zs:Qwen/Qwen2.5-7B-Instruct"; do
        IFS=':' read -r BT BD <<< "$bm"
        DST="${RESULT_BASE}/${BT}_${TAG}"
        smart_eval "${BD}" "${TMPDIR}" "${DST}" "${TASKS}" "${GPU_MEM_UTIL}"
    done
done

echo ""
echo "=== ${FINAL} ZS Done ==="
