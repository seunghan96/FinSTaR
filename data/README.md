# FinTSR-Bench Data

## Download

Download the pre-built benchmark from HuggingFace:

```bash
huggingface-cli download seunghanlee/FinTSR-Bench --local-dir .
```

## Files

| File | Samples | Description |
|---|---|---|
| `train_cot.json` | 35,000 | Training data with CoT annotations (3,500 per task) |
| `train_ao.json` | 35,000 | Training data, answer-only (no CoT) |
| `test_sft.json` | 10,000 | Test A: ID stocks, OOD period (2023-2025) |
| `test_b_ood_stock.json` | 10,000 | Test B: OOD stocks (50 held-out), ID period |
| `test_c_ood_stock_period.json` | 10,000 | Test C: OOD stocks + OOD period |

## Tasks (10 total)

### Assessment (4 tasks)
- **F1_drawdown** (4-class): Classify peak-to-trough decline severity
- **F5_volatility** (3-class): Classify recent vs. overall volatility ratio
- **F6_trend** (5-class): Classify 120-day cumulative return regime
- **F2_correlation** (3-class): Classify return correlation of two stocks

### Prediction (6 tasks)
- **F3_event** (2-class): Mean-reversion or persistence after shock
- **F7_breakout** (2-class): Breakout or bounce near key level
- **F8_drawdown_recovery** (2-class): Recovery or further decline
- **F9_volatility_forecast** (2-class): Volatility increase or decrease
- **F4_momentum** (2-class): Which stock outperforms
- **F10_pair_convergence** (2-class): Spread converges or diverges

## Generate from Scratch

```bash
# Step 1: Generate QA pairs from raw stock data
python src/data_generation/generate_qa_10tasks.py \
    --data_dir raw_stock_data/ \
    --output_dir data/raw/ \
    --samples_per_task 3500

# Step 2: Generate CoT annotations
python src/data_generation/prepare_final_data.py \
    --input_dir data/raw/ \
    --output_dir data/
```
