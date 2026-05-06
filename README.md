# FinSTaR: Towards Financial Reasoning with Time Series Reasoning Models

<p align="center">
  <a href="https://arxiv.org/pdf/2605.03460"><img src="https://img.shields.io/badge/arXiv-2605.03460-b31b1b.svg" alt="arXiv"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

<p align="center">
  <img src="assets/overview.png" width="85%">
</p>

**FinSTaR** (**Fin**ancial Time **S**eries **T**hinking **a**nd **R**easoning) is the first Time Series Reasoning Model (TSRM) designed specifically for the financial domain. It employs two structurally different chain-of-thought (CoT) strategies tailored to the epistemological properties of financial reasoning:

- **Compute-in-CoT** for *assessment* tasks (deterministic, computable from observable prices)
- **Scenario-Aware CoT** for *prediction* tasks (probabilistic, subject to unobservable factors)

FinSTaR achieves **78.9%** overall accuracy on FinTSR-Bench, outperforming 15+ baselines spanning LLMs, TSRMs, and TS forecasting models.

<br>

---

## Overview

### Four Capability Categories

We define core capabilities of a Financial TSRM along two axes, forming a 2x2 taxonomy:

|  | **Single-Stock** | **Multi-Stock** |
|---|---|---|
| **Assessment** | Drawdown, Volatility Regime, Trend Direction | Correlation |
| **Prediction** | Event Response, Support/Resistance, Drawdown Recovery, Volatility Forecast | Relative Performance, Pair Convergence |

### Key Results

| Model | Assessment Avg. | Prediction Avg. | Overall |
|---|---|---|---|
| Qwen2.5-7B (zero-shot) | 53.5 | 50.3 | 51.6 |
| TimeOmni-1-7B (zero-shot) | 48.3 | 53.2 | 51.3 |
| Qwen2.5-7B (SFT w/ CoT) | 67.7 | 51.1 | 57.8 |
| **FinSTaR (Ours)** | **95.0** | **68.2** | **78.9** |

<br>

---

## Installation

```bash
git clone https://github.com/seunghan96/FinSTaR.git
cd FinSTaR
pip install -r requirements.txt
```

<br>

---

## FinTSR-Bench

### Data Generation

FinTSR-Bench is constructed from 250 S&P 500 stocks (2010-2025). To generate the benchmark from raw stock data:

```bash
# Step 1: Generate QA pairs (10 tasks, ~3,500 samples each)
python src/data_generation/generate_qa_10tasks.py \
    --data_dir raw_stock_data/ \
    --output_dir data/raw/ \
    --samples_per_task 3500

# Step 2: Generate CoT annotations and prepare final data
python src/data_generation/prepare_final_data.py \
    --input_dir data/raw/ \
    --output_dir data/
```

### Data Structure

```
data/
├── train_cot.json          # 35K samples with CoT annotations
├── train_ao.json           # 35K samples (answer-only)
├── test_sft.json           # Test A: ID stocks, OOD period (10K)
├── test_b_ood_stock.json   # Test B: OOD stocks, ID period (10K)
└── test_c_ood_stock_period.json  # Test C: OOD stocks + period (10K)
```

### Data Format

Each sample follows the chat format:
```json
{
  "task": "F1_drawdown",
  "answer": "C",
  "conversations": [
    {"role": "user", "content": "You are analyzing the stock AAPL. Below are the daily closing prices (120 days): [...]"},
    {"role": "assistant", "content": "<think>\nStep 1 — Find the peak price: ...\n</think>\n<answer>(C)</answer>"}
  ],
  "metadata": {"peak": 182.63, "current": 161.42, "drawdown": 0.116}
}
```

<br>

---

## Training

### Quick Start

```bash
# Train FinSTaR (TimeOmni-1-7B backbone, LoRA, 4 epochs)
accelerate launch --num_processes 2 --mixed_precision bf16 \
    src/training/train.py \
    --model_dir anton-hugging/TimeOmni-1-7B \
    --train_file data/train_cot.json \
    --output_dir checkpoints/finstar \
    --lora_r 32 --lora_alpha 64 \
    --batch_size 1 --grad_accum 16 \
    --max_length 4096 --num_epochs 4 --lr 5e-5
```

### Training Configurations

| Config | Backbone | CoT | Description |
|---|---|---|---|
| `02_cot_train_timeomni.sh` | TimeOmni-1-7B | Compute + Scenario | **FinSTaR** (main) |
| `03_cot_train_qwen.sh` | Qwen2.5-7B | Compute + Scenario | SFT baseline (w/ CoT) |
| `04_ao_train_timeomni.sh` | TimeOmni-1-7B | None (answer-only) | Ablation (w/o CoT) |
| `05_ao_train_qwen.sh` | Qwen2.5-7B | None (answer-only) | SFT baseline (w/o CoT) |

<br>

---

## Evaluation

### Zero-Shot Evaluation

```bash
# Evaluate any model zero-shot on FinTSR-Bench
python src/evaluation/inference.py \
    --model_dir anton-hugging/TimeOmni-1-7B \
    --test_file data/test_sft.json \
    --output_dir results/timeomni_zs_test_a
```

### FinSTaR Evaluation

```bash
# Evaluate FinSTaR (LoRA adapter)
python src/evaluation/inference.py \
    --model_dir anton-hugging/TimeOmni-1-7B \
    --lora_dir checkpoints/finstar/lora \
    --test_file data/test_sft.json \
    --output_dir results/finstar_test_a
```

### Forecasting Baselines

```bash
# Statistical baselines (Last Value, MA, ETS, Drift, Momentum)
bash scripts/12_statistical_baselines.sh

# Deep learning baselines (PatchTST, DLinear, Chronos, etc.)
bash scripts/13_dl_baselines.sh
```

<br>

---

## Project Structure

```
FinSTaR/
├── README.md
├── requirements.txt
├── configs/
│   └── accelerate_config.yaml      # Multi-GPU training config
├── src/
│   ├── data_generation/             # FinTSR-Bench construction
│   │   ├── generate_qa_10tasks.py   # QA pair generation (10 tasks)
│   │   ├── generate_cot.py          # Compute-in-CoT annotation
│   │   ├── generate_compute_cot.py  # Extended CoT with computation
│   │   ├── prepare_final_data.py    # Final data preparation
│   │   ├── prepare_fair_data.py     # Fair evaluation data
│   │   └── utils.py                 # Financial indicators & utilities
│   ├── training/
│   │   ├── train.py                 # LoRA SFT training
│   │   ├── data_utils.py            # Dataset & prompt utilities
│   │   └── train_utils.py           # Model loading & LoRA config
│   └── evaluation/
│       ├── inference.py             # Batch inference (vLLM)
│       ├── get_score.py             # Metric computation
│       └── utils.py                 # Evaluation helpers
├── scripts/                         # Experiment shell scripts
│   ├── 01_zs.sh                     # Zero-shot baselines
│   ├── 02_cot_train_timeomni.sh     # FinSTaR training
│   ├── 06_cot_eval_timeomni.sh      # FinSTaR evaluation
│   └── ...
└── data/                            # FinTSR-Bench (generate via src/data_generation/)
```

<br>

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{lee2026finstar,
  title={FinSTaR: Towards Financial Reasoning with Time Series Reasoning Models},
  author={Lee, Seunghan and Seo, Jun and Lee, Jaehoon and Yoo, Sungdong and Kim, Minjae and Lim, Tae Yoon and Kang, Dongwan and Choi, Hwanil and Lee, SoonYoung and Ahn, Wonbin},
  journal={arXiv preprint arXiv:2605.03460},
  year={2026}
}
```


## Acknowledgements

FinSTaR builds upon [TimeOmni-1](https://arxiv.org/abs/2509.24803) as its backbone. 
We thank the TimeOmni team for releasing model weights. Stock price data is sourced from publicly available S&P 500 historical data.

