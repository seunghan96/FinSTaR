"""Generate F8/F9/F10 from existing Final_A raw data.

Instead of loading stock CSVs, extracts price sequences from Final_A's
existing QA samples (which already contain embedded price data).
This allows generating new tasks on any server, regardless of stock CSV availability.

Usage:
    python ideas/fin_tsr/generate_qa_3new_from_existing.py \
        --existing_dir data/final_A_raw \
        --output_dir data/final_E_new_tasks \
        --cap_per_task 3500 --test_cap 1000
"""

import argparse
import json
import os
import re
import random
import numpy as np
from collections import Counter, defaultdict


def extract_prices_from_question(question):
    """Extract price arrays from question text."""
    brackets = list(re.finditer(r'\[([\d\s,\.]+)\]', question))
    prices = []
    for br in brackets:
        try:
            vals = [float(x.strip()) for x in br.group(1).split(',') if x.strip()]
            if len(vals) >= 60:
                prices.append(vals)
        except:
            pass
    return prices


def extract_ticker(sample):
    """Extract ticker from sample."""
    ticker = sample.get('ticker', '')
    if '_' in ticker:
        return ticker.split('_')
    return [ticker]


def format_ts_list(prices, decimals=2):
    """Format price list as string."""
    return '[' + ', '.join(f'{p:.{decimals}f}' for p in prices) + ']'


def balance_classes(samples):
    """Balance classes by undersampling majority class."""
    by_class = defaultdict(list)
    for s in samples:
        by_class[s['answer']].append(s)
    min_count = min(len(v) for v in by_class.values())
    balanced = []
    for cls, items in by_class.items():
        random.shuffle(items)
        balanced.extend(items[:min_count])
    return balanced


# ══════════════════════════════════════════════════════════
# Build stock-level price database from existing QA
# ══════════════════════════════════════════════════════════

def build_price_database(existing_data):
    """Extract (ticker, price_sequence) pairs from existing F1/F5/F6 samples.
    These are single-stock tasks with 120-day windows."""
    db = defaultdict(list)  # ticker -> list of price arrays
    for d in existing_data:
        task = d.get('task', '')
        if task not in ('F1_drawdown', 'F5_volatility', 'F6_trend'):
            continue
        ticker = d.get('ticker', '')
        if not ticker:
            continue
        prices = extract_prices_from_question(d.get('question', ''))
        if prices and len(prices[0]) >= 100:
            db[ticker].append(np.array(prices[0]))
    return db


def build_pair_database(existing_data):
    """Extract (ticker_pair, price_seqs) from existing F2/F4 samples."""
    db = []
    for d in existing_data:
        task = d.get('task', '')
        if task not in ('F2_correlation', 'F4_momentum'):
            continue
        prices = extract_prices_from_question(d.get('question', ''))
        if len(prices) >= 2 and len(prices[0]) >= 100 and len(prices[1]) >= 100:
            tickers = extract_ticker(d)
            if len(tickers) >= 2:
                db.append({
                    'tickers': tickers[:2],
                    'prices': [np.array(prices[0]), np.array(prices[1])],
                })
    return db


# ══════════════════════════════════════════════════════════
# F8: Drawdown Recovery
# ══════════════════════════════════════════════════════════

def generate_f8_from_prices(price_db, n_samples, forward_window=20,
                             min_drawdown=0.05, seed=42):
    rng = random.Random(seed)
    tickers = list(price_db.keys())
    samples = []

    for _ in range(n_samples * 20):
        if len(samples) >= n_samples * 3:
            break
        ticker = rng.choice(tickers)
        all_seqs = price_db[ticker]
        prices = rng.choice(all_seqs)
        n = len(prices)

        if n < 80 + forward_window:
            continue

        # Use first portion as "observation", rest as "forward"
        obs_end = rng.randint(60, n - forward_window)
        seg = prices[:obs_end]
        fwd = prices[obs_end:obs_end + forward_window]

        if len(seg) < 60 or len(fwd) < 10:
            continue

        peak = float(np.max(seg))
        peak_idx = int(np.argmax(seg))
        current = float(seg[-1])
        dd = (peak - current) / peak

        if dd < min_drawdown:
            continue

        fwd_best = float(np.max(fwd))
        fwd_worst = float(np.min(fwd))

        recovery_target = peak * 0.97
        deepen_target = current * 0.98

        if fwd_best >= recovery_target:
            ans = 'A'
        elif fwd_worst <= deepen_target:
            ans = 'B'
        else:
            continue

        question = (
            f"You are analyzing the stock {ticker}.\n"
            f"Below are the daily closing prices for the most recent "
            f"{len(seg)} trading days:\n\n"
            f"{format_ts_list(seg)}\n\n"
            f"The stock has experienced a drawdown of {dd*100:.1f}% from its recent peak "
            f"(peak: {peak:.2f}, current: {current:.2f}).\n\n"
            f"Based on the price history, predict the stock's behavior "
            f"over the next {forward_window} trading days:\n\n"
            f"(A) Recovery — the price recovers toward the peak level\n"
            f"(B) Deepens — the drawdown deepens further\n"
        )

        samples.append({
            "task": "F8_drawdown_recovery",
            "question": question,
            "answer": ans,
            "ticker": ticker,
            "metadata": {
                "peak_price": round(peak, 2),
                "peak_day": peak_idx,
                "current_price": round(current, 2),
                "current_drawdown": round(dd, 4),
                "forward_best": round(fwd_best, 2),
                "forward_worst": round(fwd_worst, 2),
                "forward_window": forward_window,
                "window_len": len(seg),
            },
        })

    rng.shuffle(samples)
    return samples[:n_samples * 2]


# ══════════════════════════════════════════════════════════
# F9: Volatility Forecast
# ══════════════════════════════════════════════════════════

def generate_f9_from_prices(price_db, n_samples, recent=20,
                             forward_window=20, change_threshold=0.25, seed=42):
    rng = random.Random(seed)
    tickers = list(price_db.keys())
    samples = []

    for _ in range(n_samples * 20):
        if len(samples) >= n_samples * 3:
            break
        ticker = rng.choice(tickers)
        all_seqs = price_db[ticker]
        prices = rng.choice(all_seqs)
        n = len(prices)

        if n < 60 + forward_window:
            continue

        obs_end = rng.randint(40, n - forward_window)
        seg = prices[:obs_end]
        fwd = prices[obs_end:obs_end + forward_window]

        if len(seg) < 40 or len(fwd) < 10:
            continue

        returns = [(seg[i] - seg[i-1]) / seg[i-1] * 100 for i in range(1, len(seg))]
        if len(returns) < recent + 5:
            continue
        long_vol = float(np.std(returns))
        recent_vol = float(np.std(returns[-recent:]))
        if long_vol < 1e-8:
            continue
        current_ratio = recent_vol / long_vol

        fwd_returns = [(fwd[i] - fwd[i-1]) / fwd[i-1] * 100 for i in range(1, len(fwd))]
        if len(fwd_returns) < 5:
            continue
        fwd_vol = float(np.std(fwd_returns))
        fwd_ratio = fwd_vol / long_vol

        if fwd_ratio > current_ratio * (1 + change_threshold):
            ans = 'A'
        elif fwd_ratio < current_ratio * (1 - change_threshold):
            ans = 'B'
        else:
            continue

        question = (
            f"You are analyzing the stock {ticker}.\n"
            f"Below are the daily closing prices for the most recent "
            f"{len(seg)} trading days:\n\n"
            f"{format_ts_list(seg)}\n\n"
            f"The current volatility ratio (recent {recent}-day vs overall) "
            f"is {current_ratio:.2f}.\n\n"
            f"Based on the price history and volatility pattern, predict how "
            f"volatility will change over the next {forward_window} trading days:\n\n"
            f"(A) Volatility increases — the market becomes more volatile\n"
            f"(B) Volatility decreases — the market calms down\n"
        )

        samples.append({
            "task": "F9_volatility_forecast",
            "question": question,
            "answer": ans,
            "ticker": ticker,
            "metadata": {
                "recent_vol": round(recent_vol, 4),
                "long_vol": round(long_vol, 4),
                "current_ratio": round(current_ratio, 4),
                "forward_vol": round(fwd_vol, 4),
                "forward_ratio": round(fwd_ratio, 4),
                "recent_window": recent,
                "forward_window": forward_window,
                "window_len": len(seg),
            },
        })

    rng.shuffle(samples)
    return samples[:n_samples * 2]


# ══════════════════════════════════════════════════════════
# F10: Pair Convergence
# ══════════════════════════════════════════════════════════

def generate_f10_from_pairs(pair_db, n_samples, forward_window=20,
                             spread_margin=0.03, seed=42):
    rng = random.Random(seed)
    samples = []

    for _ in range(n_samples * 20):
        if len(samples) >= n_samples * 3:
            break
        pair = rng.choice(pair_db)
        tickers = pair['tickers']
        prices_a = pair['prices'][0]
        prices_b = pair['prices'][1]
        n = min(len(prices_a), len(prices_b))

        if n < 60 + forward_window:
            continue

        obs_end = rng.randint(40, n - forward_window)
        seg_a = prices_a[:obs_end]
        seg_b = prices_b[:obs_end]
        fwd_a = prices_a[obs_end:obs_end + forward_window]
        fwd_b = prices_b[obs_end:obs_end + forward_window]

        if len(fwd_a) < 10 or len(fwd_b) < 10:
            continue

        norm_a = [p / seg_a[0] for p in seg_a]
        norm_b = [p / seg_b[0] for p in seg_b]
        current_spread = abs(norm_a[-1] - norm_b[-1])

        fwd_norm_a = [p / seg_a[0] for p in fwd_a]
        fwd_norm_b = [p / seg_b[0] for p in fwd_b]
        fwd_spread = abs(fwd_norm_a[-1] - fwd_norm_b[-1])

        spread_change = fwd_spread - current_spread

        if abs(spread_change) < spread_margin:
            continue

        ans = 'A' if spread_change < -spread_margin else 'B'

        question = f"You are comparing two stocks to analyze their price relationship.\n\n"
        for i, (t, seg) in enumerate(zip(tickers, [seg_a, seg_b])):
            question += (
                f"Stock {chr(65+i)} ({t}) — daily closing prices ({len(seg)} days):\n"
                f"{format_ts_list(seg)}\n\n"
            )
        question += (
            f"The two stocks have shown a price spread (normalized difference) "
            f"of {current_spread:.3f} at the end of the observation period.\n\n"
            f"Based on the price histories, predict how the spread will change "
            f"over the next {forward_window} trading days:\n\n"
            f"(A) Convergence — the spread narrows (prices move closer together)\n"
            f"(B) Divergence — the spread widens (prices move further apart)\n"
        )

        samples.append({
            "task": "F10_pair_convergence",
            "question": question,
            "answer": ans,
            "ticker": f"{tickers[0]}_{tickers[1]}",
            "metadata": {
                "tickers": tickers,
                "current_spread": round(float(current_spread), 4),
                "forward_spread": round(float(fwd_spread), 4),
                "spread_change": round(float(spread_change), 4),
                "forward_window": forward_window,
                "window_len": len(seg_a),
            },
        })

    rng.shuffle(samples)
    return samples[:n_samples * 2]


def generate_for_split(existing_data, n_samples, seed, split_name, cap_per_task):
    print(f"\n=== {split_name} ({len(existing_data)} existing samples) ===")

    price_db = build_price_database(existing_data)
    pair_db = build_pair_database(existing_data)
    print(f"  Price DB: {len(price_db)} tickers, {sum(len(v) for v in price_db.values())} sequences")
    print(f"  Pair DB: {len(pair_db)} pairs")

    all_qa = []

    qa = generate_f8_from_prices(price_db, n_samples, seed=seed * 10 + 8)
    all_qa.extend(qa)
    print(f"  F8: {len(qa)} raw, classes={dict(Counter(d['answer'] for d in qa))}")

    qa = generate_f9_from_prices(price_db, n_samples, seed=seed * 10 + 9)
    all_qa.extend(qa)
    print(f"  F9: {len(qa)} raw, classes={dict(Counter(d['answer'] for d in qa))}")

    qa = generate_f10_from_pairs(pair_db, n_samples, seed=seed * 10 + 10)
    all_qa.extend(qa)
    print(f"  F10: {len(qa)} raw, classes={dict(Counter(d['answer'] for d in qa))}")

    by_task = defaultdict(list)
    for d in all_qa:
        by_task[d['task']].append(d)

    balanced_all = []
    task_num_map = {'F8_drawdown_recovery': 8, 'F9_volatility_forecast': 9, 'F10_pair_convergence': 10}
    for task, task_samples in sorted(by_task.items()):
        random.seed(seed * 10 + task_num_map.get(task, 0))
        balanced = balance_classes(task_samples)
        if cap_per_task and len(balanced) > cap_per_task:
            random.seed(seed * 10 + task_num_map.get(task, 0) + 100)
            balanced = random.sample(balanced, cap_per_task)
        balanced_all.extend(balanced)
        cls = Counter(d['answer'] for d in balanced)
        print(f"  {task}: balanced={len(balanced)}, classes={dict(sorted(cls.items()))}")

    random.seed(seed + 999)
    random.shuffle(balanced_all)

    for i, d in enumerate(balanced_all):
        d['question_id'] = f"{split_name}_{d['task']}_{i:05d}"
        d['domain'] = 'financial'
        d['system'] = 'You are a financial time series analyst.'

    print(f"  TOTAL: {len(balanced_all)}")
    return balanced_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing_dir', required=True, help='Path to final_A_raw')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--cap_per_task', type=int, default=3500)
    parser.add_argument('--test_cap', type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits = [
        ('train_sft.json', 42, args.cap_per_task),
        ('test_sft.json', 123, args.test_cap),
        ('test_b_ood_stock.json', 789, args.test_cap),
        ('test_c_ood_stock_period.json', 456, args.test_cap),
    ]

    for filename, seed, cap in splits:
        inpath = os.path.join(args.existing_dir, filename)
        if not os.path.exists(inpath):
            print(f"[WARN] {inpath} not found, skipping")
            continue
        existing = json.load(open(inpath))
        data = generate_for_split(existing, n_samples=50000, seed=seed,
                                  split_name=filename.replace('.json', ''),
                                  cap_per_task=cap)
        json.dump(data, open(os.path.join(args.output_dir, filename), 'w'),
                  indent=2, ensure_ascii=False)

    print(f"\nNew task data saved to {args.output_dir}")


if __name__ == '__main__':
    main()
