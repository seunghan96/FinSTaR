"""Generate QA with 10 tasks — extends v5 with 3 new prediction tasks.

New tasks:
  F8_drawdown_recovery: predict if drawdown recovers or deepens (single-stock prediction)
  F9_volatility_forecast: predict if volatility increases or decreases (single-stock prediction)
  F10_pair_convergence: predict if spread converges or diverges (multi-stock prediction)

These extend the 2x2 taxonomy by pairing each assessment task with a prediction task:
  Drawdown (F1) → Drawdown Recovery (F8)
  Volatility (F5) → Volatility Forecast (F9)
  Correlation (F2) → Pair Convergence (F10)
"""

import argparse
import json
import os
import sys
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'new_idea_v3'))
sys.path.insert(0, os.path.dirname(__file__))

from generate_qa_new_v3 import (
    generate_f1_drawdown, generate_f2_correlation,
    generate_f3_event_v2, balance_classes, format_ts_list
)
from generate_qa_f567 import generate_f5_volatility, generate_f6_trend, generate_f7_breakout
from generate_qa_7tasks_v2 import load_stocks_with_universe
from generate_qa_7tasks_v5 import (
    generate_f2_parameterized, generate_f4_pairwise,
    generate_f3_quantitative, generate_f5_parameterized,
)
from collections import Counter


# ══════════════════════════════════════════════════════════
# F8: Drawdown Recovery (2-class)
# ══════════════════════════════════════════════════════════

def generate_f8_drawdown_recovery(stock_data, n_samples, window=120,
                                   forward_window=20, min_drawdown=0.05,
                                   recovery_margin=0.03, deepen_margin=0.02,
                                   seed=42):
    """F8: After a drawdown, will the stock recover toward its peak or deepen?

    Picks windows where stock is in drawdown (>min_drawdown from peak).
    Looks forward to classify:
      A: Recovery — forward best price gets within recovery_margin of peak
      B: Deepens — forward worst price drops further than current by deepen_margin
    Ambiguous cases (neither clear recovery nor deepening) are discarded.
    """
    rng = random.Random(seed)
    tickers = list(stock_data.keys())
    samples = []

    for _ in range(n_samples * 15):
        if len(samples) >= n_samples * 3:
            break
        ticker = rng.choice(tickers)
        prices = stock_data[ticker]
        if len(prices) < window + forward_window:
            continue

        start = rng.randint(0, len(prices) - window - forward_window)
        seg = prices[start:start + window]
        fwd = prices[start + window:start + window + forward_window]

        peak = float(np.max(seg))
        peak_idx = int(np.argmax(seg))
        current = float(seg[-1])
        drawdown_pct = (peak - current) / peak

        if drawdown_pct < min_drawdown:
            continue

        fwd_best = float(np.max(fwd))
        fwd_worst = float(np.min(fwd))

        recovery_target = peak * (1 - recovery_margin)
        deepen_target = current * (1 - deepen_margin)

        if fwd_best >= recovery_target:
            ans = 'A'
        elif fwd_worst <= deepen_target:
            ans = 'B'
        else:
            continue

        dd_str = f"{drawdown_pct*100:.1f}"
        question = (
            f"You are analyzing the stock {ticker}.\n"
            f"Below are the daily closing prices for the most recent "
            f"{len(seg)} trading days:\n\n"
            f"{format_ts_list(seg)}\n\n"
            f"The stock has experienced a drawdown of {dd_str}% from its recent peak "
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
                "current_drawdown": round(drawdown_pct, 4),
                "forward_best": round(fwd_best, 2),
                "forward_worst": round(fwd_worst, 2),
                "forward_window": forward_window,
                "window_len": window,
            },
        })

    rng.shuffle(samples)
    return samples[:n_samples * 2]


# ══════════════════════════════════════════════════════════
# F9: Volatility Forecast (2-class)
# ══════════════════════════════════════════════════════════

def generate_f9_volatility_forecast(stock_data, n_samples, window=120,
                                     recent=20, forward_window=20,
                                     change_threshold=0.25, seed=42):
    """F9: Will recent volatility increase or decrease in the next period?

    Computes current vol ratio (recent/long), then checks forward vol ratio.
    A: Volatility increases (forward ratio > current ratio * (1 + change_threshold))
    B: Volatility decreases (forward ratio < current ratio * (1 - change_threshold))
    Ambiguous cases discarded.
    """
    rng = random.Random(seed)
    tickers = list(stock_data.keys())
    samples = []

    for _ in range(n_samples * 15):
        if len(samples) >= n_samples * 3:
            break
        ticker = rng.choice(tickers)
        prices = stock_data[ticker]
        if len(prices) < window + forward_window:
            continue

        start = rng.randint(0, len(prices) - window - forward_window)
        seg = prices[start:start + window]
        fwd = prices[start + window:start + window + forward_window]

        returns = [(seg[i] - seg[i-1]) / seg[i-1] * 100 for i in range(1, len(seg))]
        long_vol = float(np.std(returns))
        recent_returns = returns[-recent:]
        recent_vol = float(np.std(recent_returns))
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
            f"The current volatility ratio (recent {recent}-day vs overall {len(seg)}-day) "
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
                "window_len": window,
            },
        })

    rng.shuffle(samples)
    return samples[:n_samples * 2]


# ══════════════════════════════════════════════════════════
# F10: Pair Convergence (2-class)
# ══════════════════════════════════════════════════════════

def generate_f10_pair_convergence(stock_data, n_samples, window=120,
                                   forward_window=20, spread_margin=0.03,
                                   seed=42):
    """F10: Will the spread between two stocks converge or diverge?

    Computes normalized price spread (log ratio) over the window.
    Measures if the spread narrows (converges) or widens (diverges) in forward period.
    A: Converges — spread narrows
    B: Diverges — spread widens
    """
    rng = random.Random(seed)
    tickers = list(stock_data.keys())
    samples = []

    for _ in range(n_samples * 15):
        if len(samples) >= n_samples * 3:
            break
        chosen = rng.sample(tickers, 2)
        prices_list = [stock_data[t] for t in chosen]
        max_len = min(len(p) for p in prices_list)
        if max_len < window + forward_window:
            continue

        start = rng.randint(0, max_len - window - forward_window)
        segs = [p[start:start + window] for p in prices_list]
        fwds = [p[start + window:start + window + forward_window] for p in prices_list]

        # Normalize to start=1
        norm_a = [p / segs[0][0] for p in segs[0]]
        norm_b = [p / segs[1][0] for p in segs[1]]

        # Current spread = abs(normalized price diff) at end of window
        current_spread = abs(norm_a[-1] - norm_b[-1])

        # Forward spread
        fwd_norm_a = [p / segs[0][0] for p in fwds[0]]
        fwd_norm_b = [p / segs[1][0] for p in fwds[1]]
        fwd_spread = abs(fwd_norm_a[-1] - fwd_norm_b[-1])

        spread_change = fwd_spread - current_spread

        if abs(spread_change) < spread_margin:
            continue

        if spread_change < -spread_margin:
            ans = 'A'  # Converges
        else:
            ans = 'B'  # Diverges

        question = (
            f"You are comparing two stocks to analyze their price relationship.\n\n"
        )
        for i, (t, seg) in enumerate(zip(chosen, segs)):
            question += (
                f"Stock {chr(65+i)} ({t}) — daily closing prices ({window} days):\n"
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
            "ticker": f"{chosen[0]}_{chosen[1]}",
            "metadata": {
                "tickers": chosen,
                "current_spread": round(float(current_spread), 4),
                "forward_spread": round(float(fwd_spread), 4),
                "spread_change": round(float(spread_change), 4),
                "forward_window": forward_window,
                "window_len": window,
            },
        })

    rng.shuffle(samples)
    return samples[:n_samples * 2]


# ══════════════════════════════════════════════════════════
# Split generation (10 tasks)
# ══════════════════════════════════════════════════════════

def generate_split(stocks, n_samples, seed, split_name, cap_per_task=None,
                   f2_pos=0.30, f2_neg=-0.10, f2_uncorr=0.05,
                   f2_classes=3, f2_c_text=None,
                   f3_margin=0.0, f3_k=2.0,
                   f4_margin=5.0, f4_forward=20,
                   f5_classes=3, f5_low=0.6, f5_high=1.6,
                   f7_proximity=4.0):
    """Generate QA for all 10 tasks."""
    print(f"\n=== {split_name} ({len(stocks)} stocks, {n_samples} raw/task) ===")

    task_seeds = {f'F{i}': seed * 10 + i for i in range(1, 11)}

    all_qa = []
    generators = [
        # Original 7 tasks
        ('F1', generate_f1_drawdown, {'window': 120, 'seed': task_seeds['F1']}),
        ('F2', generate_f2_parameterized, {'window': 120, 'pos_thresh': f2_pos, 'neg_thresh': f2_neg, 'uncorr_thresh': f2_uncorr, 'f2_classes': f2_classes, 'f2_c_text': f2_c_text, 'seed': task_seeds['F2']}),
        ('F3', generate_f3_quantitative if f3_margin > 0 else generate_f3_event_v2,
         {'window': 120, 'post_window': 10, 'k': f3_k, 'seed': task_seeds['F3']}
         if f3_margin == 0 else
         {'window': 120, 'post_window': 10, 'k': f3_k, 'f3_margin': f3_margin, 'seed': task_seeds['F3']}),
        ('F4', generate_f4_pairwise, {'window': 120, 'forward_window': f4_forward, 'margin_pct': f4_margin, 'seed': task_seeds['F4']}),
        ('F5', generate_f5_parameterized, {'window': 120, 'recent': 20, 'f5_classes': f5_classes, 'f5_low': f5_low, 'f5_high': f5_high, 'seed': task_seeds['F5']}),
        ('F6', generate_f6_trend, {'window': 120, 'seed': task_seeds['F6']}),
        ('F7', generate_f7_breakout, {'window': 120, 'lookback': 60, 'proximity_pct': f7_proximity, 'forward_window': 10, 'seed': task_seeds['F7']}),
        # 3 new prediction tasks
        ('F8', generate_f8_drawdown_recovery, {'window': 120, 'forward_window': 20, 'seed': task_seeds['F8']}),
        ('F9', generate_f9_volatility_forecast, {'window': 120, 'forward_window': 20, 'seed': task_seeds['F9']}),
        ('F10', generate_f10_pair_convergence, {'window': 120, 'forward_window': 20, 'seed': task_seeds['F10']}),
    ]

    for name, gen_fn, kwargs in generators:
        qa = gen_fn(stock_data=stocks, n_samples=n_samples, **kwargs)
        all_qa.extend(qa)
        qa_cls = Counter(d['answer'] for d in qa)
        print(f"  {name}: {len(qa)} raw, classes={dict(sorted(qa_cls.items()))}")

    task_num_map = {
        'F1_drawdown': 1, 'F2_correlation': 2, 'F3_event': 3, 'F4_momentum': 4,
        'F5_volatility': 5, 'F6_trend': 6, 'F7_breakout': 7,
        'F8_drawdown_recovery': 8, 'F9_volatility_forecast': 9, 'F10_pair_convergence': 10,
    }

    by_task = {}
    for d in all_qa:
        by_task.setdefault(d['task'], []).append(d)

    balanced_all = []
    for task, task_samples in sorted(by_task.items()):
        task_num = task_num_map.get(task, 0)
        random.seed(seed * 10 + task_num)
        balanced = balance_classes(task_samples)
        if cap_per_task and len(balanced) > cap_per_task:
            random.seed(seed * 10 + task_num + 100)
            balanced = random.sample(balanced, cap_per_task)
        balanced_all.extend(balanced)
        cls = Counter(d['answer'] for d in balanced)
        print(f"  {task}: balanced={len(balanced)}, classes={dict(sorted(cls.items()))}")

    random.seed(seed)
    random.shuffle(balanced_all)

    for i, d in enumerate(balanced_all):
        d['question_id'] = f"{split_name}_{d['task']}_{i:05d}"
        d['domain'] = 'financial'
        d['system'] = 'You are a financial time series analyst.'

    tc = Counter(d['task'] for d in balanced_all)
    print(f"  TOTAL: {len(balanced_all)}")
    for t in sorted(tc):
        print(f"    {t}: {tc[t]}")
    return balanced_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--n_train_stocks', type=int, default=200)
    parser.add_argument('--samples_per_task', type=int, default=100000)
    parser.add_argument('--test_samples', type=int, default=10000)
    parser.add_argument('--cap_per_task', type=int, default=3500)
    parser.add_argument('--test_cap', type=int, default=1000)
    # F2-F7 params (same as v5)
    parser.add_argument('--f2_pos', type=float, default=0.30)
    parser.add_argument('--f2_neg', type=float, default=-0.10)
    parser.add_argument('--f2_uncorr', type=float, default=0.05)
    parser.add_argument('--f2_classes', type=int, default=3)
    parser.add_argument('--f2_c_text', type=str, default=None)
    parser.add_argument('--f3_margin', type=float, default=0.0)
    parser.add_argument('--f4_margin', type=float, default=5.0)
    parser.add_argument('--f4_forward', type=int, default=20)
    parser.add_argument('--f5_classes', type=int, default=3)
    parser.add_argument('--f5_low', type=float, default=0.6)
    parser.add_argument('--f5_high', type=float, default=1.6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_stocks, test_a_stocks, test_b_stocks, test_c_stocks = \
        load_stocks_with_universe(args.stock_dir, args.n_train_stocks)

    common = dict(f2_pos=args.f2_pos, f2_neg=args.f2_neg, f2_uncorr=args.f2_uncorr,
                  f2_classes=args.f2_classes, f2_c_text=args.f2_c_text,
                  f3_margin=args.f3_margin,
                  f4_margin=args.f4_margin, f4_forward=args.f4_forward,
                  f5_classes=args.f5_classes, f5_low=args.f5_low, f5_high=args.f5_high)

    train = generate_split(train_stocks, args.samples_per_task, seed=42,
                           split_name='train', cap_per_task=args.cap_per_task, **common)
    json.dump(train, open(os.path.join(args.output_dir, 'train_sft.json'), 'w'), indent=2, ensure_ascii=False)

    test_a = generate_split(test_a_stocks, args.test_samples, seed=123,
                            split_name='test_a', cap_per_task=args.test_cap, **common)
    json.dump(test_a, open(os.path.join(args.output_dir, 'test_sft.json'), 'w'), indent=2, ensure_ascii=False)

    test_b = generate_split(test_b_stocks, args.test_samples, seed=789,
                            split_name='test_b', cap_per_task=args.test_cap, **common)
    json.dump(test_b, open(os.path.join(args.output_dir, 'test_b_ood_stock.json'), 'w'), indent=2, ensure_ascii=False)

    test_c = generate_split(test_c_stocks, args.test_samples, seed=456,
                            split_name='test_c', cap_per_task=args.test_cap, **common)
    json.dump(test_c, open(os.path.join(args.output_dir, 'test_c_ood_stock_period.json'), 'w'), indent=2, ensure_ascii=False)

    json.dump({**vars(args), 'train_samples': len(train), 'test_a': len(test_a),
               'test_b': len(test_b), 'test_c': len(test_c)},
              open(os.path.join(args.output_dir, 'gen_stats.json'), 'w'), indent=2)

    print(f"\nSummary: Train={len(train)}, Test A={len(test_a)}, B={len(test_b)}, C={len(test_c)}")


if __name__ == '__main__':
    main()
