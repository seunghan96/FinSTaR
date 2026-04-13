"""Final data preparation: task-specific input + adaptive depth CoT.

Fixes:
1. F1 oversampling (2x) to prevent CoT length collapse
2. Consistent task-specific representation for both train and test
3. Adaptive depth: F1/F2/F4 = full compute CoT, F3 = minimal 1-sentence

Usage:
    python ideas/fin_tsr/prepare_final_data.py \
        --input_dir data/fin_tsr_new_v3 \
        --output_dir data/fin_tsr_final
"""

import argparse
import json
import os
import sys
import re
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from convert_task_representations import convert_v12  # task-specific
from convert_to_returns import convert_question as convert_returns
from generate_compute_cot import generate_f1_cot, generate_f2_cot, generate_f4_cot


def generate_f2_input_aligned_cot(sample):
    """F2: Full-length CoT that references the paired direction count from INPUT.

    The issue: old CoT computes "correlation = 0.253" but the input shows
    "same direction: 78/119 (66%)". The model can't compute correlation at
    inference time, so it hallucinates. This CoT references what's actually
    in the input.
    """
    m = sample['metadata']
    corr = m.get('correlation', 0)
    threshold = m.get('threshold', 0.15)
    answer = sample['answer']

    # Compute same-direction count (same as what's in the input)
    q = sample.get('question', '')
    import re
    match = re.search(r'same direction[:\s]+(\d+)/(\d+)\s+\((\d+)%\)', q, re.IGNORECASE)
    if match:
        same = int(match.group(1))
        total = int(match.group(2))
        pct = int(match.group(3))
    else:
        # Fallback: estimate from correlation
        same = int(60 + corr * 30)
        total = 119
        pct = int(same / total * 100)

    if answer == 'A':
        assessment = (
            f"A same-direction rate of {pct}% is well above the 50% expected by chance, "
            f"indicating a consistent positive co-movement pattern between the two stocks."
        )
        conclusion = "positive correlation"
    elif answer == 'B':
        opp_pct = 100 - pct
        assessment = (
            f"A same-direction rate of only {pct}% means the stocks moved in opposite directions "
            f"{opp_pct}% of the time, indicating a negative relationship."
        )
        conclusion = "negative correlation"
    else:
        assessment = (
            f"A same-direction rate of {pct}% is close to the 50% expected by chance, "
            f"suggesting no systematic relationship between the two stocks' movements."
        )
        conclusion = "no significant correlation"

    return (
        f"<think>\n"
        f"Step 1 — Observe co-movement pattern:\n"
        f"The paired daily return comparison shows the stocks moving in the same direction "
        f"on {same} out of {total} trading days ({pct}%).\n\n"
        f"Step 2 — Assess significance:\n"
        f"{assessment}\n\n"
        f"Step 3 — Classify:\n"
        f"This indicates {conclusion} → ({answer}).\n"
        f"</think>\n"
        f"<answer>({answer})</answer>"
    )


def generate_f3_minimal_cot(sample):
    """F3: 1-sentence minimal CoT."""
    m = sample['metadata']
    event_dir = m.get('event_direction', 'unknown')
    event_z = m.get('event_z', 0)
    pre_trend = m.get('pre_trend', 'unknown')
    answer = sample['answer']

    if answer == 'A':
        reason = f"The {event_dir} shock (z={abs(event_z):.1f}) suggests mean-reversion given the {pre_trend} pre-event trend."
    else:
        reason = f"The {event_dir} shock (z={abs(event_z):.1f}) suggests persistence given the {pre_trend} pre-event trend."

    return f"<think>\n{reason}\n</think>\n<answer>({answer})</answer>"


def generate_f3_compute_cot(sample):
    """F3: Compute CoT from actual price data (requires prices in question).

    Extracts prices from question, computes pre-event volatility, event magnitude,
    and pre-event trend — all from observable data only (no future info).
    """
    m = sample['metadata']
    event_dir = m.get('event_direction', 'unknown')
    answer = sample['answer']

    q = sample.get('question', '')
    bs, be = q.find('['), q.find(']')
    if bs < 0 or be < 0:
        # Fallback to minimal if no prices
        return generate_f3_minimal_cot(sample)

    try:
        prices = [float(x.strip()) for x in q[bs+1:be].split(',') if x.strip()]
    except:
        return generate_f3_minimal_cot(sample)

    if len(prices) < 30:
        return generate_f3_minimal_cot(sample)

    # Compute from observable data
    returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    event_return = returns[-1]
    event_return_pct = event_return * 100

    # Pre-event volatility (20-day before event)
    pre_returns = returns[max(0, len(returns)-21):-1]
    pre_vol = np.std(pre_returns) * 100 if pre_returns else 0

    # Pre-event trend (60-day)
    lookback = min(60, len(prices) - 1)
    pre_ret = (prices[-2] - prices[-1 - lookback]) / prices[-1 - lookback] * 100

    # Z-score from pre-event stats
    pre_mu = np.mean(pre_returns) if pre_returns else 0
    pre_sigma = np.std(pre_returns) if pre_returns else 1e-8
    z_score = abs((event_return - pre_mu) / pre_sigma) if pre_sigma > 1e-8 else 0

    # Event day price change
    peak_pre = max(prices[:-1])
    last_price = prices[-1]
    prev_price = prices[-2]

    if answer == 'A':
        prediction = "mean-reversion — the price will likely move back toward pre-event levels"
        reasoning = (
            f"The extreme {event_dir} move (z={z_score:.1f}) disrupted a "
            f"{'stable' if pre_vol < 1.5 else 'volatile'} market (daily vol={pre_vol:.2f}%). "
            f"Such outsized shocks tend to partially reverse as the market corrects the overreaction."
        )
    else:
        prediction = "persistence — the price will likely continue in the shock direction"
        reasoning = (
            f"The {event_dir} shock (z={z_score:.1f}) aligns with the "
            f"{'existing ' + ('upward' if pre_ret > 0 else 'downward') + ' trend' if abs(pre_ret) > 2 else 'recent price action'}. "
            f"Pre-event volatility ({pre_vol:.2f}%) suggests the market "
            f"{'was already adjusting' if pre_vol > 2.0 else 'may continue repricing'}."
        )

    return (
        f"<think>\n"
        f"Step 1 — Measure the shock:\n"
        f"Last price: {last_price:.2f} (prev: {prev_price:.2f}), "
        f"event return: {event_return_pct:+.2f}%, z-score: {z_score:.1f}\n\n"
        f"Step 2 — Pre-event context:\n"
        f"60-day trend: {pre_ret:+.1f}%, daily volatility: {pre_vol:.2f}%\n\n"
        f"Step 3 — Prediction:\n"
        f"{reasoning}\n"
        f"This suggests {prediction} → ({answer}).\n"
        f"</think>\n"
        f"<answer>({answer})</answer>"
    )


def generate_f4_input_aligned_cot(sample):
    """F4: Full-length CoT that references performance comparison from INPUT."""
    m = sample['metadata']
    tickers = m.get('tickers', ['A', 'B', 'C'])
    winner_ticker = m.get('winner_ticker', tickers[0])
    winner_idx = m.get('winner_idx', 0)
    answer = sample['answer']

    # Extract performance data from input
    q = sample.get('question', '')
    import re

    # Try to find "120-day returns: AMZN=+12.3%, ..."
    ret_match = re.search(r'120-day returns?:\s*(.+)', q)
    perf_line = ret_match.group(1).strip() if ret_match else f"{', '.join(tickers)}"

    # Try to find "Winner: AAPL ..."
    winner_match = re.search(r'Winner:\s*(.+)', q)
    winner_line = winner_match.group(1).strip() if winner_match else f"{winner_ticker}"

    # Try to find "Recent 10-day: ..."
    recent_match = re.search(r'Recent \d+-day:\s*(.+)', q)
    recent_line = recent_match.group(1).strip() if recent_match else ""

    if answer == 'A':
        assessment = (
            f"The momentum winner ({winner_ticker}) has shown strong recent performance. "
            f"Historical momentum tends to persist in the short term, "
            f"suggesting the winner will continue to outperform the portfolio average."
        )
        conclusion = "momentum continues"
    else:
        assessment = (
            f"Despite {winner_ticker}'s strong past momentum, "
            f"{'recent deceleration suggests' if recent_line else 'the magnitude of outperformance suggests'} "
            f"the momentum may be exhausting. Mean-reversion effects could lead to underperformance."
        )
        conclusion = "momentum reverses"

    return (
        f"<think>\n"
        f"Step 1 — Identify the momentum winner:\n"
        f"Performance: {perf_line}\n"
        f"Winner: {winner_line}\n\n"
        f"Step 2 — Assess momentum sustainability:\n"
        f"{f'Recent trend: {recent_line}' if recent_line else 'Examining recent price action.'}\n"
        f"{assessment}\n\n"
        f"Step 3 — Prediction:\n"
        f"Based on the momentum analysis, {conclusion} → ({answer}).\n"
        f"</think>\n"
        f"<answer>({answer})</answer>"
    )


def apply_task_specific_repr(sample):
    """Apply best input representation per task."""
    task = sample['task']
    if task == 'F1_drawdown':
        # Returns + running max + summary
        return convert_returns(sample['question'], task)
    elif task == 'F2_correlation':
        # Paired direction comparison
        return convert_v12(sample)
    elif task == 'F3_event':
        # Keep original (task-specific context from question)
        return sample['question']
    elif task == 'F4_momentum':
        # Performance comparison
        return convert_v12(sample)
    return sample['question']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--f1_oversample', type=int, default=2,
                        help='Oversample F1 by this factor')
    parser.add_argument('--f4_oversample', type=int, default=1,
                        help='Oversample F4 by this factor')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ['train_sft.json', 'test_sft.json']:
        data = json.load(open(os.path.join(args.input_dir, split)))
        is_train = 'train' in split

        output_repr = []  # task-specific represented data
        output_ao = []    # answer-only
        output_cot = []   # adaptive depth CoT

        for d in data:
            # Apply task-specific representation
            new_q = apply_task_specific_repr(d)
            d_repr = {**d, 'question': new_q, 'question_original': d['question']}
            output_repr.append(d_repr)

            if is_train:
                # AO version
                output_ao.append({
                    'task': d['task'],
                    'conversations': [
                        {'role': 'user', 'content': new_q},
                        {'role': 'assistant', 'content': f"<answer>({d['answer']})</answer>"},
                    ],
                    'metadata': d.get('metadata', {}),
                    'answer': d['answer'],
                })

                # Adaptive CoT
                # F1, F4: full compute CoT (computational)
                # F2: input-aligned full CoT (references direction count from input)
                # F3: minimal CoT (pattern recognition, full CoT hurts)
                try:
                    if d['task'] == 'F1_drawdown':
                        cot = generate_f1_cot(d_repr)
                    elif d['task'] == 'F2_correlation':
                        cot = generate_f2_input_aligned_cot(d_repr)
                    elif d['task'] == 'F3_event':
                        cot = generate_f3_minimal_cot(d_repr)
                    elif d['task'] == 'F4_momentum':
                        cot = generate_f4_cot(d_repr)
                    else:
                        continue

                    cot_item = {
                        'task': d['task'],
                        'conversations': [
                            {'role': 'user', 'content': new_q},
                            {'role': 'assistant', 'content': cot},
                        ],
                        'metadata': d.get('metadata', {}),
                        'answer': d['answer'],
                    }
                    output_cot.append(cot_item)

                    # Oversampling for compute CoT tasks
                    oversample = 1
                    if d['task'] == 'F1_drawdown':
                        oversample = args.f1_oversample
                    elif d['task'] == 'F4_momentum':
                        oversample = args.f4_oversample

                    if oversample > 1:
                        for _ in range(oversample - 1):
                            output_cot.append(cot_item)
                            output_ao.append(output_ao[-1])
                except:
                    pass

        # Save represented data (for test evaluation)
        json.dump(output_repr, open(os.path.join(args.output_dir, split), 'w'),
                  indent=2, ensure_ascii=False)

        if is_train:
            json.dump(output_ao, open(os.path.join(args.output_dir, 'train_ao.json'), 'w'),
                      indent=2, ensure_ascii=False)
            json.dump(output_cot, open(os.path.join(args.output_dir, 'train_cot.json'), 'w'),
                      indent=2, ensure_ascii=False)

            # Stats
            tc = Counter(d['task'] for d in output_cot)
            total = len(output_cot)
            print(f"\n{split} CoT data: {total} samples")
            for t in sorted(tc):
                lens = [len(d['conversations'][1]['content']) for d in output_cot if d['task'] == t]
                print(f"  {t}: {tc[t]} ({tc[t]/total*100:.1f}%), avg CoT = {np.mean(lens):.0f} chars")
        else:
            print(f"\n{split}: {len(output_repr)} samples")

    print(f"\nSaved to: {args.output_dir}/")
    print(f"Files: train_sft.json, test_sft.json, train_ao.json, train_cot.json")


if __name__ == '__main__':
    main()
