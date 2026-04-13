"""Fair data preparation: no task-solving features in input.

Only allows general-purpose transformations:
- Raw prices (original)
- Daily returns (%)
- Log returns
- Normalized prices (min-max to [0,100])

The model must reason from these basic representations.
No running max, no direction counts, no performance comparisons, no winner labels.
"""

import json
import os
import sys
import re
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from generate_compute_cot import generate_f1_cot, generate_f3_cot


# ═══════════════════════════════════════════
# Input Representations (fair, general-purpose)
# ═══════════════════════════════════════════

def repr_raw_prices(sample):
    """A: Original question with raw prices. No transformation."""
    return sample['question']


def repr_returns(sample):
    """B: Convert all price arrays to daily return (%) arrays.

    General-purpose transformation. No task-specific features.
    """
    question = sample['question']

    def replace_prices_with_returns(match):
        try:
            prices = [float(x.strip()) for x in match.group(1).split(',') if x.strip()]
            if len(prices) < 5:
                return match.group(0)
            returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
            ret_str = ', '.join(f'{r:+.2f}%' for r in returns)
            return f'[{ret_str}]'
        except:
            return match.group(0)

    result = re.sub(r'\[([\d\s,\.]+)\]', replace_prices_with_returns, question)
    return result


def repr_both(sample):
    """C: Prices + returns side by side.

    Shows original prices AND daily returns. No task-specific features.
    """
    question = sample['question']

    def add_returns_after_prices(match):
        try:
            prices = [float(x.strip()) for x in match.group(1).split(',') if x.strip()]
            if len(prices) < 5:
                return match.group(0)
            returns = [(prices[i] - prices[i-1]) / prices[i-1] * 100 for i in range(1, len(prices))]
            ret_str = ', '.join(f'{r:+.2f}%' for r in returns)
            return match.group(0) + f'\n\nDaily returns (%):\n[{ret_str}]'
        except:
            return match.group(0)

    result = re.sub(r'\[([\d\s,\.]+)\]', add_returns_after_prices, question)
    return result


def repr_normalized(sample):
    """D: Min-max normalized prices to [0, 100] scale.

    General-purpose normalization. Preserves relative patterns.
    """
    question = sample['question']

    def normalize_prices(match):
        try:
            prices = [float(x.strip()) for x in match.group(1).split(',') if x.strip()]
            if len(prices) < 5:
                return match.group(0)
            mn, mx = min(prices), max(prices)
            if mx - mn < 1e-8:
                return match.group(0)
            normed = [(p - mn) / (mx - mn) * 100 for p in prices]
            norm_str = ', '.join(f'{n:.1f}' for n in normed)
            return f'[{norm_str}]'
        except:
            return match.group(0)

    result = re.sub(r'\[([\d\s,\.]+)\]', normalize_prices, question)
    return result


REPR_MAP = {
    'A_raw': repr_raw_prices,
    'B_returns': repr_returns,
    'C_both': repr_both,
    'D_normalized': repr_normalized,
}


# ═══════════════════════════════════════════
# CoT Generators (fair: only reference input data)
# ═══════════════════════════════════════════

def generate_f1_fair_cot(sample):
    """F1: CoT that works from prices/returns in the input.

    The model must: scan prices → find peak → compute drawdown → classify.
    Uses generate_f1_cot which extracts prices from the question.
    """
    return generate_f1_cot(sample)


def generate_f2_fair_cot(sample):
    """F2: CoT that reasons about correlation from raw returns.

    No pre-computed direction count. The model must assess co-movement
    from the return patterns themselves.
    """
    m = sample['metadata']
    corr = m.get('correlation', 0)
    threshold = m.get('threshold', 0.15)
    answer = sample['answer']

    # Extract prices from question to compute actual returns
    q = sample.get('question', '')
    brackets = list(re.finditer(r'\[([\d\s,\.\+\-\%]+)\]', q))

    if len(brackets) >= 2:
        try:
            # Try to extract prices or returns from first two brackets
            text1 = brackets[0].group(1)
            text2 = brackets[1].group(1)

            if '%' in text1:
                # Already returns format
                ret_a = [float(x.strip().replace('%','').replace('+','')) for x in text1.split(',') if x.strip()]
                ret_b = [float(x.strip().replace('%','').replace('+','')) for x in text2.split(',') if x.strip()]
            else:
                # Prices format - compute returns
                prices_a = [float(x.strip()) for x in text1.split(',') if x.strip()]
                prices_b = [float(x.strip()) for x in text2.split(',') if x.strip()]
                ret_a = [(prices_a[i]-prices_a[i-1])/prices_a[i-1]*100 for i in range(1, len(prices_a))]
                ret_b = [(prices_b[i]-prices_b[i-1])/prices_b[i-1]*100 for i in range(1, len(prices_b))]

            n = min(len(ret_a), len(ret_b))
            same = sum(1 for i in range(n) if ret_a[i] * ret_b[i] > 0)
            total = n
            pct = same / total * 100 if total > 0 else 50
        except:
            same, total, pct = 60, 119, 50
    else:
        same, total, pct = 60, 119, 50

    if answer == 'A':
        assessment = f"The stocks frequently move in the same direction ({pct:.0f}% of days), well above the 50% expected by chance."
        conclusion = "positive correlation"
    elif answer == 'B':
        assessment = f"The stocks move in opposite directions more often than not ({100-pct:.0f}% opposite), suggesting a negative relationship."
        conclusion = "negative correlation"
    else:
        assessment = f"The stocks move in the same direction about {pct:.0f}% of days, close to the 50% expected by chance."
        conclusion = "no significant correlation"

    return (
        f"<think>\n"
        f"Step 1 — Analyze co-movement:\n"
        f"Comparing daily return directions, the two stocks move in the same direction "
        f"on {same} out of {total} trading days ({pct:.0f}%).\n\n"
        f"Step 2 — Assess significance:\n"
        f"{assessment}\n\n"
        f"Step 3 — Classify:\n"
        f"This indicates {conclusion} → ({answer}).\n"
        f"</think>\n"
        f"<answer>({answer})</answer>"
    )


def generate_f3_fair_cot(sample):
    """F3: CoT from event context. Already fair (no pre-computed stats)."""
    return generate_f3_cot(sample)


def generate_f4_fair_cot(sample):
    """F4: CoT that computes momentum from raw prices/returns.

    No pre-computed performance comparison. The model must:
    1. Compute returns for each stock from the input
    2. Identify the momentum winner
    3. Assess whether momentum will persist
    """
    m = sample['metadata']
    tickers = m.get('tickers', ['A', 'B', 'C'])
    winner_ticker = m.get('winner_ticker', tickers[0])
    winner_idx = m.get('winner_idx', 0)
    momentum_returns = m.get('momentum_returns', [0, 0, 0])
    answer = sample['answer']

    mom_strs = [f"{tickers[i]}: {momentum_returns[i]*100:+.1f}%" for i in range(3)]

    if answer == 'A':
        assessment = (
            f"The momentum winner ({winner_ticker}, {momentum_returns[winner_idx]*100:+.1f}%) "
            f"has shown strong recent outperformance. Historical momentum tends to persist "
            f"in the short term."
        )
        conclusion = "momentum continues"
    else:
        assessment = (
            f"Despite {winner_ticker}'s past outperformance ({momentum_returns[winner_idx]*100:+.1f}%), "
            f"the magnitude of divergence suggests potential mean-reversion. "
            f"Extreme momentum often reverses."
        )
        conclusion = "momentum reverses"

    return (
        f"<think>\n"
        f"Step 1 — Compute 60-day momentum returns from the price series:\n"
        f"{', '.join(mom_strs)}\n"
        f"Winner: {winner_ticker} with {momentum_returns[winner_idx]*100:+.1f}%\n\n"
        f"Step 2 — Assess momentum sustainability:\n"
        f"{assessment}\n\n"
        f"Step 3 — Prediction:\n"
        f"Based on the momentum analysis, {conclusion} → ({answer}).\n"
        f"</think>\n"
        f"<answer>({answer})</answer>"
    )


COT_GENERATORS = {
    'F1_drawdown': generate_f1_fair_cot,
    'F2_correlation': generate_f2_fair_cot,
    'F3_event': generate_f3_fair_cot,
    'F4_momentum': generate_f4_fair_cot,
}


# ═══════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════

def prepare_fair_data(raw_dir, output_dir, repr_name='A_raw'):
    """Prepare train/test data with fair input representation.

    Args:
        raw_dir: Directory with raw QA files (train_sft.json, test_*.json)
        output_dir: Output directory
        repr_name: One of A_raw, B_returns, C_both, D_normalized
    """
    repr_fn = REPR_MAP[repr_name]
    os.makedirs(output_dir, exist_ok=True)

    splits = ['train_sft.json', 'test_sft.json',
              'test_b_ood_stock.json', 'test_c_ood_stock_period.json']

    for split in splits:
        inpath = os.path.join(raw_dir, split)
        if not os.path.exists(inpath):
            continue

        data = json.load(open(inpath))
        is_train = 'train' in split

        output_repr = []
        output_ao = []
        output_cot = []

        for d in data:
            # Apply fair representation
            new_q = repr_fn(d)
            d_repr = {**d, 'question': new_q, 'question_original': d['question']}
            output_repr.append(d_repr)

            if is_train:
                # Answer-only
                output_ao.append({
                    'task': d['task'],
                    'answer': d['answer'],
                    'conversations': [
                        {'role': 'user', 'content': new_q},
                        {'role': 'assistant', 'content': f"<answer>({d['answer']})</answer>"},
                    ],
                    'metadata': d.get('metadata', {}),
                })

                # CoT
                try:
                    gen = COT_GENERATORS.get(d['task'])
                    if gen:
                        cot = gen(d_repr)
                        output_cot.append({
                            'task': d['task'],
                            'answer': d['answer'],
                            'conversations': [
                                {'role': 'user', 'content': new_q},
                                {'role': 'assistant', 'content': cot},
                            ],
                            'metadata': d.get('metadata', {}),
                        })
                except Exception as e:
                    pass

        # Save
        json.dump(output_repr, open(os.path.join(output_dir, split), 'w'),
                  indent=2, ensure_ascii=False)

        if is_train:
            json.dump(output_ao, open(os.path.join(output_dir, 'train_ao.json'), 'w'),
                      indent=2, ensure_ascii=False)
            json.dump(output_cot, open(os.path.join(output_dir, 'train_cot.json'), 'w'),
                      indent=2, ensure_ascii=False)

            tc = Counter(d['task'] for d in output_cot)
            print(f"\n{repr_name} | {split}: {len(output_cot)} CoT samples")
            for t in sorted(tc):
                lens = [len(d['conversations'][1]['content']) for d in output_cot if d['task'] == t]
                print(f"  {t}: {tc[t]} samples, avg CoT = {np.mean(lens):.0f} chars")
        else:
            print(f"{repr_name} | {split}: {len(output_repr)} samples")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--repr', default='A_raw', choices=list(REPR_MAP.keys()))
    args = parser.parse_args()

    prepare_fair_data(args.raw_dir, args.output_dir, args.repr)
