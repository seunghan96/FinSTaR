"""V8: Generate Compute-Augmented CoT from metadata.

Creates CoT training data where the reasoning chain includes
EXACT numerical computation steps derived from metadata.
The model learns to compute statistics from raw time series.

Input: raw time series + question (NO stats injected)
Target: CoT with precise computation steps + answer

Usage:
    python ideas/fin_tsr/generate_compute_cot.py \
        --train_file data/fin_tsr_new_v3/train_sft.json \
        --output_file data/fin_tsr_new_v8/train_compute_cot.json
"""

import argparse
import json
import os
import numpy as np


def generate_f1_cot(sample: dict) -> str:
    """F1 Drawdown: compute peak, current, drawdown percentage."""
    m = sample['metadata']
    drawdown = m['drawdown']
    peak_day = m['peak_day']
    window_len = m['window_len']
    answer = sample['answer']

    # Extract actual prices from question
    question = sample['question']
    bracket_start = question.find('[')
    bracket_end = question.find(']')
    if bracket_start >= 0 and bracket_end >= 0:
        prices_str = question[bracket_start + 1:bracket_end]
        prices = [float(x.strip()) for x in prices_str.split(',')]
        peak_price = max(prices)
        current_price = prices[-1]
        peak_idx = prices.index(peak_price)
    else:
        peak_price = 0
        current_price = 0
        peak_idx = peak_day

    dd_pct = drawdown * 100

    thresholds = {
        'A': ('less than 3%', 'At/Near Peak'),
        'B': ('between 3% and 10%', 'Pullback'),
        'C': ('between 10% and 20%', 'Correction'),
        'D': ('20% or more', 'Severe Decline'),
    }
    range_desc, label = thresholds[answer]

    return (
        f"<think>\n"
        f"Step 1 — Find the peak price:\n"
        f"Scanning the 120-day series, the highest price is {peak_price:.2f} (around day {peak_idx + 1}).\n\n"
        f"Step 2 — Current price:\n"
        f"The last price in the series is {current_price:.2f}.\n\n"
        f"Step 3 — Calculate drawdown:\n"
        f"Drawdown = (Peak - Current) / Peak = ({peak_price:.2f} - {current_price:.2f}) / {peak_price:.2f} = {dd_pct:.1f}%\n\n"
        f"Step 4 — Classify:\n"
        f"{dd_pct:.1f}% is {range_desc}, which corresponds to ({answer}) {label}.\n"
        f"</think>\n"
        f"<answer>({answer})</answer>"
    )


def generate_f2_cot(sample: dict) -> str:
    """F2 Correlation: compute correlation and classify."""
    m = sample['metadata']
    corr = m['correlation']
    threshold = m['threshold']
    ticker_a = m['ticker_a']
    ticker_b = m['ticker_b']
    answer = sample['answer']

    abs_corr = abs(corr)

    if answer == 'A':
        direction = "positive"
        explanation = f"The correlation {corr:.3f} is above the threshold {threshold}, indicating the stocks tend to move in the same direction."
    elif answer == 'B':
        direction = "negative"
        explanation = f"The correlation {corr:.3f} is below -{threshold}, indicating the stocks tend to move in opposite directions."
    else:
        direction = "no significant"
        explanation = f"The absolute correlation |{corr:.3f}| = {abs_corr:.3f} is below the threshold {threshold}, indicating no clear relationship."

    return (
        f"<think>\n"
        f"Step 1 — Compute daily returns for both stocks and their correlation:\n"
        f"The Pearson correlation between {ticker_a} and {ticker_b} daily returns is {corr:.3f}.\n\n"
        f"Step 2 — Compare with threshold ({threshold}):\n"
        f"|correlation| = {abs_corr:.3f}\n\n"
        f"Step 3 — Classify:\n"
        f"{explanation}\n"
        f"This indicates {direction} correlation → ({answer}).\n"
        f"</think>\n"
        f"<answer>({answer})</answer>"
    )


def generate_f3_cot(sample: dict) -> str:
    """F3 Post-Event Volatility: predict vol increase vs decrease.

    Labeling:
      A: post_vol > pre_vol (volatility increases)
      B: post_vol <= pre_vol (volatility decreases)

    Predictive features (verified p<0.001):
      - pre-event volatility: low pre_vol → more likely to increase (78%)
      - |z-score|: larger shock → more likely to increase (68%)
      - event direction: negative shocks increase vol more (61% vs 50%)
    """
    m = sample['metadata']
    event_z = m['event_z']
    event_return = m['event_return']
    event_direction = m['event_direction']
    pre_trend = m['pre_trend']
    answer = sample['answer']

    event_return_pct = event_return * 100
    abs_z = abs(event_z)

    # Compute pre-event volatility from prices
    q = sample.get('question', '')
    pre_vol_pct = None
    bs, be = q.find('['), q.find(']')
    if bs >= 0 and be >= 0:
        try:
            prices = [float(x.strip()) for x in q[bs+1:be].split(',') if x.strip()]
            returns = [(prices[i]-prices[i-1])/prices[i-1] for i in range(1, len(prices))]
            pre_returns = returns[max(0, len(returns)-21):len(returns)-1]
            pre_vol_pct = np.std(pre_returns) * 100
        except:
            pass

    vol_str = f"{pre_vol_pct:.2f}%" if pre_vol_pct else "moderate"
    vol_level = "low" if (pre_vol_pct and pre_vol_pct < 1.5) else "high" if (pre_vol_pct and pre_vol_pct > 2.5) else "moderate"

    if answer == 'A':
        # Volatility increases
        if vol_level == 'low':
            reasoning = (
                f"Pre-event daily volatility was low ({vol_str}), indicating a calm market. "
                f"The extreme shock (z={abs_z:.1f}, return={event_return_pct:+.1f}%) represents a major "
                f"disruption to this calm state. When low-volatility stocks experience extreme events, "
                f"the shock typically triggers a regime shift toward higher volatility as the market "
                f"adjusts to new uncertainty."
            )
        elif abs_z > 4.0:
            reasoning = (
                f"The shock magnitude is very large (z={abs_z:.1f}), well beyond normal fluctuations. "
                f"Even with pre-event volatility at {vol_str}, shocks of this extreme magnitude tend "
                f"to trigger volatility clustering — a well-documented phenomenon where large moves "
                f"beget further large moves in subsequent days."
            )
        else:
            reasoning = (
                f"The {event_direction} shock (z={abs_z:.1f}) combined with "
                f"{'the negative direction (which tends to amplify volatility more than positive shocks)' if event_direction == 'negative' else 'the current market conditions'} "
                f"suggests elevated post-event turbulence. Pre-event volatility ({vol_str}) "
                f"is likely to increase as the market processes the shock."
            )
        conclusion = "post-event volatility will increase"
    else:
        # Volatility decreases
        if vol_level == 'high':
            reasoning = (
                f"Pre-event volatility was already elevated ({vol_str}), indicating an active market. "
                f"While the shock (z={abs_z:.1f}) is significant, the market was already in a "
                f"high-volatility state. In such conditions, the shock may represent a climactic move "
                f"that exhausts selling/buying pressure, leading to subsequent stabilization."
            )
        else:
            reasoning = (
                f"Despite the shock (z={abs_z:.1f}, return={event_return_pct:+.1f}%), the overall context "
                f"suggests the event will be absorbed. Pre-event volatility ({vol_str}) and "
                f"{'the positive direction of the shock' if event_direction == 'positive' else 'market conditions'} "
                f"suggest the move may be a one-time adjustment rather than the start of sustained turbulence."
            )
        conclusion = "post-event volatility will decrease"

    return (
        f"<think>\n"
        f"Step 1 — Event identification:\n"
        f"A {event_direction} shock occurred with z-score = {event_z:.2f} "
        f"(return = {event_return_pct:+.2f}%). This is an extreme deviation.\n\n"
        f"Step 2 — Pre-event volatility assessment:\n"
        f"Pre-event daily volatility: {vol_str} ({vol_level} level). "
        f"Pre-event trend: {pre_trend}.\n\n"
        f"Step 3 — Volatility prediction:\n"
        f"{reasoning}\n\n"
        f"Step 4 — Conclusion:\n"
        f"Based on the shock characteristics and pre-event conditions, "
        f"{conclusion} → ({answer}).\n"
        f"</think>\n"
        f"<answer>({answer})</answer>"
    )


def generate_f4_cot(sample: dict) -> str:
    """F4 Momentum: compare momentum and forward returns."""
    m = sample['metadata']
    tickers = m['tickers']
    momentum_returns = m['momentum_returns']
    forward_returns = m['forward_returns']
    winner_idx = m['winner_idx']
    winner_ticker = m['winner_ticker']
    winner_fwd = m['winner_fwd_return']
    portfolio_fwd = m['portfolio_fwd_return']
    answer = sample['answer']

    mom_strs = [f"{tickers[i]}: {momentum_returns[i]*100:+.1f}%" for i in range(3)]
    fwd_strs = [f"{tickers[i]}: {forward_returns[i]*100:+.1f}%" for i in range(3)]

    if answer == 'A':
        outcome = "momentum continues"
        explanation = f"{winner_ticker}'s forward return ({winner_fwd*100:+.1f}%) exceeds the portfolio average ({portfolio_fwd*100:+.1f}%)."
    else:
        outcome = "momentum reverses"
        explanation = f"{winner_ticker}'s forward return ({winner_fwd*100:+.1f}%) is below the portfolio average ({portfolio_fwd*100:+.1f}%)."

    return (
        f"<think>\n"
        f"Step 1 — Compute recent momentum (60-day returns):\n"
        f"{', '.join(mom_strs)}\n"
        f"Winner: {winner_ticker} with {momentum_returns[winner_idx]*100:+.1f}%\n\n"
        f"Step 2 — Compute forward returns (next 20 days):\n"
        f"{', '.join(fwd_strs)}\n"
        f"Portfolio average: {portfolio_fwd*100:+.1f}%\n\n"
        f"Step 3 — Compare winner vs portfolio:\n"
        f"{explanation}\n"
        f"This indicates {outcome} → ({answer}).\n"
        f"</think>\n"
        f"<answer>({answer})</answer>"
    )


GENERATORS = {
    'F1_drawdown': generate_f1_cot,
    'F2_correlation': generate_f2_cot,
    'F3_event': generate_f3_cot,
    'F4_momentum': generate_f4_cot,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    with open(args.train_file) as f:
        train_data = json.load(f)

    print(f"Loaded {len(train_data)} samples")

    output = []
    import numpy as np

    for d in train_data:
        task = d['task']
        generator = GENERATORS.get(task)
        if not generator:
            continue

        try:
            cot = generator(d)
        except Exception as e:
            print(f"Error on {d['question_id']}: {e}")
            continue

        output.append({
            'task': task,
            'conversations': [
                {'role': 'user', 'content': d['question']},
                {'role': 'assistant', 'content': cot},
            ],
            'metadata': d.get('metadata', {}),
            'answer': d['answer'],
        })

    # Stats
    from collections import Counter
    task_counts = Counter(d['task'] for d in output)
    cot_lens = [len(d['conversations'][1]['content']) for d in output]

    print(f"\nGenerated: {len(output)} samples")
    for task, count in sorted(task_counts.items()):
        task_lens = [len(d['conversations'][1]['content']) for d in output if d['task'] == task]
        print(f"  {task}: {count} samples, avg CoT = {np.mean(task_lens):.0f} chars")
    print(f"Overall CoT length: mean={np.mean(cot_lens):.0f}, median={np.median(cot_lens):.0f}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.output_file}")


if __name__ == '__main__':
    main()
