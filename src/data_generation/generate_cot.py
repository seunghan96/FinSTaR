"""Stage C: Generate Rule-based CoT annotations for financial TSQA.

Instead of using a model to generate CoT (which fails due to ~0% zero-shot accuracy
on financial tasks), we programmatically construct step-by-step reasoning chains
using the metadata and computed statistics from Stage B.

This is analogous to the "human-guided reasoning template" in TimeOmni-1 (Section 4),
but fully deterministic — guaranteeing 100% acceptance rate.

Usage:
    python ideas/fin_tsr/generate_cot.py \
        --data_dir data/fin_tsr \
        --output_dir data/fin_tsr
"""

import argparse
import json
import random
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# Rule-based CoT Generators
#
# Each function takes a QA sample (with metadata) and produces
# a structured reasoning chain that arrives at the correct answer.
# ══════════════════════════════════════════════════════════════

def generate_f1_cot(qa: dict) -> str:
    """Generate step-by-step reasoning for F1 (Market Regime)."""
    m = qa["metadata"]
    answer = qa["answer"]
    ticker = qa["ticker"]
    period = qa["period"]

    total_return = m["total_return"]
    vol = m["annualized_vol"]
    drawdown = m["max_drawdown"]
    num_days = m["num_days"]

    # Describe trend direction
    if total_return > 0.05:
        trend_desc = "upward"
        trend_detail = f"prices increased by {total_return:.1%} over the period"
    elif total_return < -0.05:
        trend_desc = "downward"
        trend_detail = f"prices declined by {abs(total_return):.1%} over the period"
    else:
        trend_desc = "relatively flat"
        trend_detail = f"prices changed only {total_return:+.1%} over the period"

    # Describe volatility
    if vol > 0.40:
        vol_desc = "very high"
        vol_detail = "significantly above typical levels, indicating extreme uncertainty"
    elif vol > 0.25:
        vol_desc = "elevated"
        vol_detail = "above average, suggesting heightened market activity"
    elif vol > 0.15:
        vol_desc = "moderate"
        vol_detail = "within normal range for equities"
    else:
        vol_desc = "low"
        vol_detail = "below average, indicating a calm market environment"

    # Describe drawdown
    if drawdown < -0.20:
        dd_desc = f"The maximum drawdown of {drawdown:.1%} is severe"
    elif drawdown < -0.10:
        dd_desc = f"The maximum drawdown of {drawdown:.1%} is notable"
    else:
        dd_desc = f"The maximum drawdown of {drawdown:.1%} is relatively contained"

    # Regime-specific reasoning
    regime_reasons = {
        "A": (
            f"The short-term moving average is above the long-term moving average, and both are trending upward. "
            f"The strong positive return of {total_return:.1%} confirms sustained buying pressure. "
            f"Volatility is {vol_desc}, consistent with a healthy uptrend."
        ),
        "B": (
            f"The short-term moving average has crossed below the long-term moving average, and both are declining. "
            f"The negative return of {total_return:.1%} reflects persistent selling pressure. "
            f"{dd_desc}, indicating sustained downward momentum."
        ),
        "C": (
            f"The moving averages are converging and relatively flat, with neither showing strong directional bias. "
            f"The small period return of {total_return:.1%} reflects a lack of directional conviction. "
            f"Volatility is {vol_desc} ({vol_detail}), with price oscillating around a mean level."
        ),
        "D": (
            f"Volatility at {vol:.1%} annualized is {vol_desc} — {vol_detail}. "
            f"Despite the overall return of {total_return:.1%}, the severe drawdown of {drawdown:.1%} "
            f"indicates large reversals within the period. The trend is unstable with no clear direction."
        ),
    }

    reasoning = (
        f"Step 1 — Basic Statistics:\n"
        f"Analyzing {ticker} over {num_days} trading days ({period}). "
        f"The period return is {total_return:.1%}. Annualized volatility is {vol:.1%}. "
        f"Maximum drawdown from peak is {drawdown:.1%}.\n\n"
        f"Step 2 — Trend Analysis:\n"
        f"The overall price trajectory is {trend_desc}: {trend_detail}. "
        f"Examining the moving average structure to determine trend strength and direction.\n\n"
        f"Step 3 — Volatility Assessment:\n"
        f"Annualized volatility of {vol:.1%} is {vol_desc} ({vol_detail}). "
        f"{dd_desc}.\n\n"
        f"Step 4 — Regime Classification:\n"
        f"{regime_reasons[answer]} "
        f"This pattern is most consistent with: {qa['answer_text']}."
    )

    return f"<think>\n{reasoning}\n</think>\n<answer>({answer})</answer>"


def generate_f2_cot(qa: dict) -> str:
    """Generate step-by-step reasoning for F2 (Cross-Asset Causality)."""
    m = qa["metadata"]
    answer = qa["answer"]
    ticker_a = m["ticker_a"]
    ticker_b = m["ticker_b"]
    pair_type = m["pair_type"]
    best_lag = m["best_lag"]
    best_corr = m["best_corr"]
    lag0_corr = m["lag0_corr"]

    # Describe pair type
    if pair_type == "same_sector":
        pair_desc = "Both stocks are in the same sector, which increases the likelihood of a meaningful relationship through shared industry drivers, supply-chain links, or competitive dynamics."
    else:
        pair_desc = "These stocks are from different sectors, so any relationship would indicate broader market linkages rather than direct industry connections."

    # Describe correlation strength
    abs_corr = abs(best_corr)
    if abs_corr > 0.6:
        corr_strength = "strong"
    elif abs_corr > 0.3:
        corr_strength = "moderate"
    elif abs_corr > 0.15:
        corr_strength = "weak"
    else:
        corr_strength = "negligible"

    # Describe lag
    if best_lag > 0:
        lag_desc = f"Stock A's movements appear to precede Stock B's by approximately {best_lag} trading days"
    elif best_lag < 0:
        lag_desc = f"Stock B's movements appear to precede Stock A's by approximately {abs(best_lag)} trading days"
    else:
        lag_desc = "Both stocks appear to move simultaneously without a clear time delay"

    # Relationship-specific reasoning
    rel_reasons = {
        "A": (
            f"The peak cross-correlation of {best_corr:.3f} occurs at lag {best_lag}, meaning {ticker_a}'s price changes "
            f"are followed by similar movements in {ticker_b} after approximately {best_lag} days. "
            f"This lead-lag pattern is {corr_strength} and statistically meaningful."
        ),
        "B": (
            f"The peak cross-correlation of {best_corr:.3f} occurs at lag {best_lag}, meaning {ticker_b}'s price changes "
            f"are followed by similar movements in {ticker_a} after approximately {abs(best_lag)} days. "
            f"This lead-lag pattern is {corr_strength} and suggests B is the information leader."
        ),
        "C": (
            f"The contemporaneous correlation (lag 0) of {lag0_corr:.3f} is the strongest, "
            f"and no significant lead-lag structure exists at other time offsets. "
            f"This indicates that both stocks respond to the same drivers simultaneously."
        ),
        "D": (
            f"The maximum cross-correlation of {best_corr:.3f} is {corr_strength}, "
            f"and no lag produces a statistically significant relationship. "
            f"The two stocks appear to be driven by independent factors."
        ),
    }

    reasoning = (
        f"Step 1 — Individual Series Analysis:\n"
        f"Examining the price movements of {ticker_a} (Stock A) and {ticker_b} (Stock B) "
        f"over the given period. Both series show their own trends and volatility patterns.\n\n"
        f"Step 2 — Co-movement Detection:\n"
        f"The contemporaneous correlation (at lag 0) between the two return series is {lag0_corr:.3f}, "
        f"which is {'positive' if lag0_corr > 0 else 'negative' if lag0_corr < 0 else 'near zero'}. "
        f"This provides initial evidence of {'co-movement' if abs(lag0_corr) > 0.2 else 'limited co-movement'}.\n\n"
        f"Step 3 — Lead-Lag Analysis:\n"
        f"Computing cross-correlation at various lags (-10 to +10 days). "
        f"The peak correlation is {best_corr:.3f} at lag {best_lag}. "
        f"{lag_desc}.\n\n"
        f"Step 4 — Sector & Fundamental Context:\n"
        f"{pair_desc}\n\n"
        f"Step 5 — Relationship Classification:\n"
        f"{rel_reasons[answer]} "
        f"Conclusion: {qa['answer_text']}."
    )

    return f"<think>\n{reasoning}\n</think>\n<answer>({answer})</answer>"


def generate_f3_cot(qa: dict) -> str:
    """Generate step-by-step reasoning for F3 (Event-Driven Forecast)."""
    m = qa["metadata"]
    answer = qa["answer"]
    ticker = qa["ticker"]
    event_date = m["event_date"]
    event_desc = m["event_description"]
    event_return = m["event_return"]
    five_day_return = m["five_day_return"]
    horizon = m["forecast_horizon"]

    # Describe event type
    is_positive_event = event_return > 0
    event_magnitude = abs(event_return)

    if event_magnitude > 0.10:
        mag_desc = "extremely large"
    elif event_magnitude > 0.05:
        mag_desc = "significant"
    else:
        mag_desc = "notable"

    # Describe what happened
    if "price surge" in event_desc:
        event_type = "a sharp price surge"
        event_nature = "buying pressure"
    elif "price drop" in event_desc:
        event_type = "a sharp price decline"
        event_nature = "selling pressure"
    elif "volume spike" in event_desc:
        event_type = "an unusual volume spike"
        event_nature = "heightened market activity"
    else:
        event_type = "a notable market event"
        event_nature = "changed market dynamics"

    # Post-event reasoning per answer
    direction_reasons = {
        "A": (
            f"After this event, the price moved upward by {five_day_return:.1%} over the next {horizon} trading days. "
            f"{'The initial positive momentum carried through, suggesting strong conviction behind the move.' if is_positive_event else 'Despite the negative event, the market overreacted and prices recovered — a classic mean-reversion pattern.'}"
        ),
        "B": (
            f"After this event, the price declined by {abs(five_day_return):.1%} over the next {horizon} trading days. "
            f"{'Despite the initial surge, the move proved unsustainable — suggesting a dead-cat bounce or profit-taking.' if is_positive_event else 'The negative momentum continued, indicating genuine deterioration rather than a temporary dip.'}"
        ),
        "C": (
            f"After this event, the price remained relatively flat (change of {five_day_return:+.1%}) over the next {horizon} trading days. "
            f"The market quickly digested the information and stabilized, with neither bulls nor bears gaining a decisive edge."
        ),
    }

    reasoning = (
        f"Step 1 — Pre-Event Trend Analysis:\n"
        f"Examining {ticker}'s price trajectory leading up to {event_date}. "
        f"The pre-event context provides baseline expectations for the stock's behavior.\n\n"
        f"Step 2 — Event Characterization:\n"
        f"On {event_date}, {ticker} experienced {event_type}: {event_desc}. "
        f"The event-day return of {event_return:+.1%} is {mag_desc}, indicating {event_nature}.\n\n"
        f"Step 3 — Post-Event Pattern Reasoning:\n"
        f"After a {mag_desc} {'positive' if is_positive_event else 'negative'} move, "
        f"two scenarios are most likely: momentum continuation (price keeps moving in the same direction) "
        f"or mean-reversion (price pulls back toward pre-event levels). "
        f"The magnitude of the move and market conditions determine which is more probable.\n\n"
        f"Step 4 — Momentum vs Mean-Reversion:\n"
        f"{'A large positive shock often leads to short-term profit-taking (mean-reversion) unless driven by fundamental news.' if is_positive_event else 'A large negative shock often triggers panic selling, but oversold conditions can lead to a bounce.'} "
        f"The event return of {event_return:+.1%} suggests the market needs time to find a new equilibrium.\n\n"
        f"Step 5 — Direction Prediction:\n"
        f"{direction_reasons[answer]} "
        f"Prediction: {qa['answer_text']}."
    )

    return f"<think>\n{reasoning}\n</think>\n<answer>({answer})</answer>"


def generate_f4_cot(qa: dict) -> str:
    """Generate step-by-step reasoning for F4 (Portfolio Decision)."""
    m = qa["metadata"]
    answer = qa["answer"]
    portfolio = m["portfolio"]
    lookback = m["lookback_returns"]
    forward = m["forward_returns"]
    action_rets = m["action_returns"]
    horizon = m["eval_horizon"]

    # Find best/worst performers
    sorted_by_lookback = sorted(lookback.items(), key=lambda x: x[1], reverse=True)
    best_stock = sorted_by_lookback[0]
    worst_stock = sorted_by_lookback[-1]

    # Average lookback return
    avg_lookback = sum(lookback.values()) / len(lookback)

    # Stock-by-stock analysis
    stock_analyses = []
    for ticker in portfolio:
        ret = lookback[ticker]
        if ret > 0.05:
            trend = "strong uptrend"
        elif ret > 0:
            trend = "mild uptrend"
        elif ret > -0.05:
            trend = "mild downtrend"
        else:
            trend = "notable downtrend"

        stock_analyses.append(
            f"  - {ticker}: lookback return {ret:+.1%} ({trend})"
        )

    # Strategy descriptions
    strategy_descs = {
        "A": f"Hold (equal weight): Expected return based on averaging all stocks' forward performance.",
        "B": f"Defensive (50% cash): Reduces exposure to protect against downside, but limits upside.",
        "C": f"Concentrate on best performer ({best_stock[0]}): Doubles weight on the strongest momentum stock.",
        "D": f"Rebalance to lowest drawdown: Shifts weight toward the most stable stock for risk reduction.",
    }

    # Why this action wins
    action_reasons = {
        "A": (
            f"The equal-weight hold strategy produces a return of {action_rets['A']:.1%}. "
            f"All three stocks contribute relatively balanced returns, making diversification the optimal choice. "
            f"No single stock dominates enough to justify concentration, and the overall trend doesn't warrant going defensive."
        ),
        "B": (
            f"The defensive strategy (50% cash) produces a return of {action_rets['B']:.1%}. "
            f"Given the {'negative' if avg_lookback < 0 else 'uncertain'} recent performance (average lookback: {avg_lookback:+.1%}), "
            f"reducing exposure protects capital. While other strategies {'also lose' if action_rets['A'] < 0 else 'may have higher returns'}, "
            f"the risk-adjusted return favors capital preservation."
        ),
        "C": (
            f"The concentration strategy on {best_stock[0]} produces a return of {action_rets['C']:.1%}. "
            f"{best_stock[0]} showed the strongest recent momentum ({best_stock[1]:+.1%}) and this momentum carries forward. "
            f"The forward return of {forward[best_stock[0]]:+.1%} for {best_stock[0]} outperforms the equal-weight portfolio."
        ),
        "D": (
            f"The rebalance strategy produces a return of {action_rets['D']:.1%}. "
            f"Shifting weight toward the stock with lowest recent drawdown provides better risk-adjusted returns. "
            f"This avoids concentrated exposure to high-momentum stocks that may reverse."
        ),
    }

    reasoning = (
        f"Step 1 — Individual Stock Assessment:\n"
        f"Analyzing the {len(portfolio)}-stock portfolio over the lookback period:\n"
        + "\n".join(stock_analyses) + "\n"
        f"Best performer: {best_stock[0]} ({best_stock[1]:+.1%}), "
        f"worst: {worst_stock[0]} ({worst_stock[1]:+.1%}).\n\n"
        f"Step 2 — Portfolio-Level Analysis:\n"
        f"The equal-weight portfolio averaged {avg_lookback:+.1%} in the lookback period. "
        f"{'Performance is spread across multiple stocks.' if abs(best_stock[1] - worst_stock[1]) < 0.10 else 'There is significant dispersion among the stocks, with a gap of ' + f'{abs(best_stock[1] - worst_stock[1]):.1%}' + ' between the best and worst.'}\n\n"
        f"Step 3 — Forward-Looking Risk Assessment:\n"
        f"{'The overall positive trend suggests continued growth potential, but concentration risk exists.' if avg_lookback > 0 else 'The recent weakness raises concerns about continued downside, warranting caution.'}\n\n"
        f"Step 4 — Strategy Evaluation:\n"
        + "\n".join(f"  - {strategy_descs[k]}" for k in ["A", "B", "C", "D"]) + "\n\n"
        f"Step 5 — Optimal Action:\n"
        f"{action_reasons[answer]} "
        f"Over the next {horizon} trading days, this strategy yields the best outcome. "
        f"Selected action: {qa['answer_text']}."
    )

    return f"<think>\n{reasoning}\n</think>\n<answer>({answer})</answer>"


# ══════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════

COT_GENERATORS = {
    "F1_regime": generate_f1_cot,
    "F2_causality": generate_f2_cot,
    "F3_forecast": generate_f3_cot,
    "F4_decision": generate_f4_cot,
}


def main():
    parser = argparse.ArgumentParser(description="Generate Rule-based CoT for FinTSR")
    parser.add_argument("--data_dir", default="data/fin_tsr",
                        help="Directory with Stage B QA data")
    parser.add_argument("--output_dir", default="data/fin_tsr",
                        help="Output directory for CoT data")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--tasks", nargs="+",
                        default=["F1_regime", "F2_causality", "F3_forecast", "F4_decision"],
                        help="Tasks to process")
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load raw QA files, filtered to TRAIN split only ──
    # Raw qa_*.json has all fields (answer_text, period, etc.) needed by CoT generators.
    # train_sft.json is used as an index to exclude test samples (avoid data leakage).
    task_to_file = {
        "F1_regime": "qa_f1_regime.json",
        "F2_causality": "qa_f2_causality.json",
        "F3_forecast": "qa_f3_forecast.json",
        "F4_decision": "qa_f4_decision.json",
    }

    # Build train question set from train_sft.json for filtering
    train_questions = set()
    train_path = data_dir / "train_sft.json"
    if train_path.exists():
        with open(train_path) as f:
            train_data = json.load(f)
        for sample in train_data:
            # Extract question text from conversations format
            q_text = sample["conversations"][0]["content"]
            train_questions.add(q_text)
        print(f"  Train index loaded: {len(train_questions)} questions from {train_path}")
    else:
        print(f"  [ERROR] {train_path} not found! Run generate_qa.py first.")
        return

    # Load raw QA files and filter to train-only
    all_qa = {}
    for task in args.tasks:
        qa_path = data_dir / task_to_file[task]
        if not qa_path.exists():
            print(f"  [WARN] {qa_path} not found, skipping {task}")
            continue
        with open(qa_path) as f:
            raw_samples = json.load(f)
        # Filter: keep only samples whose question appears in train_sft.json
        filtered = [s for s in raw_samples if s["question"] in train_questions]
        all_qa[task] = filtered
        print(f"  {task}: {len(filtered)}/{len(raw_samples)} samples (train only)")

    if not all_qa:
        print("ERROR: No QA data found. Run generate_qa.py first.")
        return

    # ── Generate CoT for each task (100% for SFT, no RL split) ──
    cot_data_all = []
    stats = {}

    for task, qa_list in all_qa.items():
        print(f"\n{'=' * 60}")
        print(f"Stage C: Generating Rule-based CoT for {task} ({len(qa_list)} questions)")
        print(f"{'=' * 60}")

        generator = COT_GENERATORS[task]

        for qi, qa in enumerate(qa_list):
            cot_response = generator(qa)
            cot_sample = {
                "task": task,
                "ticker": qa.get("ticker", ""),
                "conversations": [
                    {"role": "user", "content": qa["question"]},
                    {"role": "assistant", "content": cot_response},
                ],
                "answer": qa["answer"],
                "reasoning_length": len(cot_response),
                "metadata": qa.get("metadata", {}),
            }
            cot_data_all.append(cot_sample)

        print(f"  CoT generated: {len(qa_list)} samples (100%)")

        stats[task] = {
            "total": len(qa_list),
            "cot_data": len(qa_list),
            "acceptance_rate": 100.0,  # Rule-based → always 100%
        }

    # ── Save output ──
    cot_path = output_dir / "fin_cot_data.json"
    stats_path = output_dir / "stage_c_stats.json"

    with open(cot_path, "w") as f:
        json.dump(cot_data_all, f, indent=2, ensure_ascii=False)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("Stage C Complete! (Rule-based CoT)")
    print(f"{'=' * 60}")
    print(f"  Fin-CoT-Data: {len(cot_data_all)} samples → {cot_path}")
    print(f"  Stats:        {stats_path}")
    print()
    for task, s in stats.items():
        print(f"  {task}: {s['cot_data']}/{s['total']} CoT "
              f"(acceptance: {s['acceptance_rate']}%)")


if __name__ == "__main__":
    main()
