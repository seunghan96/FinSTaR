"""Technical analysis utilities for financial time series QA generation."""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Technical Indicators
# ──────────────────────────────────────────────

def compute_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series: pd.Series, window=20, num_std=2):
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_rolling_volatility(series: pd.Series, window=20):
    returns = series.pct_change()
    return returns.rolling(window=window).std() * np.sqrt(252)


def compute_daily_returns(series: pd.Series) -> pd.Series:
    return series.pct_change()


# ──────────────────────────────────────────────
# Regime Classification
# ──────────────────────────────────────────────

REGIME_LABELS = {
    "bull_trend": "A",
    "bear_trend": "B",
    "range_bound": "C",
    "volatile_transition": "D",
}

REGIME_DESCRIPTIONS = {
    "A": "Bull trend (sustained upward momentum with rising moving averages)",
    "B": "Bear trend (sustained downward momentum with falling moving averages)",
    "C": "Range-bound (sideways movement with low directional bias)",
    "D": "Volatile transition (high volatility with unclear direction)",
}


def classify_regime(close: pd.Series, window_short=20, window_long=60) -> str | None:
    """Classify market regime. Returns label letter or None if insufficient data."""
    if len(close) < window_long + 10:
        return None

    ma_short = compute_ma(close, window_short)
    ma_long = compute_ma(close, window_long)
    vol = compute_rolling_volatility(close, window_short)

    ma_s = ma_short.iloc[-1]
    ma_l = ma_long.iloc[-1]
    current_vol = vol.iloc[-1]

    if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(current_vol):
        return None

    # Slopes over last 10 trading days
    ma_s_slope = (ma_short.iloc[-1] - ma_short.iloc[-11]) / (abs(ma_short.iloc[-11]) + 1e-9)
    ma_l_slope = (ma_long.iloc[-1] - ma_long.iloc[-11]) / (abs(ma_long.iloc[-11]) + 1e-9)

    vol_median = vol.dropna().median()
    if pd.isna(vol_median) or vol_median == 0:
        vol_median = 0.2

    # High volatility → volatile transition
    if current_vol > vol_median * 1.5 and abs(ma_s_slope) < 0.01:
        return "D"

    # Bull: short MA > long MA, positive slopes
    if ma_s > ma_l and ma_s_slope > 0.003 and ma_l_slope > 0:
        return "A"

    # Bear: short MA < long MA, negative slopes
    if ma_s < ma_l and ma_s_slope < -0.003 and ma_l_slope < 0:
        return "B"

    # Sideways
    return "C"


# ──────────────────────────────────────────────
# Cross-Correlation / Causality
# ──────────────────────────────────────────────

CAUSALITY_LABELS = {
    "a_leads_b": "A",
    "b_leads_a": "B",
    "contemporaneous": "C",
    "no_relationship": "D",
}

CAUSALITY_DESCRIPTIONS = {
    "A": "Stock A leads Stock B (A's movements precede B's with a time lag)",
    "B": "Stock B leads Stock A (B's movements precede A's with a time lag)",
    "C": "Contemporaneous co-movement (both move together without clear lead-lag)",
    "D": "No significant relationship (movements are largely independent)",
}


def compute_lead_lag(series_a: pd.Series, series_b: pd.Series, max_lag=10):
    """Compute lead-lag relationship via cross-correlation of returns.

    Returns (label_letter, best_lag, best_corr, lag0_corr).
    Positive best_lag means A leads B by that many days.
    """
    ret_a = series_a.pct_change().dropna()
    ret_b = series_b.pct_change().dropna()
    common = ret_a.index.intersection(ret_b.index)
    if len(common) < max_lag + 20:
        return None

    ra = ret_a.loc[common].values
    rb = ret_b.loc[common].values

    # Standardize
    ra = (ra - ra.mean()) / (ra.std() + 1e-9)
    rb = (rb - rb.mean()) / (rb.std() + 1e-9)

    correlations = {}
    n = len(ra)
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            corr = np.dot(ra[:n - lag], rb[lag:]) / (n - lag)
        elif lag < 0:
            corr = np.dot(ra[-lag:], rb[:n + lag]) / (n + lag)
        else:
            corr = np.dot(ra, rb) / n
        correlations[lag] = corr

    best_lag = max(correlations, key=lambda k: abs(correlations[k]))
    best_corr = correlations[best_lag]
    lag0_corr = correlations[0]

    significance_threshold = 2.0 / np.sqrt(n)  # ~95% CI

    if abs(best_corr) < significance_threshold:
        return ("D", 0, best_corr, lag0_corr)

    if best_lag == 0 and abs(lag0_corr) > significance_threshold:
        return ("C", 0, lag0_corr, lag0_corr)

    if best_lag >= 1:
        return ("A", best_lag, best_corr, lag0_corr)  # A leads B
    elif best_lag <= -1:
        return ("B", abs(best_lag), best_corr, lag0_corr)  # B leads A
    else:
        return ("C", 0, lag0_corr, lag0_corr)


# ──────────────────────────────────────────────
# Event Detection
# ──────────────────────────────────────────────

def detect_events(close: pd.Series, volume: pd.Series = None,
                  return_threshold_sigma=2.0, volume_threshold_mult=2.5,
                  min_gap_days=10):
    """Detect significant price/volume events.

    Returns list of (date, daily_return, vol_ratio, event_description).
    """
    returns = close.pct_change()
    rolling_std = returns.rolling(60).std()
    events = []
    last_event_idx = -min_gap_days

    for i in range(61, len(close)):
        date = close.index[i]
        ret = returns.iloc[i]
        sigma = rolling_std.iloc[i]

        if pd.isna(ret) or pd.isna(sigma) or sigma == 0:
            continue

        z_score = ret / sigma
        vol_ratio = None
        if volume is not None:
            avg_vol = volume.iloc[max(0, i - 60):i].mean()
            if avg_vol > 0:
                vol_ratio = volume.iloc[i] / avg_vol

        is_event = False
        desc_parts = []

        if abs(z_score) >= return_threshold_sigma:
            is_event = True
            direction = "surge" if ret > 0 else "drop"
            desc_parts.append(f"price {direction} ({ret:+.1%})")

        if vol_ratio is not None and vol_ratio >= volume_threshold_mult:
            is_event = True
            desc_parts.append(f"volume spike ({vol_ratio:.1f}x average)")

        if is_event and (i - last_event_idx) >= min_gap_days:
            events.append({
                "date": date,
                "daily_return": float(ret),
                "z_score": float(z_score),
                "vol_ratio": float(vol_ratio) if vol_ratio else None,
                "description": ", ".join(desc_parts),
            })
            last_event_idx = i

    return events


# ──────────────────────────────────────────────
# Time Series Formatting
# ──────────────────────────────────────────────

def format_ts(series: pd.Series, max_len: int = 120, decimals: int = 2) -> str:
    """Format a time series as a compact string for LLM input."""
    values = series.tail(max_len).values
    formatted = [f"{v:.{decimals}f}" for v in values if not pd.isna(v)]
    return "[" + ", ".join(formatted) + "]"


def format_ts_with_dates(series: pd.Series, max_len: int = 120,
                         decimals: int = 2, date_freq: int = 20) -> str:
    """Format with periodic date annotations for context."""
    s = series.tail(max_len).dropna()
    lines = []
    for i, (date, val) in enumerate(s.items()):
        if i % date_freq == 0:
            lines.append(f"  # {date.strftime('%Y-%m-%d')}")
        lines.append(f"  {val:.{decimals}f},")
    return "[\n" + "\n".join(lines) + "\n]"


def compute_summary_stats(close: pd.Series) -> dict:
    """Compute summary statistics for a price series."""
    returns = close.pct_change().dropna()
    return {
        "mean_price": float(close.mean()),
        "std_price": float(close.std()),
        "min_price": float(close.min()),
        "max_price": float(close.max()),
        "total_return": float((close.iloc[-1] / close.iloc[0]) - 1),
        "annualized_vol": float(returns.std() * np.sqrt(252)),
        "max_drawdown": float((close / close.cummax() - 1).min()),
        "sharpe_approx": float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)),
    }
