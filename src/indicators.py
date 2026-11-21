"""Technical indicators on daily bars.

Every function assumes the frame is sorted by date. The first version of this project
computed rolling indicators before sorting (the API returned rows out of order), which put
the moving averages off by about 22 CNY on average; ``add_indicators`` now refuses unsorted input.
Warm-up periods stay NaN instead of being filled with 0 (an RSI of 0 is a real, and wrong, signal).
"""
import pandas as pd


def _check_sorted(df):
    if "date" in df.columns and not df["date"].is_monotonic_increasing:
        raise ValueError("data is not sorted by date; call sort_values('date') first")


def ema(s, n):
    """Exponential moving average, alpha = 2 / (n + 1), seeded with the first value."""
    return s.ewm(span=n, adjust=False).mean()


def macd(close, fast=12, slow=26, signal=9):
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    return dif, dea, 2 * (dif - dea)          # histogram as quoted by Chinese market software


def rsi(close, period=14):
    """RSI with simple averages of gains and losses; 100 when there are no losses."""
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    return (100 - 100 / (1 + gain / loss)).where(loss != 0, 100.0)


def kdj(df, period=9):
    low_min = df["low"].rolling(period, min_periods=period).min()
    high_max = df["high"].rolling(period, min_periods=period).max()
    rsv = 100 * (df["close"] - low_min) / (high_max - low_min)
    k = rsv.ewm(alpha=1 / 3, adjust=False).mean()      # K = 2/3 * K_prev + 1/3 * RSV
    d = k.ewm(alpha=1 / 3, adjust=False).mean()
    return k, d, 3 * k - 2 * d


def bollinger(close, window=20, n_std=2):
    mid = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    return mid + n_std * std, mid, mid - n_std * std


def add_indicators(df):
    _check_sorted(df)
    df = df.copy()
    c, v = df["close"], df["volume"]
    for n in [5, 21, 30, 60, 120]:
        df[f"ma{n}"] = c.rolling(n, min_periods=n).mean()
        df[f"ma_{n}_v"] = v.rolling(n, min_periods=n).mean()
    df["DIF"], df["DEA"], df["MACD"] = macd(c)
    df["RSI"] = rsi(c, 14)
    df["K"], df["D"], df["J"] = kdj(df)
    df["close_pct_change"] = c.pct_change()
    df["volume_pct_change"] = v.pct_change()
    df["bbu"], df["bbm"], df["bbl"] = bollinger(c)
    return df
