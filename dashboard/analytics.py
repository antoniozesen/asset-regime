import numpy as np
import pandas as pd


def zscore(series, window=60):
    s = pd.Series(series)
    roll_mean = s.rolling(window, min_periods=12).mean()
    roll_std = s.rolling(window, min_periods=12).std()
    return ((s - roll_mean) / roll_std.replace(0, np.nan)).values


def percentile_rank(series):
    s = pd.Series(series).dropna()
    if len(s) < 5:
        return np.nan
    return float((s < s.iloc[-1]).sum() / len(s) * 100)


def momentum(df, periods):
    if df.empty or len(df) < periods:
        return np.nan
    return float((df["value"].iloc[-1] / df["value"].iloc[-periods] - 1) * 100)


def max_drawdown(series):
    s = pd.Series(np.asarray(series).reshape(-1)).dropna()
    if len(s) < 2:
        return np.nan
    roll_max = s.cummax()
    dd = (s - roll_max) / roll_max
    return float(dd.min() * 100)
