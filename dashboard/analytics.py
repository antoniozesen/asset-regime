from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(series, window=60):
    s = pd.Series(series, dtype=float)
    roll_mean = s.rolling(window, min_periods=12).mean()
    roll_std = s.rolling(window, min_periods=12).std().replace(0, np.nan)
    return (s - roll_mean) / roll_std


def percentile_rank(series):
    s = pd.Series(series).dropna()
    if len(s) < 5:
        return np.nan
    return float((s < s.iloc[-1]).mean() * 100)


def momentum(df: pd.DataFrame, periods: int) -> float:
    if df.empty or len(df) <= periods:
        return np.nan
    return float((df["value"].iloc[-1] / df["value"].iloc[-periods] - 1) * 100)


def max_drawdown(series):
    s = pd.Series(np.asarray(series).reshape(-1)).dropna()
    if len(s) < 2:
        return np.nan
    peak = s.cummax()
    dd = (s - peak) / peak
    return float(dd.min() * 100)


def transform_by_type(df: pd.DataFrame, series_type: str) -> pd.DataFrame:
    s = pd.Series(df["value"], index=df.index, dtype=float)
    out = pd.DataFrame(index=s.index)
    st = series_type.upper()

    if st == "LEVEL":
        yoy = s.pct_change(12) * 100
        out["core"] = yoy
        out["z"] = zscore(yoy)
    elif st == "RATE":
        out["core"] = s
        out["z"] = zscore(s)
        out["chg3m"] = s.diff(3)
        out["chg12m"] = s.diff(12)
    elif st == "SPREAD":
        out["core"] = s
        out["z"] = zscore(s)
    elif st == "INDEX":
        out["core"] = s
        out["z"] = zscore(s)
        out["chg3m"] = s.diff(3)
    else:
        out["core"] = s
        out["z"] = zscore(s)

    return out.replace([np.inf, -np.inf], np.nan)


def sanitize_returns(ret: pd.DataFrame, min_rows: int = 24) -> tuple[pd.DataFrame, str]:
    if ret is None or ret.empty:
        return pd.DataFrame(), "empty_returns"
    x = ret.replace([np.inf, -np.inf], np.nan).dropna(how="all", axis=0).dropna(how="all", axis=1)
    x = x.fillna(0.0)
    if len(x) < min_rows or x.shape[1] < 2:
        return pd.DataFrame(), f"insufficient_matrix rows={len(x)} cols={x.shape[1] if not x.empty else 0}"
    if not np.isfinite(x.values).all():
        return pd.DataFrame(), "non_finite_after_clean"
    return x, "ok"
