from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st


@dataclass
class SeriesSelection:
    concept: str
    region: str
    series_id: str | None
    title: str
    units: str
    frequency: str
    source_mode: str
    score: float
    reason: str
    excluded_reason: str


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


def ensure_timeseries(df: pd.DataFrame | pd.Series | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["value"])

    if isinstance(df, pd.Series):
        out = df.to_frame("value")
    else:
        out = df.copy()

    if out.empty:
        return pd.DataFrame(columns=["value"])

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]) if isinstance(c, tuple) else str(c) for c in out.columns]

    if "value" not in out.columns:
        candidates = ["Close", "Adj Close", "close", "VALUE", "OBS_VALUE", "obs_value"]
        found = next((c for c in candidates if c in out.columns), None)
        if found is None:
            numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
            if not numeric_cols:
                return pd.DataFrame(columns=["value"])
            found = numeric_cols[0]
        out = out[[found]].rename(columns={found: "value"})
    else:
        out = out[["value"]]

    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value"])

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    return out.sort_index()


def recent_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    ts = ensure_timeseries(df)
    if ts.empty:
        return ts
    cutoff = ts.index.max() - pd.DateOffset(months=months)
    return ts[ts.index >= cutoff]


@st.cache_data(ttl=3600, show_spinner=False)
def get_fred_client(api_key: str | None):
    if not api_key:
        return None
    try:
        from fredapi import Fred

        return Fred(api_key=api_key)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_id: str, api_key: str | None) -> pd.DataFrame:
    fred = get_fred_client(api_key)
    if fred is None:
        return pd.DataFrame(columns=["value"])
    try:
        s = fred.get_series(series_id)
        return ensure_timeseries(s)
    except Exception:
        return pd.DataFrame(columns=["value"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_info(series_id: str, api_key: str | None) -> dict[str, Any]:
    fred = get_fred_client(api_key)
    if fred is None:
        return {}
    try:
        info = fred.get_series_info(series_id)
        return {
            "title": str(info.get("title", "")),
            "units": str(info.get("units", "")),
            "frequency": str(info.get("frequency", "")),
        }
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def search_fred(query: str, api_key: str | None, limit: int = 12) -> pd.DataFrame:
    fred = get_fred_client(api_key)
    if fred is None:
        return pd.DataFrame()
    try:
        out = fred.search(query)
        return out.head(limit) if isinstance(out, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _frequency_score(freq: str) -> float:
    f = (freq or "").lower()
    if "monthly" in f:
        return 1.0
    if "quarterly" in f:
        return 0.6
    if "weekly" in f:
        return 0.5
    if "daily" in f:
        return 0.3
    return 0.2


def _series_quality_score(df: pd.DataFrame, info: dict[str, Any], concept: str) -> tuple[float, dict[str, Any]]:
    ts = ensure_timeseries(df)
    if ts.empty:
        return -1e9, {"staleness": np.nan, "missingness": 1.0, "len": 0}

    last_obs = ts.index.max()
    staleness_days = int((_now_utc() - pd.Timestamp(last_obs)).days)
    missingness = float(ts["value"].isna().mean())
    hist_len = len(ts)

    recency_penalty = 0.0
    if staleness_days > 120:
        recency_penalty -= 1.5
    if staleness_days > 365:
        recency_penalty -= 2.5

    score = (
        2.0 * _frequency_score(info.get("frequency", ""))
        + 2.0 * np.clip(1 - staleness_days / 365.0, -2, 1)
        + 1.5 * np.clip(1 - missingness, 0, 1)
        + 1.0 * np.log1p(hist_len / 12)
        + recency_penalty
    )

    return float(score), {"staleness": staleness_days, "missingness": missingness, "len": hist_len}


def pick_best_series(concept: str, region: str, candidates: list[str], search_query: str, api_key: str | None):
    rows: list[dict[str, Any]] = []

    for sid in candidates:
        df = fetch_fred(sid, api_key)
        info = fetch_fred_info(sid, api_key)
        score, q = _series_quality_score(df, info, concept)
        rows.append(
            {
                "concept": concept,
                "region": region,
                "series_id": sid,
                "title": info.get("title", ""),
                "units": info.get("units", ""),
                "frequency": info.get("frequency", ""),
                "source_mode": "candidate",
                "score": score,
                "staleness_days": q["staleness"],
                "missingness_pct": q["missingness"] * 100,
                "history_len": q["len"],
            }
        )

    if rows:
        scored = pd.DataFrame(rows).sort_values("score", ascending=False)
        top = scored.iloc[0]
        if np.isfinite(top["score"]) and top["history_len"] > 24:
            st = top['staleness_days']
            stale_txt = 'na' if pd.isna(st) else str(int(st))
            reason = f"best candidate score={top['score']:.2f}, stale={stale_txt}d"
            return top["series_id"], reason, "candidate", scored

    fb = search_fred(search_query, api_key, limit=15)
    if not fb.empty:
        for _, item in fb.iterrows():
            sid = str(item.get("id", ""))
            if not sid:
                continue
            df = fetch_fred(sid, api_key)
            info = fetch_fred_info(sid, api_key)
            score, q = _series_quality_score(df, info, concept)
            rows.append(
                {
                    "concept": concept,
                    "region": region,
                    "series_id": sid,
                    "title": info.get("title", ""),
                    "units": info.get("units", ""),
                    "frequency": info.get("frequency", ""),
                    "source_mode": "search_fallback",
                    "score": score,
                    "staleness_days": q["staleness"],
                    "missingness_pct": q["missingness"] * 100,
                    "history_len": q["len"],
                }
            )

    scored = pd.DataFrame(rows).sort_values("score", ascending=False) if rows else pd.DataFrame()
    if not scored.empty and np.isfinite(scored.iloc[0]["score"]):
        top = scored.iloc[0]
        st = top['staleness_days']
        stale_txt = 'na' if pd.isna(st) else str(int(st))
        reason = f"fallback selection score={top['score']:.2f}, stale={stale_txt}d"
        return top["series_id"], reason, top["source_mode"], scored

    return None, "no valid series", "none", scored


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ecb(flow, key, start="2005-01"):
    try:
        url = f"https://data-api.ecb.europa.eu/service/data/{flow}/{key}?startPeriod={start}&format=jsondata"
        r = requests.get(url, headers={"Accept": "application/json"}, timeout=15)
        r.raise_for_status()
        d = r.json()
        series = list(d["dataSets"][0]["series"].values())[0]
        periods = d["structure"]["dimensions"]["observation"][0]["values"]
        rows = [(pd.to_datetime(p["id"]), series["observations"].get(str(i), [None])[0]) for i, p in enumerate(periods)]
        return ensure_timeseries(pd.DataFrame(rows, columns=["date", "value"]).set_index("date"))
    except Exception:
        return pd.DataFrame(columns=["value"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bis(flow, key):
    try:
        url = f"https://stats.bis.org/api/v1/data/{flow}/{key}?startPeriod=2000&format=csv"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        time_col = [c for c in df.columns if "TIME" in c.upper() or "PERIOD" in c.upper() or "DATE" in c.upper()]
        val_col = [c for c in df.columns if "OBS" in c.upper() or "VALUE" in c.upper()]
        if not time_col or not val_col:
            return pd.DataFrame(columns=["value"])
        out = df[[time_col[0], val_col[0]]].copy()
        out.columns = ["date", "value"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return ensure_timeseries(out.set_index("date"))
    except Exception:
        return pd.DataFrame(columns=["value"])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo(ticker, period="5y"):
    try:
        import yfinance as yf

        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame(columns=["value"])
        close = df.get("Close")
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        out = pd.DataFrame({"value": close}, index=df.index)
        return ensure_timeseries(out)
    except Exception:
        return pd.DataFrame(columns=["value"])
