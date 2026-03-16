from io import StringIO

import pandas as pd
import requests
import streamlit as st


def ensure_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["value"])

    out = df.copy()

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

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).set_index("date")

    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[~out.index.isna()]

    return out.sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_id, api_key, limit=500):
    if not api_key:
        return pd.DataFrame()
    try:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={api_key}&file_type=json"
            f"&sort_order=desc&limit={limit}"
        )
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        df = pd.DataFrame(
            [{"date": o["date"], "value": float(o["value"])} for o in obs if o.get("value") not in (".", None)]
        )
        if df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        return ensure_timeseries(df.sort_values("date").set_index("date"))
    except Exception:
        return pd.DataFrame()


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
        df = pd.DataFrame(rows, columns=["date", "value"]).dropna()
        return ensure_timeseries(df.set_index("date"))
    except Exception:
        return pd.DataFrame()


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
        out = pd.DataFrame({"value": close})
        return ensure_timeseries(out)
    except Exception:
        return pd.DataFrame()


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
            return pd.DataFrame()
        out = df[[time_col[0], val_col[0]]].copy()
        out.columns = ["date", "value"]
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return ensure_timeseries(out.dropna().sort_values("date").set_index("date"))
    except Exception:
        return pd.DataFrame()
