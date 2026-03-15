import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Cross-Asset Monitor", page_icon="📊", layout="wide")

LANG = st.sidebar.selectbox("🌐 Language / Idioma", ["English", "Español"], index=0)

def t(en: str, es: str) -> str:
    return es if LANG == "Español" else en

st.markdown(
    """
<style>
body, .stApp { background-color: #0d1117; color: #e6edf3; }
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("## ⚙️ Configuration")
fred_key = st.sidebar.text_input("FRED API Key", type="password")
lookback = st.sidebar.slider(t("History (months)", "Histórico (meses)"), 12, 120, 60, 12)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred(series_id: str, api_key: str, limit: int = 1000) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=asc&limit={limit}"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        obs = [
            {"date": pd.to_datetime(o["date"]), "value": float(o["value"])}
            for o in resp.json().get("observations", [])
            if o.get("value") not in (".", None)
        ]
        return pd.DataFrame(obs).set_index("date") if obs else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_yahoo(ticker: str, period: str = "5y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            return pd.DataFrame()
        return df[["Close"]].rename(columns={"Close": "value"})
    except Exception:
        return pd.DataFrame()


def percentile_rank(series: pd.Series) -> float:
    s = pd.Series(series).dropna()
    if len(s) < 5:
        return np.nan
    return float((s < s.iloc[-1]).mean() * 100)


def momentum(df: pd.DataFrame, days: int) -> float:
    if df.empty or len(df) <= days:
        return np.nan
    return float((df["value"].iloc[-1] / df["value"].iloc[-days] - 1) * 100)


def max_drawdown(series: pd.Series) -> float:
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return np.nan
    peak = s.cummax()
    dd = (s - peak) / peak
    return float(dd.min() * 100)


def line_chart(title: str, datasets: list[tuple[str, pd.DataFrame, str]]) -> go.Figure:
    fig = go.Figure()
    for name, df, color in datasets:
        if df.empty:
            continue
        fig.add_trace(
            go.Scatter(x=df.index, y=df["value"], mode="lines", name=name, line={"color": color, "width": 1.5})
        )
    fig.update_layout(
        title=title,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font={"color": "#e6edf3"},
        xaxis={"gridcolor": "#30363d"},
        yaxis={"gridcolor": "#30363d"},
        height=340,
    )
    return fig


with st.spinner(t("Loading data…", "Cargando datos…")):
    rates = {
        "US 10Y": fetch_fred("DGS10", fred_key),
        "US 2Y": fetch_fred("DGS2", fred_key),
        "US CPI": fetch_fred("CPIAUCSL", fred_key),
        "VIX": fetch_fred("VIXCLS", fred_key),
    }
    tickers = {
        "SPY": "S&P 500",
        "VGK": "Europe",
        "EWJ": "Japan",
        "IEMG": "EM",
        "TLT": "US 20Y Bond",
        "GLD": "Gold",
        "EURUSD=X": "EUR/USD",
        "CL=F": "WTI Oil",
    }
    prices = {tk: fetch_yahoo(tk) for tk in tickers}

st.title("📊 " + t("Cross-Asset Market Monitor", "Monitor Cross-Asset"))
st.caption(datetime.now().strftime("%d %b %Y %H:%M"))

tabs = st.tabs([
    t("🌐 Overview", "🌐 Resumen"),
    t("📈 Assets", "📈 Activos"),
    t("🧭 Macro", "🧭 Macro"),
    t("🔧 Sources", "🔧 Fuentes"),
])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    for col, name in zip([c1, c2, c3, c4], ["US 10Y", "US 2Y", "VIX", "US CPI"]):
        df = rates[name]
        val = f"{df['value'].iloc[-1]:.2f}" if not df.empty else "—"
        col.metric(name, val)

    st.plotly_chart(
        line_chart(
            t("US Rates", "Tipos EEUU"),
            [
                ("US 10Y", rates["US 10Y"].last(f"{lookback}ME"), "#58a6ff"),
                ("US 2Y", rates["US 2Y"].last(f"{lookback}ME"), "#f85149"),
            ],
        ),
        use_container_width=True,
    )

with tabs[1]:
    rows = []
    for tk, name in tickers.items():
        df = prices.get(tk, pd.DataFrame())
        if df.empty:
            continue
        rows.append(
            {
                "Ticker": tk,
                t("Asset", "Activo"): name,
                "Last": round(df["value"].iloc[-1], 2),
                "1M %": round(momentum(df, 21), 1),
                "3M %": round(momentum(df, 63), 1),
                "12M %": round(momentum(df, 252), 1),
                "Hist.Pct": round(percentile_rank(df["value"]), 1),
                "Max DD %": round(max_drawdown(df["value"]), 1),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.plotly_chart(
        line_chart(
            t("Selected Assets", "Activos Seleccionados"),
            [
                ("SPY", prices["SPY"].last(f"{lookback}ME"), "#3fb950"),
                ("TLT", prices["TLT"].last(f"{lookback}ME"), "#58a6ff"),
                ("GLD", prices["GLD"].last(f"{lookback}ME"), "#d29922"),
                ("WTI", prices["CL=F"].last(f"{lookback}ME"), "#f85149"),
            ],
        ),
        use_container_width=True,
    )

with tabs[2]:
    st.markdown("#### " + t("Macro notes", "Notas macro"))
    messages = []
    if not rates["US 10Y"].empty and not rates["US 2Y"].empty:
        slope = rates["US 10Y"].join(rates["US 2Y"], lsuffix="_10", rsuffix="_2").dropna()
        if not slope.empty:
            current = slope.iloc[-1]["value_10"] - slope.iloc[-1]["value_2"]
            messages.append(f"US 10Y-2Y slope: {current:+.2f}%")
    if not rates["US CPI"].empty:
        cpi = rates["US CPI"]["value"].pct_change(12).iloc[-1] * 100
        messages.append(f"US CPI YoY: {cpi:.2f}%")
    if not prices["EURUSD=X"].empty:
        fx = prices["EURUSD=X"]["value"].iloc[-1]
        messages.append(f"EUR/USD: {fx:.4f}")

    for m in messages or [t("No macro data. Add FRED key.", "Sin datos macro. Añade clave FRED.")]:
        st.markdown(f"- {m}")

with tabs[3]:
    src = pd.DataFrame(
        [
            ["FRED", "api.stlouisfed.org", "API key", "Rates, inflation, volatility"],
            ["Yahoo Finance", "query1.finance.yahoo.com", "None", "ETFs, futures, FX"],
        ],
        columns=[t("Source", "Fuente"), "URL", "Auth", t("Coverage", "Cobertura")],
    )
    st.dataframe(src, use_container_width=True, hide_index=True)
