import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.analytics import max_drawdown, momentum, percentile_rank, zscore
from dashboard.data import fetch_bis, fetch_ecb, fetch_fred, fetch_yahoo
from dashboard.theme import DARK_COLORS, apply_theme

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Cross-Asset Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

LANG = st.sidebar.selectbox("🌐 Language / Idioma", ["English", "Español"], index=0)

def t(en, es):
    return es if LANG == "Español" else en

st.sidebar.markdown("## ⚙️ Configuration")
fred_key = st.sidebar.text_input("FRED API Key", type="password")
lookback = st.sidebar.slider(t("History (months)", "Histórico (meses)"), 12, 120, 60, 12)


def line_fig(datasets, title, height=350, ref_zero=False):
    fig = go.Figure()
    for i, (label, df, col) in enumerate(datasets):
        if df.empty:
            continue
        fig.add_trace(
            go.Scatter(x=df.index, y=df["value"], name=label, mode="lines", line=dict(color=col, width=1.5))
        )
    if ref_zero:
        fig.add_hline(y=0, line_dash="dash", line_color=DARK_COLORS["muted"], line_width=1)
    fig.update_layout(
        height=height,
        title=title,
        paper_bgcolor=DARK_COLORS["bg"],
        plot_bgcolor=DARK_COLORS["panel"],
        font=dict(color=DARK_COLORS["text"], size=11),
        xaxis=dict(gridcolor=DARK_COLORS["grid"]),
        yaxis=dict(gridcolor=DARK_COLORS["grid"]),
        legend=dict(bgcolor=DARK_COLORS["panel"], bordercolor=DARK_COLORS["grid"], borderwidth=1),
    )
    return fig


with st.spinner(t("Loading market data…", "Cargando datos de mercado…")):
    fred = {}
    if fred_key:
        for k, sid in {
            "us10y": "DGS10",
            "us2y": "DGS2",
            "us_cpi": "CPIAUCSL",
            "ig_oas": "BAMLC0A0CM",
            "hy_oas": "BAMLH0A0HYM2",
            "vix": "VIXCLS",
            "real10y": "DFII10",
        }.items():
            fred[k] = fetch_fred(sid, fred_key)

    ecb_ea10y = fetch_ecb("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y", "2005-01")
    ecb_ea2y = fetch_ecb("YC", "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y", "2005-01")
    ecb_eurusd = fetch_ecb("EXR", "D.USD.EUR.SP00.A", "2005-01")
    bis_credit_us = fetch_bis("WS_TC_C", "Q.US.P.A.M.770.A")

    tickers = {
        "SPY": "S&P 500 ETF",
        "VGK": "Europe ETF",
        "EWJ": "Japan ETF",
        "IEMG": "EM ETF",
        "TLT": "US 20Y Bond ETF",
        "LQD": "US IG Corp ETF",
        "HYG": "US HY ETF",
        "GLD": "Gold ETF",
        "CL=F": "WTI Oil",
        "EURUSD=X": "EUR/USD",
    }
    prices = {tk: fetch_yahoo(tk, "5y") for tk in tickers}


st.markdown(f"# 📊 {t('Cross-Asset Market Monitor', 'Monitor Cross-Asset')}")
st.caption(datetime.now().strftime("%d %b %Y %H:%M"))

tabs = st.tabs(
    [
        t("🌐 Overview", "🌐 Resumen"),
        t("📈 Equities", "📈 Renta Variable"),
        t("🏦 Fixed Income", "🏦 Renta Fija"),
        t("💱 FX", "💱 Divisas"),
        t("🛢 Commodities", "🛢 Materias Primas"),
        t("🧭 Macro", "🧭 Macro"),
        t("🔧 Sources", "🔧 Fuentes"),
    ]
)

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EA 10Y", f"{ecb_ea10y['value'].iloc[-1]:.2f}%" if not ecb_ea10y.empty else "—")
    c2.metric("EA 2Y", f"{ecb_ea2y['value'].iloc[-1]:.2f}%" if not ecb_ea2y.empty else "—")
    c3.metric("EUR/USD", f"{ecb_eurusd['value'].iloc[-1]:.4f}" if not ecb_eurusd.empty else "—")
    c4.metric("VIX", f"{fred['vix']['value'].iloc[-1]:.2f}" if fred_key and not fred.get("vix", pd.DataFrame()).empty else "—")

    st.plotly_chart(
        line_fig(
            [
                ("EA 10Y", ecb_ea10y.last(f"{lookback}ME"), DARK_COLORS["blue"]),
                ("EA 2Y", ecb_ea2y.last(f"{lookback}ME"), DARK_COLORS["violet"]),
                ("US 10Y", fred.get("us10y", pd.DataFrame()).last(f"{lookback}ME"), DARK_COLORS["red"]),
            ],
            t("Rates Overview", "Resumen de Tipos"),
        ),
        use_container_width=True,
    )

with tabs[1]:
    rows = []
    for tk in ["SPY", "VGK", "EWJ", "IEMG"]:
        df = prices.get(tk, pd.DataFrame())
        if df.empty:
            continue
        rows.append(
            {
                "Ticker": tk,
                t("Asset", "Activo"): tickers[tk],
                "1M %": round(momentum(df, 21), 1),
                "3M %": round(momentum(df, 63), 1),
                "12M %": round(momentum(df, 252), 1),
                t("Vol 3M", "Vol 3M"): round(df["value"].pct_change().rolling(63).std().iloc[-1] * np.sqrt(252) * 100, 1),
                t("Max DD", "Max DD"): round(max_drawdown(df["value"]), 1),
                t("Hist.Pct", "Pct.Hist."): round(percentile_rank(df["value"]), 1),
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tabs[2]:
    st.plotly_chart(
        line_fig(
            [
                ("TLT", prices["TLT"].last(f"{lookback}ME"), DARK_COLORS["blue"]),
                ("LQD", prices["LQD"].last(f"{lookback}ME"), DARK_COLORS["green"]),
                ("HYG", prices["HYG"].last(f"{lookback}ME"), DARK_COLORS["red"]),
            ],
            t("Bond ETF Proxies", "ETFs Proxy de Bonos"),
        ),
        use_container_width=True,
    )

    if fred_key and not fred.get("ig_oas", pd.DataFrame()).empty:
        c1, c2 = st.columns(2)
        c1.metric("US IG OAS", f"{fred['ig_oas']['value'].iloc[-1]:.0f}bp")
        c2.metric("US HY OAS", f"{fred['hy_oas']['value'].iloc[-1]:.0f}bp" if not fred.get("hy_oas", pd.DataFrame()).empty else "—")

with tabs[3]:
    st.plotly_chart(
        line_fig(
            [
                ("EUR/USD", ecb_eurusd.last(f"{lookback}ME"), DARK_COLORS["blue"]),
                ("EUR/USD (Yahoo)", prices["EURUSD=X"].last(f"{lookback}ME"), DARK_COLORS["green"]),
            ],
            "EUR/USD",
        ),
        use_container_width=True,
    )

with tabs[4]:
    st.plotly_chart(
        line_fig(
            [
                ("GLD", prices["GLD"].last(f"{lookback}ME"), DARK_COLORS["amber"]),
                ("WTI", prices["CL=F"].last(f"{lookback}ME"), DARK_COLORS["red"]),
            ],
            t("Commodities", "Materias Primas"),
        ),
        use_container_width=True,
    )

with tabs[5]:
    st.markdown(f"### {t('Macro Snapshot', 'Snapshot Macro')}")
    macro_rows = []
    if not ecb_ea10y.empty and not ecb_ea2y.empty:
        slope = (ecb_ea10y.join(ecb_ea2y, lsuffix="_10", rsuffix="_2").dropna())
        if not slope.empty:
            macro_rows.append({"Signal": "EA 10Y-2Y", "Value": round(slope.iloc[-1]["value_10"] - slope.iloc[-1]["value_2"], 2)})
    if fred_key and not fred.get("us_cpi", pd.DataFrame()).empty:
        us_cpi_yoy = fred["us_cpi"]["value"].pct_change(12).iloc[-1] * 100
        macro_rows.append({"Signal": "US CPI YoY", "Value": round(us_cpi_yoy, 2)})
    if not bis_credit_us.empty:
        macro_rows.append({"Signal": "BIS US Credit/GDP", "Value": round(bis_credit_us["value"].iloc[-1], 2)})
    if macro_rows:
        st.dataframe(pd.DataFrame(macro_rows), use_container_width=True, hide_index=True)

    if fred_key and not fred.get("us_cpi", pd.DataFrame()).empty:
        cpi_series = fred["us_cpi"]["value"].pct_change(12) * 100
        z = zscore(cpi_series.dropna().values, 60)
        if len(z) > 0 and not np.isnan(z[-1]):
            st.metric("US CPI Z-score", f"{z[-1]:+.2f}")

with tabs[6]:
    sources = [
        ["ECB", "data-api.ecb.europa.eu", "None", "EA yields, FX"],
        ["FRED", "api.stlouisfed.org", "API key", "US rates, CPI, spreads"],
        ["BIS", "stats.bis.org", "None", "Credit/GDP"],
        ["Yahoo Finance", "query1.finance.yahoo.com", "None", "ETF, commodity, FX prices"],
    ]
    st.dataframe(
        pd.DataFrame(sources, columns=[t("Source", "Fuente"), "URL", "Auth", t("Coverage", "Cobertura")]),
        use_container_width=True,
        hide_index=True,
    )

    health = {
        "ECB EA10Y": len(ecb_ea10y),
        "ECB EURUSD": len(ecb_eurusd),
        "BIS US Credit": len(bis_credit_us),
        "Yahoo SPY": len(prices.get("SPY", pd.DataFrame())),
    }
    if fred_key:
        health["FRED US10Y"] = len(fred.get("us10y", pd.DataFrame()))
        health["FRED US CPI"] = len(fred.get("us_cpi", pd.DataFrame()))
    st.dataframe(pd.DataFrame(list(health.items()), columns=["Series", "Observations"]), hide_index=True)
