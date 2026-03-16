from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.analytics import (
    max_drawdown,
    momentum,
    percentile_rank,
    sanitize_returns,
    transform_by_type,
    zscore,
)
from dashboard.data import (
    fetch_bis,
    fetch_fred,
    fetch_fred_info,
    fetch_yahoo,
    pick_best_series,
    recent_months,
)
from dashboard.theme import DARK_COLORS, apply_theme

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Cross-Asset Monitor", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
apply_theme()

LANG = st.sidebar.selectbox("🌐 Language / Idioma", ["English", "Español"], index=0)

def t(en, es):
    return es if LANG == "Español" else en

try:
    api_key = st.secrets.get("FRED_API_KEY", "")
except Exception:
    api_key = ""
fred_key = st.sidebar.text_input("FRED API Key", type="password", value=api_key)
lookback = st.sidebar.slider(t("History (months)", "Histórico (meses)"), 12, 180, 84, 12)

CANDIDATES = {
    "US": {
        "growth": ["INDPRO", "RRSFS", "PAYEMS", "IPMAN"],
        "inflation": ["CPIAUCSL", "PCEPI", "CPILFESL", "PCEPILFE"],
        "labor": ["UNRATE", "PAYEMS", "CIVPART", "U6RATE"],
        "leading": ["ICSA", "USSLIND", "UMCSENT", "DGORDER"],
        "rates": ["DGS2", "DGS10", "FEDFUNDS", "DFII10"],
        "conditions": ["NFCI", "ANFCI", "STLFSI4", "GS10"],
        "stress": ["VIXCLS", "MOVEINDEX", "BAA10YM", "TEDRATE"],
    },
    "Europe": {
        "growth": ["CLVMNACSCAB1GQEA19", "OECDPRINTO01GYSAM", "EA19RGDPEQDSMEI"],
        "inflation": ["CP0000EZ19M086NEST", "CPHPTT01EZM659N", "EA19CPALTT01IXOBSAM"],
        "labor": ["LRHUTTTTEZM156S", "LFWA64TTEZM647S", "LRUN64TTEZM156S"],
        "leading": ["OECDLOLITOAASTSAM", "BSCICP03EZM665S", "CSCICP03EZM665S", "OECDCBLI01EUM661N"],
        "rates": ["IR3TIB01EZM156N", "IRLTLT01EZM156N", "IRSTCB01EZM156N"],
    },
    "Japan": {
        "growth": ["JPNPROINDMISMEI", "JPNRGDPEXP", "JPNPRINTO01GYSAM"],
        "inflation": ["JPNCPIALLMINMEI", "JPNCPALTT01IXOBSAM"],
        "labor": ["LRUNTTTTJPM156S", "LREMTTTTJPM156S"],
        "leading": ["OECDLOLITOAASTSAM", "JPNLOLITONOSTSAM"],
        "rates": ["IR3TIB01JPM156N", "IRLTLT01JPM156N", "IRSTCI01JPM156N"],
    },
    "EM": {
        "growth": ["OECDPRINTO01GYSAM", "CPALTT01CNM657N"],
        "inflation": ["FPCPITOTLZGCHN", "FPCPITOTLZGIND", "FPCPITOTLZGZAF"],
    },
}

SERIES_TYPES = {
    "growth": "LEVEL",
    "inflation": "LEVEL",
    "labor": "RATE",
    "leading": "INDEX",
    "rates": "RATE",
    "conditions": "SPREAD",
    "stress": "SPREAD",
}

TICKERS = {
    "SPY": "S&P 500 ETF", "VGK": "Europe ETF", "EWJ": "Japan ETF", "IEMG": "EM ETF",
    "GLD": "Gold ETF", "TLT": "US 20Y Bond ETF", "IEF": "US 7-10Y Bond", "LQD": "US IG Corp ETF",
    "HYG": "US HY ETF", "BIL": "US T-Bill ETF", "XLK": "Tech", "XLF": "Financials",
    "XLI": "Industrials", "XLV": "Healthcare", "XLP": "Cons.Staples", "XLU": "Utilities",
    "XLE": "Energy", "XLB": "Materials", "XLY": "Cons.Discret.", "XLRE": "Real Estate", "XLC": "Comm.Services",
    "QUAL": "Quality Factor", "MTUM": "Momentum Factor", "USMV": "Low Vol Factor", "VLUE": "Value Factor",
    "VUG": "Growth Factor", "IVE": "US Value", "IVW": "US Growth", "GC=F": "Gold Futures",
    "BZ=F": "Brent Oil", "CL=F": "WTI Oil", "NG=F": "Nat.Gas", "HG=F": "Copper", "SI=F": "Silver",
    "EURUSD=X": "EUR/USD", "USDJPY=X": "USD/JPY", "GBPUSD=X": "GBP/USD", "USDCHF=X": "USD/CHF",
}

MACRO_ATLAS = {
    "US Activity": ["INDPRO", "PAYEMS", "RRSFS", "IPMAN", "HOUST", "DGORDER"],
    "US Inflation": ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "PPIFID", "MEDCPIM158SFRBCLE"],
    "US Labor": ["UNRATE", "U6RATE", "CIVPART", "AHETPI", "ICSA", "CCSA"],
    "US Rates & Conditions": ["DGS2", "DGS5", "DGS10", "DGS30", "DFII10", "FEDFUNDS", "NFCI", "BAA10YM", "TEDRATE", "VIXCLS"],
    "Surveys": ["UMCSENT", "BUSLOANS", "PMI", "NAPM", "AMTMNO", "BSCICP03USM665S"],
    "Europe": ["CP0000EZ19M086NEST", "IR3TIB01EZM156N", "IRLTLT01EZM156N", "BSCICP03EZM665S", "CSCICP03EZM665S", "LRHUTTTTEZM156S"],
    "Japan": ["JPNCPIALLMINMEI", "JPNPROINDMISMEI", "IRLTLT01JPM156N", "IR3TIB01JPM156N", "LRUNTTTTJPM156S"],
    "EM / Global": ["FPCPITOTLZGCHN", "FPCPITOTLZGIND", "FPCPITOTLZGZAF", "PALLFNFINDEXM", "GOLDAMGBD228NLBM", "DCOILBRENTEU"],
}


def safe_last(df: pd.DataFrame, fmt: str = "{:.2f}", suffix: str = ""):
    if df is None or df.empty:
        return "—"
    v = df["value"].dropna()
    if v.empty:
        return "—"
    return fmt.format(float(v.iloc[-1])) + suffix


def line_fig(datasets, title, height=320):
    fig = go.Figure()
    for lbl, df, color in datasets:
        if df is None or df.empty:
            continue
        fig.add_trace(go.Scatter(x=df.index, y=df["value"], mode="lines", name=lbl, line=dict(color=color, width=1.7)))
    fig.update_layout(
        height=height,
        title=title,
        paper_bgcolor=DARK_COLORS["bg"],
        plot_bgcolor=DARK_COLORS["panel"],
        font=dict(color=DARK_COLORS["text"]),
        xaxis=dict(gridcolor=DARK_COLORS["grid"]),
        yaxis=dict(gridcolor=DARK_COLORS["grid"]),
        legend=dict(bgcolor=DARK_COLORS["panel"]),
    )
    return fig


def build_curve(fred_map: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, str]:
    two = fred_map.get("DGS2", pd.DataFrame())
    ten = fred_map.get("DGS10", pd.DataFrame())
    if two.empty or ten.empty:
        return pd.DataFrame(columns=["2Y", "10Y", "Slope"]), "Curve unavailable (missing DGS2 or DGS10)"
    c = two.rename(columns={"value": "2Y"}).join(ten.rename(columns={"value": "10Y"}), how="inner")
    if c.empty:
        return pd.DataFrame(columns=["2Y", "10Y", "Slope"]), "Curve unavailable (no overlap between DGS2 and DGS10)"
    c["Slope"] = c["10Y"] - c["2Y"]
    return c[["2Y", "10Y", "Slope"]], "ok"


def compute_regime(features: pd.DataFrame, min_months: int = 60):
    eff = features.replace([np.inf, -np.inf], np.nan).dropna()
    if len(eff) < min_months:
        proxy_score = 0
        if not eff.empty:
            last = eff.iloc[-1]
            proxy_score = (1 if last.get("growth_z", 0) > 0 else -1) + (1 if last.get("inflation_z", 0) > 0 else -1)
        label = "Goldilocks" if proxy_score >= 1 else ("Stagflation" if proxy_score <= -1 else "Slowdown")
        return None, {"mode": "proxy", "effective_months": len(eff), "label": label}

    try:
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=4, covariance_type="full", random_state=42)
        probs = gmm.fit_predict(eff.values)
        out = eff.copy()
        out["cluster"] = probs
        centers = out.groupby("cluster")[["growth_z", "inflation_z"]].mean()
        mapping = {}
        for cl, row in centers.iterrows():
            if row["growth_z"] >= 0 and row["inflation_z"] >= 0:
                mapping[cl] = "Reflation"
            elif row["growth_z"] >= 0 and row["inflation_z"] < 0:
                mapping[cl] = "Goldilocks"
            elif row["growth_z"] < 0 and row["inflation_z"] >= 0:
                mapping[cl] = "Stagflation"
            else:
                mapping[cl] = "Slowdown"
        out["regime"] = out["cluster"].map(mapping)
        return out, {"mode": "gmm", "effective_months": len(eff), "label": out["regime"].iloc[-1]}
    except Exception:
        return None, {"mode": "proxy", "effective_months": len(eff), "label": "Slowdown"}


def optimize_allocation(monthly_ret: pd.DataFrame, regime_label: str):
    anchors = pd.Series({"SPY": 0.30, "VGK": 0.08, "EWJ": 0.07, "IEMG": 0.10, "TLT": 0.20, "LQD": 0.15, "GLD": 0.10})
    cols = [c for c in anchors.index if c in monthly_ret.columns]
    anchors = anchors[cols]
    anchors = anchors / anchors.sum()

    clean, status = sanitize_returns(monthly_ret[cols] if cols else pd.DataFrame())
    if clean.empty:
        return anchors, {"mode": "fallback", "reason": status, "binding": []}

    mu = clean.mean()
    if regime_label in ("Reflation", "Goldilocks"):
        mu = mu + clean.tail(3).mean() * 0.3
    else:
        mu = mu + clean.tail(3).mean() * -0.1

    cov = clean.cov()
    try:
        from sklearn.covariance import LedoitWolf

        cov = pd.DataFrame(LedoitWolf().fit(clean.values).covariance_, index=clean.columns, columns=clean.columns)
    except Exception:
        pass

    n = len(cols)
    b = np.array([anchors[c] for c in cols])
    w0 = b.copy()
    bounds = [(max(0, bi - 0.15), min(0.6, bi + 0.15)) for bi in b]

    def obj(w):
        return -(w @ mu.values - 3.0 * np.sqrt(max(1e-9, w @ cov.values @ w)))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    try:
        from scipy.optimize import minimize

        res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            bounds = [(0.0, 0.7) for _ in range(n)]
            res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
        if not res.success:
            return anchors, {"mode": "fallback", "reason": res.message, "binding": []}

        w = pd.Series(res.x, index=cols)
        binding = [c for c, (lo, hi) in zip(cols, bounds) if abs(w[c] - lo) < 1e-4 or abs(w[c] - hi) < 1e-4]
        return w, {"mode": "optimized", "reason": "ok", "binding": binding}
    except Exception:
        # deterministic heuristic fallback when scipy is unavailable
        tilt = (mu - mu.mean()).clip(-0.01, 0.01)
        w = (anchors + tilt).clip(lower=0)
        w = w / w.sum()
        return w, {"mode": "fallback", "reason": "scipy_unavailable", "binding": []}


with st.spinner(t("Loading data from FRED + Yahoo…", "Cargando datos de FRED + Yahoo…")):
    dictionary_rows = []
    selected = {}

    for region, concepts in CANDIDATES.items():
        for concept, cands in concepts.items():
            sid, reason, mode, scored = pick_best_series(concept, region, cands, f"{region} {concept}", fred_key)
            if sid:
                df = fetch_fred(sid, fred_key)
                info = fetch_fred_info(sid, fred_key)
                selected[(region, concept)] = df
                dictionary_rows.append(
                    {
                        "region": region,
                        "concept": concept,
                        "series_id": sid,
                        "title": info.get("title", ""),
                        "units": info.get("units", ""),
                        "frequency": info.get("frequency", ""),
                        "transformation": SERIES_TYPES.get(concept, "LEVEL"),
                        "last_obs_date": str(df.index.max().date()) if not df.empty else "",
                        "staleness_days": (pd.Timestamp.now() - pd.Timestamp(df.index.max())).days if not df.empty else np.nan,
                        "missingness_pct": float(df["value"].isna().mean() * 100) if not df.empty else 100.0,
                        "source_mode": mode,
                        "excluded_reason": "",
                        "selection_reason": reason,
                    }
                )
            else:
                dictionary_rows.append(
                    {
                        "region": region,
                        "concept": concept,
                        "series_id": "",
                        "title": "",
                        "units": "",
                        "frequency": "",
                        "transformation": SERIES_TYPES.get(concept, "LEVEL"),
                        "last_obs_date": "",
                        "staleness_days": np.nan,
                        "missingness_pct": 100.0,
                        "source_mode": "none",
                        "excluded_reason": reason,
                        "selection_reason": reason,
                    }
                )

    data_dict = pd.DataFrame(dictionary_rows)

    fred_curve = {"DGS2": fetch_fred("DGS2", fred_key), "DGS10": fetch_fred("DGS10", fred_key), "DFII10": fetch_fred("DFII10", fred_key), "VIXCLS": fetch_fred("VIXCLS", fred_key), "NFCI": fetch_fred("NFCI", fred_key), "BAA10YM": fetch_fred("BAA10YM", fred_key)}
    curve_df, curve_status = build_curve(fred_curve)

    prices = {tk: fetch_yahoo(tk, "5y") for tk in TICKERS}

    atlas_series = {}
    atlas_info = {}
    for grp, series_ids in MACRO_ATLAS.items():
        for sid in series_ids:
            if sid in atlas_series:
                continue
            atlas_series[sid] = fetch_fred(sid, fred_key)
            atlas_info[sid] = fetch_fred_info(sid, fred_key)

    metrics = {}
    for tk, df in prices.items():
        if df.empty:
            continue
        v = df["value"]
        metrics[tk] = {
            "m1": momentum(df, 21),
            "m3": momentum(df, 63),
            "m12": momentum(df, 252),
            "dd": max_drawdown(v.values),
            "pct": percentile_rank(v),
            "vol": float(v.pct_change().rolling(63).std().iloc[-1] * np.sqrt(252) * 100) if len(v) > 80 else np.nan,
            "vs_ma200": float((v.iloc[-1] / v.rolling(200).mean().iloc[-1] - 1) * 100) if len(v) > 220 else np.nan,
        }

# regime features
feature_parts = {}
for concept in ["growth", "inflation", "labor", "stress", "rates", "conditions"]:
    src = selected.get(("US", concept), pd.DataFrame())
    if src.empty:
        continue
    tr = transform_by_type(src, SERIES_TYPES.get(concept, "LEVEL"))
    if "z" in tr.columns:
        feature_parts[f"{concept}_z"] = tr["z"]

if not curve_df.empty:
    sl = curve_df[["Slope"]].rename(columns={"Slope": "slope"})
    feature_parts["slope_z"] = zscore(sl["slope"])

features = pd.DataFrame(feature_parts).resample("ME").last() if feature_parts else pd.DataFrame()
regime_df, regime_meta = compute_regime(features)

monthly_prices = pd.DataFrame({k: v["value"].resample("ME").last() for k, v in prices.items() if not v.empty})
monthly_ret = monthly_prices.pct_change().dropna(how="all")
weights, alloc_meta = optimize_allocation(monthly_ret, regime_meta["label"])

# Self-checks
checks = {
    "curve_schema_ok": list(curve_df.columns) == ["2Y", "10Y", "Slope"] if not curve_df.empty else True,
    "returns_finite": bool(np.isfinite(monthly_ret.replace([np.inf, -np.inf], np.nan).fillna(0).values).all()) if not monthly_ret.empty else True,
    "data_dict_complete": bool(data_dict[["region", "concept", "transformation", "source_mode"]].notna().all().all()),
}

st.markdown(f"# 📊 {t('Cross-Asset Market Monitor', 'Monitor Cross-Asset')}")
st.caption(t(f"Live data · {datetime.now().strftime('%d %b %Y %H:%M')}", f"Datos en vivo · {datetime.now().strftime('%d %b %Y %H:%M')}"))


tabs = st.tabs([
    t("🌐 Overview", "🌐 Resumen"),
    t("📉 Markets", "📉 Mercados"),
    t("🧭 Macro", "🧭 Macro"),
    t("📊 Signals", "📊 Señales"),
    t("⚠️ Risk", "⚠️ Riesgo"),
    t("🧮 Allocation", "🧮 Asignación"),
    t("🗂 Data Quality", "🗂 Calidad de Datos"),
    t("🧱 Macro Atlas", "🧱 Atlas Macro"),
    t("📚 ETFs & Indices", "📚 ETFs e Índices"),
])

with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", regime_meta["label"], f"mode={regime_meta['mode']}")
    c2.metric("Effective sample", str(regime_meta["effective_months"]), "months")
    c3.metric("VIX", safe_last(fred_curve["VIXCLS"]), f"{percentile_rank(fred_curve['VIXCLS']['value']) if not fred_curve['VIXCLS'].empty else np.nan:.0f}th pct")
    c4.metric("Breadth >200D", f"{np.mean([1 if (not prices[k].empty and len(prices[k])>220 and prices[k]['value'].iloc[-1] > prices[k]['value'].rolling(200).mean().iloc[-1]) else 0 for k in ['SPY','VGK','EWJ','IEMG']])*100:.0f}%")

    rs = []
    for tk in ["QUAL", "MTUM", "VLUE", "USMV", "VUG"]:
        if tk in monthly_prices and "SPY" in monthly_prices:
            rel = (monthly_prices[tk] / monthly_prices["SPY"]).dropna()
            rs.append((tk, float(rel.pct_change(6).iloc[-1] * 100) if len(rel) > 7 else np.nan))
    if rs:
        st.dataframe(pd.DataFrame(rs, columns=["Factor", "6M rel vs SPY %"]), width="stretch", hide_index=True)

    perf = pd.DataFrame({"Ticker": list(metrics.keys()), "12M": [metrics[k]["m12"] for k in metrics]}).dropna().sort_values("12M")
    if not perf.empty:
        fig = px.bar(perf.tail(15), x="12M", y="Ticker", orientation="h", title="Top 12M movers", color="12M", color_continuous_scale="RdYlGn")
        fig.update_layout(paper_bgcolor=DARK_COLORS["bg"], plot_bgcolor=DARK_COLORS["panel"], font=dict(color=DARK_COLORS["text"]))
        st.plotly_chart(fig, width="stretch")

    takeaways = []
    if not curve_df.empty:
        slope = curve_df["Slope"].dropna()
        if not slope.empty:
            takeaways.append(f"US 10Y-2Y slope is {slope.iloc[-1]:+.2f}%.")
    if not fred_curve["BAA10YM"].empty:
        takeaways.append(f"Credit spread percentile: {percentile_rank(fred_curve['BAA10YM']['value']):.0f}th.")
    takeaways.append(f"Regime mode={regime_meta['mode']} with effective sample {regime_meta['effective_months']} months.")
    for tx in takeaways:
        st.markdown(f"- {tx.replace('nan', 'N/A')}")

with tabs[1]:
    st.warning(curve_status) if curve_status != "ok" else None
    if not curve_df.empty:
        plot_curve = recent_months(curve_df.rename(columns={"Slope": "value"})[["value"]], lookback)
        st.plotly_chart(line_fig([("Slope 10Y-2Y", plot_curve, DARK_COLORS["amber"])], "US Curve Slope"), width="stretch")

    st.plotly_chart(
        line_fig(
            [
                ("2Y", recent_months(fred_curve["DGS2"], lookback), DARK_COLORS["blue"]),
                ("10Y", recent_months(fred_curve["DGS10"], lookback), DARK_COLORS["red"]),
                ("Real10Y", recent_months(fred_curve["DFII10"], lookback), DARK_COLORS["green"]),
            ],
            "Rates & Real Yield",
        ),
        width="stretch",
    )

with tabs[2]:
    regions = ["US", "Europe", "Japan", "EM"]
    for reg in regions:
        st.markdown(f"#### {reg}")
        cols = st.columns(3)
        for i, concept in enumerate(["growth", "inflation", "labor"]):
            df = selected.get((reg, concept), pd.DataFrame())
            tr_type = SERIES_TYPES.get(concept, "LEVEL")
            if df.empty:
                cols[i].warning(f"{concept}: missing")
                continue
            tr = transform_by_type(df, tr_type)
            cols[i].metric(f"{concept} z", f"{tr['z'].dropna().iloc[-1]:+.2f}" if not tr['z'].dropna().empty else "—", f"last {df.index.max().date()}")

with tabs[3]:
    heat_rows = []
    bucket_map = {
        "regions": ["SPY", "VGK", "EWJ", "IEMG"],
        "sectors": ["XLK", "XLF", "XLI", "XLV", "XLP", "XLU", "XLE", "XLB", "XLY", "XLRE", "XLC"],
        "factors": ["QUAL", "MTUM", "USMV", "VLUE", "VUG", "IVE", "IVW"],
        "bonds_gold": ["TLT", "IEF", "LQD", "HYG", "GLD"],
    }
    for b, tks in bucket_map.items():
        for tk in tks:
            if tk in metrics:
                heat_rows.append({"bucket": b, "ticker": tk, "m3": metrics[tk]["m3"], "vol": metrics[tk]["vol"], "vs_ma200": metrics[tk]["vs_ma200"]})
    heat = pd.DataFrame(heat_rows)
    if not heat.empty:
        pivot = heat.pivot_table(index="ticker", columns="bucket", values="m3", aggfunc="mean")
        fig = px.imshow(pivot.fillna(0), color_continuous_scale="RdYlGn", aspect="auto", title="Momentum heatmap by bucket")
        fig.update_layout(paper_bgcolor=DARK_COLORS["bg"], font=dict(color=DARK_COLORS["text"]))
        st.plotly_chart(fig, width="stretch")

with tabs[4]:
    core = [c for c in ["SPY", "TLT", "LQD", "HYG", "GLD"] if c in monthly_ret.columns]
    if core:
        vol = monthly_ret[core].rolling(12).std() * np.sqrt(12) * 100
        dd = (monthly_prices[core] / monthly_prices[core].cummax() - 1) * 100
        st.plotly_chart(line_fig([(c, vol[[c]].rename(columns={c: "value"}), DARK_COLORS["blue"]) for c in core], "Rolling 12M Vol"), width="stretch")
        st.plotly_chart(line_fig([(c, dd[[c]].rename(columns={c: "value"}), DARK_COLORS["red"]) for c in core], "Drawdowns"), width="stretch")
        corr = monthly_ret[core].tail(36).corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation heatmap (36m)")
        fig.update_layout(paper_bgcolor=DARK_COLORS["bg"], font=dict(color=DARK_COLORS["text"]))
        st.plotly_chart(fig, width="stretch")

with tabs[5]:
    anchor = pd.Series({"SPY": 0.30, "VGK": 0.08, "EWJ": 0.07, "IEMG": 0.10, "TLT": 0.20, "LQD": 0.15, "GLD": 0.10})
    alloc = pd.DataFrame({"anchor": anchor, "recommended": weights}).fillna(0)
    alloc["turnover"] = (alloc["recommended"] - alloc["anchor"]).abs()
    st.dataframe((alloc * 100).round(2), width="stretch")
    st.caption(f"Mode: {alloc_meta['mode']} · reason: {alloc_meta['reason']} · binding: {', '.join(alloc_meta['binding']) if alloc_meta['binding'] else 'none'}")

    scenario = st.selectbox("What-if scenario", ["regime_flip_to_stagflation", "regime_flip_to_goldilocks", "risk_off"])
    adj = alloc["recommended"].copy()
    if scenario == "risk_off":
        for tk in ["TLT", "GLD"]:
            if tk in adj.index:
                adj[tk] += 0.03
    if scenario == "regime_flip_to_stagflation":
        for tk in ["GLD", "XLE"]:
            if tk in adj.index:
                adj[tk] += 0.02
    if scenario == "regime_flip_to_goldilocks":
        for tk in ["SPY", "XLK"]:
            if tk in adj.index:
                adj[tk] += 0.02
    if adj.sum() > 0:
        adj = adj / adj.sum()
    st.dataframe(pd.DataFrame({"scenario_weight": (adj * 100).round(2)}), width="stretch")

with tabs[6]:
    st.dataframe(data_dict, width="stretch", hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("usable_series", str((data_dict["series_id"] != "").sum()))
    c2.metric("missing_series", str((data_dict["series_id"] == "").sum()))
    c3.metric("effective_regime_months", str(regime_meta["effective_months"]))

    stale = data_dict["staleness_days"].dropna()
    if not stale.empty:
        fig = px.histogram(stale, nbins=20, title="Staleness distribution")
        fig.update_layout(paper_bgcolor=DARK_COLORS["bg"], plot_bgcolor=DARK_COLORS["panel"], font=dict(color=DARK_COLORS["text"]))
        st.plotly_chart(fig, width="stretch")

    dropped = data_dict[(data_dict["staleness_days"] > 365) | (data_dict["missingness_pct"] > 30)]
    if not dropped.empty:
        st.markdown("**Dropped / weak series**")
        st.dataframe(dropped[["region", "concept", "series_id", "staleness_days", "missingness_pct", "selection_reason"]], width="stretch", hide_index=True)

    st.markdown("**Self-checks**")
    st.json(checks)

with tabs[7]:
    st.markdown("### Macro Atlas — Full Coverage from FRED")
    for grp, series_ids in MACRO_ATLAS.items():
        st.markdown(f"#### {grp}")
        rows = []
        chart_sets = []
        palette = [DARK_COLORS["blue"], DARK_COLORS["green"], DARK_COLORS["red"], DARK_COLORS["amber"], DARK_COLORS["violet"]]
        for i, sid in enumerate(series_ids):
            df = atlas_series.get(sid, pd.DataFrame())
            info = atlas_info.get(sid, {})
            last_obs = str(df.index.max().date()) if not df.empty else "—"
            stale = (pd.Timestamp.now() - pd.Timestamp(df.index.max())).days if not df.empty else np.nan
            rows.append(
                {
                    "series_id": sid,
                    "title": info.get("title", ""),
                    "units": info.get("units", ""),
                    "freq": info.get("frequency", ""),
                    "last_obs": last_obs,
                    "staleness_days": stale,
                    "latest_value": safe_last(df),
                    "zscore": f"{zscore(df['value']).dropna().iloc[-1]:+.2f}" if not df.empty and not zscore(df["value"]).dropna().empty else "—",
                }
            )
            if not df.empty:
                chart_sets.append((sid, recent_months(df, lookback), palette[i % len(palette)]))
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        if chart_sets:
            st.plotly_chart(line_fig(chart_sets[:6], f"{grp} — recent history", height=340), width="stretch")

    st.markdown("#### Survey signal scorecard")
    survey_ids = MACRO_ATLAS["Surveys"]
    survey_rows = []
    for sid in survey_ids:
        df = atlas_series.get(sid, pd.DataFrame())
        if df.empty:
            continue
        zz = zscore(df["value"]).dropna()
        survey_rows.append({"series": sid, "z": zz.iloc[-1] if not zz.empty else np.nan, "pct": percentile_rank(df["value"])})
    if survey_rows:
        sdf = pd.DataFrame(survey_rows).sort_values("z")
        fig = px.bar(sdf, x="z", y="series", orientation="h", color="z", color_continuous_scale="RdYlGn", title="Surveys z-score ranking")
        fig.update_layout(paper_bgcolor=DARK_COLORS["bg"], plot_bgcolor=DARK_COLORS["panel"], font=dict(color=DARK_COLORS["text"]))
        st.plotly_chart(fig, width="stretch")

with tabs[8]:
    st.markdown("### ETF & Indices Deep Monitor")
    by_bucket = {
        "Regions": ["SPY", "VGK", "EWJ", "IEMG"],
        "Sectors": ["XLK", "XLF", "XLI", "XLV", "XLP", "XLU", "XLE", "XLB", "XLY", "XLRE", "XLC"],
        "Factors": ["QUAL", "MTUM", "USMV", "VLUE", "VUG", "IVE", "IVW"],
        "Rates/Credit": ["TLT", "IEF", "LQD", "HYG", "BIL"],
        "Commodities/FX": ["GLD", "GC=F", "BZ=F", "CL=F", "NG=F", "HG=F", "SI=F", "EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCHF=X"],
    }

    for bucket, tks in by_bucket.items():
        rows = []
        for tk in tks:
            m = metrics.get(tk)
            if not m:
                rows.append({"ticker": tk, "name": TICKERS.get(tk, tk), "status": "missing", "1M": np.nan, "3M": np.nan, "12M": np.nan, "vol": np.nan, "dd": np.nan, "pct": np.nan, "vs_ma200": np.nan})
                continue
            rows.append({
                "ticker": tk,
                "name": TICKERS.get(tk, tk),
                "status": "ok",
                "1M": m["m1"],
                "3M": m["m3"],
                "12M": m["m12"],
                "vol": m["vol"],
                "dd": m["dd"],
                "pct": m["pct"],
                "vs_ma200": m["vs_ma200"],
            })
        bdf = pd.DataFrame(rows)
        st.markdown(f"#### {bucket}")
        st.dataframe(bdf, width="stretch", hide_index=True)
        ok_df = bdf[bdf["status"] == "ok"].dropna(subset=["12M"]) if not bdf.empty else pd.DataFrame()
        if not ok_df.empty:
            fig = px.bar(ok_df.sort_values("12M"), x="12M", y="ticker", orientation="h", color="12M", color_continuous_scale="RdYlGn", title=f"{bucket} 12M performance")
            fig.update_layout(paper_bgcolor=DARK_COLORS["bg"], plot_bgcolor=DARK_COLORS["panel"], font=dict(color=DARK_COLORS["text"]))
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### Price explorer")
    selected_tickers = st.multiselect("Tickers", list(TICKERS.keys()), default=["SPY", "TLT", "GLD", "EURUSD=X"])
    datasets = []
    pal = [DARK_COLORS["blue"], DARK_COLORS["green"], DARK_COLORS["red"], DARK_COLORS["amber"], DARK_COLORS["violet"]]
    for i, tk in enumerate(selected_tickers):
        df = prices.get(tk, pd.DataFrame())
        if not df.empty:
            datasets.append((tk, recent_months(df, lookback), pal[i % len(pal)]))
    if datasets:
        st.plotly_chart(line_fig(datasets, "Selected tickers", height=380), width="stretch")
