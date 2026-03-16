# asset-regime

Cross-asset Streamlit dashboard (FRED + Yahoo only) with institutional-grade governance and robust fallbacks.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

This repo includes `requirements.txt`; Cloud installs all runtime dependencies before launching `app.py`.

## Data and model design

- Data sources: **FRED** (`fredapi`, API key from `st.secrets["FRED_API_KEY"]` or sidebar) and **Yahoo Finance** (`yfinance`) only.
- Deterministic series picker per `(region, concept)` with:
  - curated candidate lists,
  - fallback FRED search,
  - scoring by frequency, staleness, missingness, and history length.
- Transformation rules by series type:
  - `LEVEL`: YoY + rolling z-score.
  - `RATE`: level z-score + 3m / 12m change (**not YoY**).
  - `SPREAD`: level z-score.
  - `INDEX`: level z-score + 3m change.
- Canonical US curve schema is always `['2Y','10Y','Slope']`.
- Regime engine uses effective sample after alignment/cleaning; if insufficient, it switches to a clearly-labeled proxy regime.
- Allocation optimizer sanitizes returns and falls back deterministically to anchor weights when optimization constraints are infeasible.

## CHANGELOG (major fixes)

1. **Regime inconsistency fixed (raw vs effective sample)**
   - Regime availability now depends on **effective aligned monthly sample**, not raw source length.
2. **Curve engine hardened**
   - `DGS2` and `DGS10` both fetched explicitly.
   - Curve dataframe always built with canonical columns `['2Y','10Y','Slope']` or returns a clear unavailability message.
3. **Optimizer crash-proofing**
   - Added strict `sanitize_returns` handling for NaN/inf/insufficient rows.
   - Added deterministic fallback mode with explicit reason and binding-constraints report.
4. **Europe-leading and stale-series handling improved**
   - Expanded candidate lists and deterministic fallback search.
   - Selection score heavily penalizes stale series.
5. **Transformation logic corrected**
   - Rates/labor yields are no longer transformed as YoY.
6. **Data governance UI added**
   - Data dictionary completeness, staleness, missingness, source mode, exclusion reason, and dropped-series transparency shown in Data Quality tab.
