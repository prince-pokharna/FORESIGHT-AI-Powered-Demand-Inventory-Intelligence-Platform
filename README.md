# Project FORESIGHT — Demand & Inventory Intelligence
**Client:** NorthBay Living · **Built by:** [Team member names]

---

## Problem

NorthBay Living, a D2C home & lifestyle brand with approximately 200 active SKUs,
plans inventory on gut feel and spreadsheets. They lose revenue from stockouts of
best-sellers and lock capital in slow-moving overstock that later gets marked down.

This project delivers a demand forecast and early-warning system so the planning
team knows exactly what to reorder, what to clear, and what to leave alone —
without needing a data scientist in the room.

---

## Data

Four simulated extracts modelled on what a real D2C brand would have:

| File | Grain | Rows | Key columns |
|---|---|---|---|
| `sales_daily.csv` | SKU × day | ~135,000 | units_sold, revenue, promo_flag |
| `sku_master.csv` | SKU | 200 | category, unit_cost, list_price, launch_date |
| `calendar.csv` | date | 730 | season, is_holiday, promo_event |
| `inventory_snapshots.csv` | SKU × week | 21,000 | on_hand_units, lead_time_days |

Date range: 2024-01-01 to 2025-12-30 (2 years of daily history).

---

## Setup & Run

```bash
git clone <your-repo-url>
cd foresight

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Place the 4 raw CSVs in data/raw/
python -m src.pipeline           # → data/processed/panel.parquet
python -m src.forecast           # → data/processed/forecasts.parquet
python -m src.risk               # → data/processed/risk_scores.parquet
streamlit run app/main.py        # → open http://localhost:8501
```

### Run tests

```bash
pytest tests/ -v
```

---

## Results

| Metric | Value |
|---|---|
| Baseline (seasonal-naive) WAPE | **[FILL IN after running 02_baseline.ipynb]%** |
| LightGBM forecast WAPE | **[FILL IN after running 03_model.ipynb]%** |
| Improvement over baseline | **[FILL IN]%** |

> These numbers come from rolling-origin cross-validation (5 folds).
> No future data touched the training features — zero leakage.

---

## Key Assumptions

- **90% service level** used for safety stock calculations (z = 1.28).
- **8-week forecast horizon** — matches typical NorthBay planning cadence.
- **Inventory snapshots are weekly.** Forward-filled to daily grain in the pipeline.
- Pre-launch sales rows excluded (date < SKU launch_date).
- Missing `units_sold` rows dropped rather than imputed — imputing demand is unreliable.

---

## Repository Structure

```
foresight/
├── data/
│   ├── raw/                         ← 4 CSV extracts (gitignored)
│   └── processed/                   ← pipeline output parquet files (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb                 ← EDA and data quality analysis (D2)
│   ├── 02_baseline.ipynb            ← Seasonal-naive baseline evaluation
│   └── 03_model.ipynb               ← LightGBM backtest and evaluation
├── src/
│   ├── config.py                    ← all constants and file paths
│   ├── pipeline.py                  ← ingest + clean + join (D1)
│   ├── features.py                  ← lag/rolling/calendar feature engineering
│   ├── metrics.py                   ← WAPE, MAPE, bias functions
│   ├── baseline.py                  ← seasonal-naive forecast
│   ├── forecast.py                  ← LightGBM forecast + backtest (D3)
│   └── risk.py                      ← stockout/overstock risk scoring (D4)
├── app/
│   ├── main.py                      ← Streamlit entry point
│   ├── pages/                       ← overview, forecast, risk, scoring pages (D5, D6)
│   └── utils/loaders.py             ← cached data loaders
├── reports/
│   ├── data_quality_eda_memo.md     ← D2 deliverable
│   └── executive_readout.md         ← D7 stakeholder readout
├── tests/
│   ├── test_metrics.py
│   ├── test_baseline.py
│   └── test_risk.py
├── .gitignore
├── requirements.txt
├── DEPLOY.md                        ← Streamlit Cloud deployment steps
└── README.md
```

---

## Live Links

| Resource | URL |
|---|---|
| Dashboard | [FILL IN after Streamlit Cloud deployment] |
| Scoring service | [FILL IN — same app, add `?sku_id=SKU0042` to URL] |

---

## Deliverables

| # | Deliverable | Status |
|---|---|---|
| D1 | Reproducible data pipeline | ✅ `python -m src.pipeline` |
| D2 | Data quality & EDA memo | `reports/data_quality_eda_memo.md` |
| D3 | Demand forecast model | `python -m src.forecast` |
| D4 | Risk scoring | `python -m src.risk` |
| D5 | Planning dashboard | `streamlit run app/main.py` |
| D6 | Deployed scoring service | `[live URL]?sku_id=SKU0001` |
| D7 | Executive readout | `reports/executive_readout.md` |