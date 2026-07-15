# Data Quality & EDA Insight Memo
**Project FORESIGHT — NorthBay Living**
**Prepared by:** [Your name]
**Date:** [Date]

---

## 1. Data Quality Findings

### 1.1 sales_daily.csv

| Issue | Count | Action Taken |
|---|---|---|
| Duplicate rows | [from pipeline log] | Removed exact duplicates, kept first |
| Missing `units_sold` | [from pipeline log] | Rows dropped — imputing demand is unreliable |
| Zero / negative `units_sold` | [from pipeline log] | Rows dropped — not valid transactions |
| Missing `revenue` | [from pipeline log] | Imputed as `units_sold × unit_price` |
| Zero / negative `unit_price` | [from pipeline log] | Rows dropped — invalid price records |
| Pre-launch rows | [from pipeline log] | Excluded based on `launch_date` from sku_master |

> **How to fill these in:** run `python -m src.pipeline` and copy the `[pipeline]` log lines
> into the Count column above. The log prints counts at every step.

### 1.2 sku_master.csv

| Issue | Count | Action Taken |
|---|---|---|
| Duplicate SKU rows | 4 | Removed, kept first occurrence |
| Inconsistent category labels | 12 rows | Normalized to 4 canonical categories (see below) |

**Category label normalization applied:**

| Raw label found | Canonical label |
|---|---|
| `furniture` | `Furniture` |
| `DECOR` | `Decor` |
| `decor` | `Decor` |
| `Small Appliance` | `Small Appliances` |
| `Bedding and Bath` | `Bedding & Bath` |

### 1.3 calendar.csv

No significant quality issues found.
Empty `promo_event` values filled with `""` (empty string) for consistent filtering.

### 1.4 inventory_snapshots.csv

No quality issues found.
Grain is weekly (one snapshot per SKU every 7 days). Forward-filled to daily grain
in the pipeline join step so the panel has a daily inventory position per SKU.

---

## 2. Demand Patterns

### 2.1 Weekly Seasonality (Day of Week)

[Describe the day-of-week pattern from `notebooks/01_eda.ipynb` Section 5 chart.]

Example: "Weekend demand (Saturday and Sunday) is approximately [X]% higher than the
weekday average, consistent with home goods purchases being a weekend activity."

### 2.2 Yearly Seasonality (Monthly Pattern)

[Describe the monthly pattern from Section 5 chart — specifically note the Oct/Nov/Dec lift.]

Example: "October, November, and December show clearly elevated demand, with October
peaking at approximately [X]% above the annual average — driven by the Diwali Sale
(Oct 18–28) and the broader festive season shopping window."

### 2.3 Promo Effect

Promotional periods (10 windows across 2 years: New Year Sale, Summer Kickoff,
Festive Pre-Sale, Diwali Sale, Year-End Clearance) drive approximately
**+[X]%** average daily demand compared to non-promo periods.

[Fill from `notebooks/01_eda.ipynb` Section 6 output — the printed lift_pct value.]

### 2.4 SKU Demand Distribution (Long Tail)

The dataset shows a classic long-tail distribution: the top **[N]** SKUs account
for 80% of total revenue, while the remaining **[M]** SKUs are slow movers with
intermittent or near-zero demand.

[Fill from Section 4 of the notebook — top 10 SKU revenue concentration.]

---

## 3. Business Insights

**These three insights are written in plain language for the Head of Operations.**

1. **Festive season demand spike is unaccounted for in current reorder points.**
   October–December drives approximately **[X]%** higher daily demand vs the
   rest of the year across Furniture and Decor SKUs. Current static reorder
   points were set without this seasonal uplift in mind, which likely explains
   the repeat stockouts the Ops team experiences each festive season. The
   FORESIGHT forecast accounts for this pattern explicitly.

2. **Diwali Sale is the single highest-impact demand event.**
   The Diwali Sale window (Oct 18–28) generates approximately **+[X]%** average
   demand lift. SKUs with lead times of 14 days or more need replenishment
   orders placed at least 3 weeks before Oct 18 to avoid stocking out during
   the peak window. The risk scoring system flags these SKUs by name.

3. **[N] SKUs have >90 days of inventory cover — capital is locked.**
   Based on the past 30 days of sales, **[N]** SKUs are selling slowly enough
   that their current on-hand stock would last more than 90 days. This represents
   approximately **₹[X]** of working capital that could be freed through targeted
   markdowns or clearance promotions. These SKUs appear in the
   **Markdown / Clear** quadrant of the FORESIGHT dashboard.

---

## 4. Baseline Forecast Performance

The seasonal-naive baseline (predict same week last year) was evaluated on a
one-fold backtest using data up to 2025-10-01, with the next 8 weeks as the
test window.

| Metric | Value |
|---|---|
| Baseline WAPE | **[paste from `notebooks/02_baseline.ipynb` Section 3]%** |
| Baseline Bias | **[paste from notebook]** units/week |

This WAPE is the accuracy bar the LightGBM model (Teammate 2) must beat.
A model whose WAPE is lower than this number is reliable enough to act on.
If LightGBM does not beat this bar for certain SKU segments, those segments
fall back to the seasonal-naive baseline — and that finding is reported
honestly in the executive readout (brief Section 7.1).