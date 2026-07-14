# FORESIGHT — Executive Readout
**NorthBay Living · Demand & Inventory Intelligence**
**Prepared for:** Head of Operations, Finance Lead
**Prepared by:** FORESIGHT Data Science Team (Zidio Internship Cohort)
**Date:** [Date]

---

## Situation

NorthBay Living stocks approximately 200 active SKUs across Furniture, Decor,
Small Appliances, and Bedding & Bath. Today, inventory planning relies on gut
feel and spreadsheets — with no systematic demand forecast in place.

This creates two simultaneous and measurable problems:

- **Best-sellers run out** — lost sales that can never be recovered and customers
  who go elsewhere.
- **Slow movers pile up** — working capital locked in stock that eventually gets
  marked down, eroding margin.

We were engaged to turn NorthBay's own sales and inventory data into a demand
forecast and early-warning system: telling the planning team exactly what to
reorder, what to clear, and what to leave alone.

---

## What We Found — The Rupee Impact

| Situation | SKUs | ₹ Value |
|---|---|---|
| At risk of stocking out in next 8 weeks | [FILL_IN] | ₹[FILL_IN] |
| Overstocked — capital locked, markdown risk | [FILL_IN] | ₹[FILL_IN] |
| Demand volatile — needs manual review | [FILL_IN] | — |
| Healthy — no action needed | [FILL_IN] | — |

> **Fill in these numbers** by reading the Overview page of the live dashboard
> or from `data/processed/risk_scores.parquet` after running the pipeline.

**Top 3 SKUs by rupee impact:**

1. [FILL_IN SKU ID] — [Category] — ₹[FILL_IN] at stake — **Action: [FILL_IN]**
2. [FILL_IN SKU ID] — [Category] — ₹[FILL_IN] at stake — **Action: [FILL_IN]**
3. [FILL_IN SKU ID] — [Category] — ₹[FILL_IN] at stake — **Action: [FILL_IN]**

> Get these from `risk_scores.parquet` sorted by `value_at_stake_inr` descending,
> or from the Priority Action List on the dashboard Overview page.

---

## The Forecast — How Accurate Is It?

We built a weekly demand forecast for all 200 SKUs over the next 8 weeks using
a machine learning model (LightGBM). Before trusting it, we compared it against
a simple benchmark: *"predict the same demand as this time last year"*
(seasonal-naive baseline). A model that cannot beat this baseline is not worth
acting on — and we report the result honestly either way.

| Method | Forecast Error (WAPE) |
|---|---|
| Seasonal-naive baseline (benchmark) | [FILL_IN from Teammate 2]% |
| LightGBM model (our forecast) | [FILL_IN from Teammate 2]% |
| Improvement over baseline | [FILL_IN]% better |

> WAPE = Weighted Absolute Percentage Error. **Lower is better.**
> Fill in from the Section 5 output of `notebooks/03_model.ipynb`.

**What this means in plain terms:**
The LightGBM model reduces forecast error by approximately [FILL_IN]% compared
to simply using last year's demand. This improvement is what makes the stockout
and overstock flags reliable enough to act on, rather than being random noise.

The forecast also provides an **80% prediction interval** for each SKU — a range
within which actual demand is expected to fall 8 times out of 10. The dashboard
shows this interval as a shaded band, so the Ops team can see both the central
forecast and how confident the model is.

---

## Recommended Actions (as of [Date])

### This week — Reorder

Order stock now for the **[FILL_IN] SKUs** flagged **REORDER NOW**.
Lead times on these SKUs range from 7 to 30 days. A delayed order equals a
guaranteed stockout.

**Total revenue exposure if you wait: ₹[FILL_IN].**

Priority reorder list is available as a downloadable CSV from the dashboard
**Risk & Actions → 🔴 Reorder Now** tab.

### This month — Clear Overstock

Run promotions or targeted discounts on the **[FILL_IN] SKUs** flagged
**MARKDOWN / CLEAR**.

**Capital that could be freed: ₹[FILL_IN].**

Markdown candidate list is available as a downloadable CSV from the dashboard
**Risk & Actions → 🔵 Markdown / Clear** tab.

### Ongoing — Watch List

**[FILL_IN] SKUs** show volatile or erratic demand patterns — both high stockout
and high overstock risk simultaneously. These should be reviewed manually before
each ordering cycle. Do not rely solely on the model's recommendation for these
SKUs.

### No Action Needed

**[FILL_IN] SKUs** are in a healthy position — adequate stock relative to
forecast demand, no immediate risk in either direction.

---

## How to Use the Dashboard

The FORESIGHT dashboard is live at: **[FILL_IN URL after Streamlit Cloud deployment]**

**Step-by-step for the Ops team:**

1. Open the dashboard URL in any browser — no login required.
2. The **Overview** page loads immediately with the full SKU decisioning grid
   and a priority action table sorted by ₹ at stake.
3. Use the **Category** filter at the top to narrow to Furniture, Decor, or
   any other category.
4. Click **Risk & Actions** in the sidebar to see four tabs:
   - 🔴 Reorder Now — download this list and send to purchasing
   - 🔵 Markdown / Clear — download for the promotions team
   - 🟠 Watch / Volatile — review these manually
   - 🟢 Healthy — no action needed
5. Click **Forecast** to see the full 8-week forecast chart for any individual
   SKU, including the uncertainty band and the baseline comparison.
6. To look up a single SKU instantly, use **Score a SKU** — type the SKU ID
   or visit `[dashboard URL]?sku_id=SKU0042`.

**No data science background is needed to use this dashboard.**

---

## How to Refresh the Data

When NorthBay has a new data export, the pipeline can be re-run with a single
command sequence:

```bash
python -m src.pipeline   # re-cleans and re-joins the raw extracts
python -m src.forecast   # re-trains the model and regenerates forecasts
python -m src.risk       # re-scores all SKUs
```

After re-running, commit the updated Parquet files to the repository and
Streamlit Cloud will automatically redeploy with the new numbers within
approximately 2 minutes.

---

## Limitations

**1. The forecast is based on historical patterns.**
Unusual events — a new product launch, a supply-chain disruption, a price
change, or an unplanned promotion — will not be captured automatically.
The planning team should override the system's recommendations when they have
information the data does not.

**2. Safety stock assumes a 90% service level.**
This means roughly 1 in 10 replenishment cycles may still experience a brief
stockout even when the REORDER NOW flag is acted on immediately. If NorthBay
requires a higher service level (e.g., 95%), the threshold can be adjusted in
`src/config.py` (change `SAFETY_STOCK_Z` from 1.65 to 1.96) and the pipeline
rerun.

**3. Data was last refreshed on [date].**
Recommendations are only as current as the data. Stale data produces stale
recommendations. We recommend refreshing the pipeline at least monthly, or
whenever a significant demand event (promotion, new SKU launch) occurs.

**4. Inventory snapshots are weekly.**
Daily stock movements between snapshots are estimated by forward-filling the
last known position. This is standard practice but introduces some imprecision
in the risk scores for very fast-moving SKUs with high daily velocity.

**5. New SKUs with less than one year of history.**
The LightGBM model uses a 52-week seasonal lag as a key feature. SKUs launched
within the past year fall back to a category-level average for that feature.
Their forecasts are less reliable than for established SKUs and should be
treated with additional caution.

---

## Appendix — Technical Summary

| Item | Detail |
|---|---|
| Forecast model | LightGBM (gradient-boosted trees) |
| Training data | 2 years of daily SKU-level sales (Jan 2024 – Dec 2025) |
| Forecast grain | Weekly, per SKU |
| Forecast horizon | 8 weeks |
| Accuracy metric | WAPE (Weighted Absolute Percentage Error) |
| Baseline compared | Seasonal-naive (same week last year) |
| Backtest method | 5-fold rolling-origin cross-validation (no data leakage) |
| Risk scoring | Rule-based (transparent, not ML) |
| Service level | 90% (z = 1.65) |
| Dashboard | Streamlit Community Cloud |
| Pipeline | Fully automated, single-command re-run |

---

*Delivered by the FORESIGHT team as part of the Zidio Development Data Science Internship.*
*Project FORESIGHT — Demand & Inventory Intelligence · Document v1.0*