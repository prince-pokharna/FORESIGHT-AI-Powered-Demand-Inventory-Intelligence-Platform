"""
src/risk.py
-----------
Deliverable D4 — Stockout / Overstock Risk Scoring.

Responsibilities:
  1. Compute a stockout risk score (0–1) per SKU from forecast + inventory.
  2. Compute an overstock risk score (0–1) per SKU from forecast + on-hand.
  3. Assign each SKU to one of four decisioning quadrants.
  4. Quantify the rupee value at stake for each SKU.
  5. Produce risk_scores.parquet consumed by Teammate 3's dashboard.

Design principle: every function is a pure, deterministic rule — no ML, no black box.
Any single SKU's score can be re-derived by hand in seconds.

Run with:
    python -m src.risk
"""

import math
import logging

import numpy as np
import pandas as pd

from src.config import (
    FORECAST_PATH,
    RISK_PATH,
    PANEL_PATH,
    SAFETY_STOCK_Z,
    OVERSTOCK_WINDOW_WEEKS,
    FORECAST_HORIZON_WEEKS,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[risk] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
QUADRANT_THRESHOLDS: dict = {
    "stockout_high": 0.5,   # stockout_risk above this → high stockout risk
    "overstock_high": 0.5,  # overstock_risk above this → high overstock risk
}


# ---------------------------------------------------------------------------
# Core scoring functions  (pure — no I/O, no side effects)
# ---------------------------------------------------------------------------

def compute_stockout_risk_score(
    on_hand: float,
    on_order: float,
    lead_time_days: int,
    weekly_forecasts: list[float],
    demand_std_weekly: float,
) -> float:
    """
    Stockout risk score in [0, 1].

    Logic
    -----
    lead_time_weeks        = lead_time_days / 7.0
    demand_over_lead_time  = sum(weekly_forecasts[:ceil(lead_time_weeks)])
    safety_stock           = SAFETY_STOCK_Z * demand_std_weekly * sqrt(lead_time_weeks)
    available_stock        = on_hand + on_order
    projected_gap          = demand_over_lead_time + safety_stock - available_stock

    If projected_gap <= 0  → 0.0   (stock is sufficient)
    If projected_gap >  0  → min(1.0, projected_gap / (demand_over_lead_time + safety_stock))

    Parameters
    ----------
    on_hand           : current on-hand stock units
    on_order          : units already ordered but not yet received
    lead_time_days    : replenishment lead time in days (7 / 10 / 14 / 21 / 30)
    weekly_forecasts  : list of FORECAST_HORIZON_WEEKS weekly demand forecasts
    demand_std_weekly : std of weekly demand from historical data (safety-stock buffer)

    Returns
    -------
    float in [0.0, 1.0]  — higher means greater stockout risk
    """
    lead_time_days   = int(lead_time_days)
    lead_time_weeks  = lead_time_days / 7.0
    n_lead_weeks     = math.ceil(lead_time_weeks)

    # Demand expected during the replenishment lead time
    forecasts_during_lead = weekly_forecasts[:n_lead_weeks]
    demand_over_lead_time = float(sum(forecasts_during_lead))

    # Safety-stock buffer at the configured service level
    safety_stock = SAFETY_STOCK_Z * max(demand_std_weekly, 0.0) * math.sqrt(lead_time_weeks)

    available_stock = float(on_hand) + float(on_order)
    projected_gap   = demand_over_lead_time + safety_stock - available_stock

    if projected_gap <= 0.0:
        return 0.0

    denominator = demand_over_lead_time + safety_stock
    if denominator <= 0.0:
        return 0.0

    risk_score = projected_gap / denominator
    return min(1.0, float(risk_score))


def compute_overstock_risk_score(
    on_hand: float,
    on_order: float,
    weekly_forecasts: list[float],
) -> float:
    """
    Overstock risk score in [0, 1].

    Logic
    -----
    total_forecast_demand  = sum(weekly_forecasts[:OVERSTOCK_WINDOW_WEEKS])
    available_stock        = on_hand + on_order

    If available_stock <= total_forecast_demand → 0.0   (no excess)
    Otherwise:
        excess            = available_stock - total_forecast_demand
        avg_weekly_demand = total_forecast_demand / OVERSTOCK_WINDOW_WEEKS + 0.001
        weeks_of_excess   = excess / avg_weekly_demand
        risk_score        = min(1.0, weeks_of_excess / 8.0)

    Parameters
    ----------
    on_hand           : current on-hand stock units
    on_order          : units already ordered but not yet received
    weekly_forecasts  : list of FORECAST_HORIZON_WEEKS weekly demand forecasts

    Returns
    -------
    float in [0.0, 1.0]  — higher means greater overstock risk
    """
    total_forecast_demand = float(sum(weekly_forecasts[:OVERSTOCK_WINDOW_WEEKS]))
    available_stock       = float(on_hand) + float(on_order)

    if available_stock <= total_forecast_demand:
        return 0.0

    excess             = available_stock - total_forecast_demand
    avg_weekly_demand  = total_forecast_demand / OVERSTOCK_WINDOW_WEEKS + 0.001
    weeks_of_excess    = excess / avg_weekly_demand
    risk_score         = min(1.0, weeks_of_excess / 8.0)
    return float(risk_score)


def assign_quadrant(
    stockout_risk: float,
    overstock_risk: float,
) -> tuple[str, str]:
    """
    Map risk scores to a decisioning quadrant and plain-language recommended action.

    Quadrant grid (mirrors brief Section 8.2 / Figure 6):

        ┌───────────────────┬─────────────────────────┐
        │  stockout ≥ 0.5   │  REORDER NOW            │  WATCH / VOLATILE
        │  overstock < 0.5  │  (high SO, low OS)      │  (high SO, high OS)
        ├───────────────────┼─────────────────────────┤
        │  stockout < 0.5   │  HEALTHY                │  MARKDOWN / CLEAR
        │  overstock < 0.5  │  (low SO, low OS)       │  (low SO, high OS)
        └───────────────────┴─────────────────────────┘

    Parameters
    ----------
    stockout_risk   : float in [0, 1]
    overstock_risk  : float in [0, 1]

    Returns
    -------
    (quadrant: str, recommended_action: str)
    """
    high_so = stockout_risk  >= QUADRANT_THRESHOLDS["stockout_high"]
    high_os = overstock_risk >= QUADRANT_THRESHOLDS["overstock_high"]

    if high_so and not high_os:
        return (
            "reorder_now",
            "REORDER NOW — raise replenishment order before stock runs out",
        )
    elif high_os and not high_so:
        return (
            "markdown_clear",
            "MARKDOWN / CLEAR — promote or discount to free up capital",
        )
    elif high_so and high_os:
        return (
            "watch_volatile",
            "WATCH / VOLATILE — demand is erratic, review manually",
        )
    else:
        return (
            "healthy",
            "HEALTHY — no action needed",
        )


def compute_value_at_stake(
    quadrant: str,
    stockout_risk: float,
    overstock_risk: float,
    weekly_forecasts: list[float],
    on_hand: float,
    on_order: float,
    lead_time_days: int,
    unit_cost: float,
    list_price: float,
) -> float:
    """
    Rupee value at stake for this SKU.

    Stockout-facing quadrants (reorder_now, watch_volatile):
        lead_time_weeks       = lead_time_days / 7.0
        demand_over_lead_time = sum(weekly_forecasts[:ceil(lead_time_weeks)])
        units_at_risk         = max(0, demand_over_lead_time - (on_hand + on_order))
        value_at_stake        = units_at_risk × list_price

    Overstock-facing quadrants (markdown_clear, watch_volatile):
        total_forecast        = sum(weekly_forecasts)
        excess_units          = max(0, (on_hand + on_order) - total_forecast)
        value_at_stake        = excess_units × unit_cost

    watch_volatile: whichever of the two values is larger.
    healthy:        0.0

    Parameters
    ----------
    quadrant          : one of "reorder_now", "markdown_clear", "watch_volatile", "healthy"
    stockout_risk     : float (unused in calculation but kept for signature completeness)
    overstock_risk    : float (unused in calculation but kept for signature completeness)
    weekly_forecasts  : list of weekly demand forecast values
    on_hand           : on-hand stock units
    on_order          : on-order units
    lead_time_days    : replenishment lead time in days
    unit_cost         : cost price per unit (used for overstock capital lock)
    list_price        : selling price per unit (used for stockout lost revenue)

    Returns
    -------
    Non-negative float  (rupees at stake)
    """
    if quadrant == "healthy":
        return 0.0

    available_stock   = float(on_hand) + float(on_order)
    lead_time_weeks   = int(lead_time_days) / 7.0
    n_lead_weeks      = math.ceil(lead_time_weeks)
    total_forecast    = float(sum(weekly_forecasts))

    # Stockout value
    demand_during_lead   = float(sum(weekly_forecasts[:n_lead_weeks]))
    units_at_risk        = max(0.0, demand_during_lead - available_stock)
    stockout_value       = units_at_risk * float(list_price)

    # Overstock value
    excess_units         = max(0.0, available_stock - total_forecast)
    overstock_value      = excess_units * float(unit_cost)

    if quadrant == "reorder_now":
        return max(0.0, stockout_value)
    elif quadrant == "markdown_clear":
        return max(0.0, overstock_value)
    elif quadrant == "watch_volatile":
        return max(0.0, max(stockout_value, overstock_value))
    else:
        return 0.0


# ---------------------------------------------------------------------------
# Batch scorer
# ---------------------------------------------------------------------------

def score_all_skus(
    forecasts: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply risk scoring to every SKU and return the decisioning DataFrame.

    Steps
    -----
    1. Extract most-recent inventory position per SKU from panel.
    2. Extract static SKU attributes per SKU from panel.
    3. Compute historical weekly demand std per SKU (last 52 weeks).
    4. For each SKU: score stockout risk, overstock risk, quadrant, value.
    5. Build output DataFrame, sort by value_at_stake_inr descending.
    6. Log summary and save to RISK_PATH.

    Output columns (exact — Teammate 3 depends on these names):
        sku_id, category, subcategory, unit_cost, list_price,
        on_hand_units, on_order_units, lead_time_days, reorder_point,
        stockout_risk, overstock_risk, quadrant, recommended_action,
        value_at_stake_inr, forecast_8w_total

    Parameters
    ----------
    forecasts : output of generate_final_forecast() filtered to is_future==True
    panel     : full daily panel from pipeline.py

    Returns
    -------
    pd.DataFrame  (also saved to RISK_PATH as parquet)
    """
    # ---- 1. Most-recent inventory position per SKU --------------------------
    panel_sorted = panel.sort_values(["sku_id", "date"])
    latest_inv = (
        panel_sorted.groupby("sku_id", sort=False)
        .agg(
            on_hand_units  =("on_hand_units",  "last"),
            on_order_units =("on_order_units", "last"),
            lead_time_days =("lead_time_days", "first"),   # constant per SKU
            reorder_point  =("reorder_point",  "first"),
        )
        .reset_index()
    )

    # ---- 2. Static SKU attributes -------------------------------------------
    static_cols = ["sku_id", "unit_cost", "list_price", "category", "subcategory"]
    # category may be Categorical dtype — cast to str for safe merging
    panel_static = panel[static_cols].copy()
    panel_static["category"] = panel_static["category"].astype(str)
    sku_static = (
        panel_static.drop_duplicates(subset="sku_id")
        .set_index("sku_id")
    )

    # ---- 3. Historical weekly demand std per SKU (last 52 weeks) -------------
    panel["date"] = pd.to_datetime(panel["date"])
    cutoff_52w    = panel["date"].max() - pd.Timedelta(weeks=52)
    recent_panel  = panel[panel["date"] >= cutoff_52w].copy()
    recent_panel["week_start"] = (
        recent_panel["date"] - pd.to_timedelta(recent_panel["date"].dt.dayofweek, unit="D")
    )
    weekly_std = (
        recent_panel.groupby(["sku_id", "week_start"])["units_sold"]
        .sum()
        .groupby("sku_id")
        .std()
        .fillna(1.0)
        .rename("demand_std_weekly")
        .reset_index()
    )

    # ---- 4. Score each SKU --------------------------------------------------
    rows: list[dict] = []
    sku_list = forecasts["sku_id"].unique().tolist()

    # Build a lookup dict for faster per-SKU forecast access
    fcst_lookup: dict[str, list[float]] = (
        forecasts.sort_values("week_start")
        .groupby("sku_id")["yhat"]
        .apply(list)
        .to_dict()
    )

    inv_lookup = latest_inv.set_index("sku_id").to_dict(orient="index")
    std_lookup  = weekly_std.set_index("sku_id")["demand_std_weekly"].to_dict()

    for sku in sku_list:
        if sku not in sku_static.index:
            logger.warning("SKU %s not found in sku_static — skipping.", sku)
            continue

        inv         = inv_lookup.get(sku, {})
        on_hand     = float(inv.get("on_hand_units",  0.0) or 0.0)
        on_order    = float(inv.get("on_order_units", 0.0) or 0.0)
        lead_time   = int(inv.get("lead_time_days",  14)  or 14)
        reorder_pt  = float(inv.get("reorder_point",   0.0) or 0.0)

        unit_cost   = float(sku_static.at[sku, "unit_cost"]   or 0.0)
        list_price  = float(sku_static.at[sku, "list_price"]  or 0.0)
        category    = str(sku_static.at[sku, "category"])
        subcategory = str(sku_static.at[sku, "subcategory"])

        wkly_fcst     = fcst_lookup.get(sku, [0.0] * FORECAST_HORIZON_WEEKS)
        # Ensure exactly FORECAST_HORIZON_WEEKS entries
        if len(wkly_fcst) < FORECAST_HORIZON_WEEKS:
            wkly_fcst = wkly_fcst + [0.0] * (FORECAST_HORIZON_WEEKS - len(wkly_fcst))
        wkly_fcst = wkly_fcst[:FORECAST_HORIZON_WEEKS]

        demand_std   = float(std_lookup.get(sku, 1.0))
        forecast_8w  = float(sum(wkly_fcst))

        # Score
        so_risk  = compute_stockout_risk_score(on_hand, on_order, lead_time,
                                               wkly_fcst, demand_std)
        os_risk  = compute_overstock_risk_score(on_hand, on_order, wkly_fcst)
        quadrant, action = assign_quadrant(so_risk, os_risk)
        value    = compute_value_at_stake(
            quadrant, so_risk, os_risk, wkly_fcst,
            on_hand, on_order, lead_time, unit_cost, list_price,
        )

        rows.append({
            "sku_id":              sku,
            "category":            category,
            "subcategory":         subcategory,
            "unit_cost":           unit_cost,
            "list_price":          list_price,
            "on_hand_units":       on_hand,
            "on_order_units":      on_order,
            "lead_time_days":      lead_time,
            "reorder_point":       reorder_pt,
            "stockout_risk":       round(so_risk, 6),
            "overstock_risk":      round(os_risk, 6),
            "quadrant":            quadrant,
            "recommended_action":  action,
            "value_at_stake_inr":  round(value, 2),
            "forecast_8w_total":   round(forecast_8w, 2),
        })

    output = pd.DataFrame(rows)
    output.sort_values("value_at_stake_inr", ascending=False, inplace=True)
    output.reset_index(drop=True, inplace=True)

    # ---- 5. Log summary ------------------------------------------------------
    q_counts = output["quadrant"].value_counts()
    n_reorder   = int(q_counts.get("reorder_now",    0))
    n_markdown  = int(q_counts.get("markdown_clear", 0))
    n_watch     = int(q_counts.get("watch_volatile", 0))
    n_healthy   = int(q_counts.get("healthy",        0))

    stockout_qs = ["reorder_now", "watch_volatile"]
    overstock_qs = ["markdown_clear", "watch_volatile"]
    total_stockout_val  = output[output["quadrant"].isin(stockout_qs)]["value_at_stake_inr"].sum()
    total_overstock_val = output[output["quadrant"].isin(overstock_qs)]["value_at_stake_inr"].sum()

    logger.info("Risk scoring complete. %d SKUs scored.", len(output))
    logger.info("  Reorder now:    %d SKUs", n_reorder)
    logger.info("  Markdown/clear: %d SKUs", n_markdown)
    logger.info("  Watch/volatile: %d SKUs", n_watch)
    logger.info("  Healthy:        %d SKUs", n_healthy)
    logger.info("  Total ₹ at risk (stockout):  ₹%s", f"{total_stockout_val:,.0f}")
    logger.info("  Total ₹ locked (overstock):  ₹%s", f"{total_overstock_val:,.0f}")

    # ---- save ----------------------------------------------------------------
    from pathlib import Path
    Path(RISK_PATH).parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(RISK_PATH, index=False)

    return output


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Orchestrates:
      1. Load forecasts (future weeks only: is_future == True)
      2. Load panel
      3. Call score_all_skus()
      4. Log completion
    """
    from pathlib import Path

    logger.info("Loading forecasts from %s ...", FORECAST_PATH)
    forecasts_all = pd.read_parquet(FORECAST_PATH)
    forecasts = forecasts_all[forecasts_all["is_future"] == True].copy()
    logger.info("  Loaded %d future-week forecast rows for %d SKUs.",
                len(forecasts), forecasts["sku_id"].nunique())

    logger.info("Loading panel from %s ...", PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)

    score_all_skus(forecasts, panel)
    logger.info("Risk scoring saved to %s", RISK_PATH)


if __name__ == "__main__":
    run()