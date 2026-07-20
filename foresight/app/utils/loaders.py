"""
app/utils/loaders.py
--------------------
Centralised data loading for the FORESIGHT Streamlit app.

ALL data reads go through this file.
The @st.cache_data decorator ensures Parquet files are read from disk only
once per session — not on every widget interaction.

Paths are hardcoded as strings relative to the repo root so this module
works identically in local dev and on Streamlit Community Cloud.
Do NOT import from src.config here — that import path is fragile on Cloud.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# File paths — relative to repo root, no src.config dependency
# ---------------------------------------------------------------------------
FORECAST_PATH = Path("data/processed/forecasts.parquet")
RISK_PATH     = Path("data/processed/risk_scores.parquet")
PANEL_PATH    = Path("data/processed/panel.parquet")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_forecasts() -> pd.DataFrame:
    """
    Load the 8-week demand forecast for all 200 SKUs.

    Columns returned:
        sku_id          object
        week_start      datetime64[ns]
        yhat            float64   (point forecast, weekly units)
        yhat_lower_80   float64   (10th-percentile interval)
        yhat_upper_80   float64   (90th-percentile interval)
        baseline_yhat   float64   (seasonal-naive baseline)
        is_future       bool      (True = forecast, False = historical fit)

    Cached until the app restarts or the cache is cleared.
    """
    if not FORECAST_PATH.exists():
        st.error(
            f"forecasts.parquet not found at `{FORECAST_PATH}`. "
            "Ask Teammate 2 to run `python -m src.forecast` and commit the output."
        )
        st.stop()
    df = pd.read_parquet(FORECAST_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])
    return df


@st.cache_data
def load_risk_scores() -> pd.DataFrame:
    """
    Load the risk scores and recommended actions for all 200 SKUs.

    Columns returned:
        sku_id              object
        category            object   (Furniture / Decor / Small Appliances / Bedding & Bath)
        subcategory         object
        unit_cost           float64
        list_price          float64
        on_hand_units       float64
        on_order_units      float64
        lead_time_days      float64
        reorder_point       float64
        stockout_risk       float64  (0–1)
        overstock_risk      float64  (0–1)
        quadrant            object   (reorder_now / markdown_clear / watch_volatile / healthy)
        recommended_action  object
        value_at_stake_inr  float64
        forecast_8w_total   float64

    Cached until the app restarts or the cache is cleared.
    """
    if not RISK_PATH.exists():
        st.error(
            f"risk_scores.parquet not found at `{RISK_PATH}`. "
            "Ask Teammate 2 to run `python -m src.risk` and commit the output."
        )
        st.stop()
    return pd.read_parquet(RISK_PATH)


@st.cache_data
def load_panel_history(last_n_weeks: int = 12) -> pd.DataFrame:
    """
    Load the last N weeks of the daily panel for historical demand charts.

    Only reads three columns (sku_id, date, units_sold) to keep memory low.
    Aggregates daily rows to weekly grain before returning.

    Parameters
    ----------
    last_n_weeks : int
        How many weeks of history to return (default 12).

    Returns
    -------
    pd.DataFrame with columns:
        sku_id            object
        week_start        datetime64[ns]   (Monday of each week)
        units_sold_weekly float64          (sum of daily units for that week)
    """
    if not PANEL_PATH.exists():
        st.error(
            f"panel.parquet not found at `{PANEL_PATH}`. "
            "Ask Teammate 1 to run `python -m src.pipeline` and commit the output."
        )
        st.stop()

    panel = pd.read_parquet(PANEL_PATH, columns=["sku_id", "date", "units_sold"])
    panel["date"] = pd.to_datetime(panel["date"])

    cutoff = panel["date"].max() - pd.Timedelta(weeks=last_n_weeks)
    panel  = panel[panel["date"] >= cutoff].copy()

    # Monday-aligned week key
    panel["week_start"] = (
        panel["date"] - pd.to_timedelta(panel["date"].dt.dayofweek, unit="D")
    )

    weekly = (
        panel.groupby(["sku_id", "week_start"])["units_sold"]
        .sum()
        .reset_index()
        .rename(columns={"units_sold": "units_sold_weekly"})
    )
    weekly["week_start"] = pd.to_datetime(weekly["week_start"])
    return weekly


def get_sku_list(risk_scores: pd.DataFrame) -> list[str]:
    """
    Return all SKU IDs sorted by value_at_stake_inr descending.

    The most financially impactful SKUs appear first in every dropdown,
    so the Ops team immediately sees what matters most.

    Parameters
    ----------
    risk_scores : pd.DataFrame
        Output of load_risk_scores().

    Returns
    -------
    list[str]  e.g. ["SKU0042", "SKU0017", ..., "SKU0180"]
    """
    return (
        risk_scores
        .sort_values("value_at_stake_inr", ascending=False)["sku_id"]
        .unique()
        .tolist()
    )
