"""
src/baseline.py
---------------
Seasonal-naive demand forecast.

This is the bar every model must beat (brief Section 7.1).
Logic: for forecast week w, predict the actual demand from the same week
       one full season (52 weeks) ago.

Leakage guarantee: only rows where date < as_of_date are ever read.
"""

import logging

import numpy as np
import pandas as pd

from src.config import FORECAST_HORIZON_WEEKS, SEASON_LENGTH_WEEKS

logging.basicConfig(level=logging.INFO, format="[baseline] %(message)s")
logger = logging.getLogger(__name__)


def seasonal_naive_forecast(
    panel: pd.DataFrame,
    sku_id: str,
    as_of_date: pd.Timestamp,
    horizon_weeks: int = FORECAST_HORIZON_WEEKS,
    season_length_weeks: int = SEASON_LENGTH_WEEKS,
) -> pd.DataFrame:
    """
    Seasonal-naive forecast for one SKU.

    For each future week w in [1 .. horizon_weeks]:
        target_week  = week starting on (as_of_date_monday + w weeks)
        lookback_week = target_week - season_length_weeks weeks
        baseline_yhat = mean daily units_sold during lookback_week × 7

    Uses ONLY rows where date < as_of_date — leakage is impossible by
    construction because we filter before any computation.

    Fallback: if fewer than season_length_weeks weeks of history exist
    before as_of_date, predict the mean of all available weekly demand
    for that SKU. Logs a warning.

    Parameters
    ----------
    panel                : full panel DataFrame (must have date, sku_id, units_sold)
    sku_id               : which SKU to forecast
    as_of_date           : training cutoff — only data strictly before this is used
    horizon_weeks        : number of weeks to forecast ahead (default 8)
    season_length_weeks  : look-back season length in weeks (default 52)

    Returns
    -------
    pd.DataFrame with columns:
        week_number   int   (1-indexed)
        week_start    Timestamp  (Monday of each forecast week)
        sku_id        str
        baseline_yhat float  (weekly units — sum over 7 days)
    """
    as_of_date = pd.Timestamp(as_of_date)

    # ---- filter to past only (leakage guard) --------------------------------
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    past = df[(df["sku_id"] == sku_id) & (df["date"] < as_of_date)].copy()

    # ---- build weekly aggregation from daily history -----------------------
    past["week_start"] = (
        past["date"] - pd.to_timedelta(past["date"].dt.dayofweek, unit="D")
    )
    weekly_hist = (
        past.groupby("week_start")["units_sold"]
        .sum()
        .reset_index()
        .rename(columns={"units_sold": "weekly_units"})
    )
    weekly_hist["week_start"] = pd.to_datetime(weekly_hist["week_start"])
    weekly_hist.sort_values("week_start", inplace=True)

    n_hist_weeks = len(weekly_hist)
    fallback = n_hist_weeks < season_length_weeks
    if fallback:
        logger.warning(
            "SKU %s: only %d weeks of history before %s "
            "(need %d for seasonal-naive). Using mean-demand fallback.",
            sku_id, n_hist_weeks, as_of_date.date(), season_length_weeks,
        )
        fallback_val = float(weekly_hist["weekly_units"].mean()) if n_hist_weeks > 0 else 0.0

    # as_of_date's own week-Monday + 1 week = first forecast week
    as_of_monday = as_of_date - pd.to_timedelta(as_of_date.dayofweek, unit="D")
    first_forecast_monday = as_of_monday + pd.Timedelta(weeks=1)

    hist_lookup = weekly_hist.set_index("week_start")["weekly_units"]

    rows: list[dict] = []
    for w in range(1, horizon_weeks + 1):
        forecast_week_start = first_forecast_monday + pd.Timedelta(weeks=w - 1)
        lookback_week_start = forecast_week_start - pd.Timedelta(weeks=season_length_weeks)

        if fallback:
            yhat = fallback_val
        elif lookback_week_start in hist_lookup.index:
            yhat = float(hist_lookup[lookback_week_start])
        else:
            # Exact week not found — use nearest available weekly mean
            yhat = float(hist_lookup.mean()) if len(hist_lookup) > 0 else 0.0

        rows.append(
            {
                "week_number":   w,
                "week_start":    forecast_week_start,
                "sku_id":        sku_id,
                "baseline_yhat": max(0.0, yhat),
            }
        )

    return pd.DataFrame(rows)


def run_baseline_all_skus(
    panel: pd.DataFrame,
    as_of_date: pd.Timestamp,
    horizon_weeks: int = FORECAST_HORIZON_WEEKS,
) -> pd.DataFrame:
    """
    Run seasonal_naive_forecast for every unique sku_id in the panel.

    Parameters
    ----------
    panel         : full panel DataFrame
    as_of_date    : training cutoff (same for all SKUs)
    horizon_weeks : weeks ahead to forecast

    Returns
    -------
    Concatenated DataFrame of all SKU forecasts.
    """
    sku_list = panel["sku_id"].unique().tolist()
    frames: list[pd.DataFrame] = []

    for sku in sku_list:
        frames.append(
            seasonal_naive_forecast(panel, sku, as_of_date, horizon_weeks)
        )

    return pd.concat(frames, ignore_index=True)