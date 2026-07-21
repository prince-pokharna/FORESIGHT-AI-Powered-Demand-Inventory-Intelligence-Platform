"""
src/features.py
---------------
Feature engineering for the FORESIGHT demand forecast model.

All features are computed per SKU group (groupby("sku_id")) so they never
bleed across SKUs — the fundamental leakage guard at the daily grain level.

The weekly-grain features used by LightGBM are built inside src/forecast.py
(add_weekly_features). This module provides the daily-grain features and
the canonical feature-column list shared across the project.
"""

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED   

# Season string → integer encoding  (must match src/forecast.py _SEASON_MAP)
_SEASON_CODE: dict[str, int] = {
    "Winter": 0,
    "Spring": 1,
    "Summer": 2,
    "Autumn": 3,
}


def build_features(
    panel: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Add lag, rolling, and calendar features to the panel.

    LEAKAGE GUARD: only rows where date <= as_of_date are processed and returned.
    All lag and rolling operations use groupby("sku_id") so no feature for SKU A
    can include demand data from SKU B.

    LAG FEATURES (per SKU, days back):
        lag_7   units_sold shifted 7 days
        lag_14  units_sold shifted 14 days
        lag_28  units_sold shifted 28 days

    ROLLING FEATURES (per SKU, shift(1) before window to exclude current row):
        rolling_mean_7   7-day rolling mean  (min_periods=1)
        rolling_mean_28  28-day rolling mean (min_periods=1)
        rolling_std_7    7-day rolling std   (min_periods=2, NaN → 0)
        rolling_std_28   28-day rolling std  (min_periods=7, NaN → 0)

    CALENDAR FEATURES:
        week_of_year  int  (ISO week 1–53)
        month         int  (1–12, ensure int dtype)
        is_holiday    int  (0 or 1, ensure int dtype)
        promo_flag    int  (0 or 1, ensure int dtype)
        is_weekend    int  (1 if dayofweek ≥ 5, else 0)
        season_code   int  (Winter=0, Spring=1, Summer=2, Autumn=3)

    SKU STATIC FEATURES:
        unit_cost    float
        list_price   float
        price_ratio  float  = list_price / unit_cost
        category_code int   = pandas Categorical codes of 'category'

    Parameters
    ----------
    panel       : cleaned panel DataFrame from pipeline.py
    as_of_date  : cutoff — only rows on or before this date are returned

    Returns
    -------
    pd.DataFrame with all original columns plus the new feature columns,
    filtered to date <= as_of_date.
    """
    as_of_date = pd.Timestamp(as_of_date)
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Apply leakage guard before computing any features
    df = df[df["date"] <= as_of_date].copy()
    df.sort_values(["sku_id", "date"], inplace=True)

    # ---- LAG FEATURES -------------------------------------------------------
    grp = df.groupby("sku_id", sort=False)["units_sold"]
    df["lag_7"]  = grp.shift(7)
    df["lag_14"] = grp.shift(14)
    df["lag_28"] = grp.shift(28)

    # ---- ROLLING FEATURES ---------------------------------------------------
    # shift(1) before rolling ensures current row's demand is excluded
    df["rolling_mean_7"]  = grp.transform(
        lambda s: s.shift(1).rolling(7,  min_periods=1).mean()
    )
    df["rolling_mean_28"] = grp.transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).mean()
    )
    df["rolling_std_7"]   = grp.transform(
        lambda s: s.shift(1).rolling(7,  min_periods=2).std()
    ).fillna(0.0)
    df["rolling_std_28"]  = grp.transform(
        lambda s: s.shift(1).rolling(28, min_periods=7).std()
    ).fillna(0.0)

    # ---- CALENDAR FEATURES --------------------------------------------------
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["month"].astype(int)
    df["is_holiday"]   = df["is_holiday"].astype(int)
    df["promo_flag"]   = df["promo_flag"].astype(int)
    df["is_weekend"]   = (df["date"].dt.dayofweek >= 5).astype(int)
    df["season_code"]  = df["season"].map(_SEASON_CODE).fillna(0).astype(int)

    # ---- SKU STATIC FEATURES ------------------------------------------------
    df["unit_cost"]  = df["unit_cost"].astype(float)
    df["list_price"] = df["list_price"].astype(float)
    df["price_ratio"] = df["list_price"] / df["unit_cost"].replace(0, np.nan)
    df["price_ratio"] = df["price_ratio"].fillna(1.0)

    # category must be Categorical for .cat.codes to work
    if not hasattr(df["category"], "cat"):
        df["category"] = pd.Categorical(df["category"])
    df["category_code"] = df["category"].cat.codes.astype(int)

    df.reset_index(drop=True, inplace=True)
    return df


def get_feature_columns() -> list[str]:
    """
    Canonical list of daily-grain feature columns consumed by models.

    This list is the contract between features.py and any model that trains
    on the daily panel. The order is fixed — do not change it without
    updating every consumer.

    Returns
    -------
    list[str]
    """
    return [
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_28",
        "week_of_year",
        "month",
        "is_holiday",
        "promo_flag",
        "is_weekend",
        "season_code",
        "unit_cost",
        "list_price",
        "price_ratio",
        "category_code",
    ]
