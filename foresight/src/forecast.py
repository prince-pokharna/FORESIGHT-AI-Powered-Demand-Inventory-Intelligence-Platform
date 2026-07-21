import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import (
    PANEL_PATH,
    FORECAST_PATH,
    BACKTEST_PATH,
    FORECAST_HORIZON_WEEKS,
    RANDOM_SEED,
    BACKTEST_N_SPLITS,
    SEASON_LENGTH_WEEKS,
)
from src.metrics import wape, mape, bias
from src.baseline import run_baseline_all_skus

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[forecast] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model hyper-parameters  (main regression model)
# ---------------------------------------------------------------------------
LIGHTGBM_PARAMS: dict = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}

# Season string → integer encoding (must match add_weekly_features)
_SEASON_MAP: dict[str, int] = {
    "Winter": 0,
    "Spring": 1,
    "Summer": 2,
    "Autumn": 3,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def aggregate_to_weekly(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the daily panel to weekly grain.

    Week key: Monday of each ISO week, computed as
        week_start = date - timedelta(days=date.dayofweek)

    Aggregation rules per (sku_id, week_start):
      units_sold      → sum   (total weekly demand — the model target)
      revenue         → sum
      promo_flag      → max   (1 if any day in the week was a promo day)
      is_holiday      → max
      on_hand_units   → last  (most recent snapshot in the week)
      on_order_units  → last
      lead_time_days  → first (constant per SKU)
      reorder_point   → first (constant per SKU)

    Static SKU columns carried forward via 'first':
      category, subcategory, unit_cost, list_price, season

    Returns a weekly-grain DataFrame sorted by (sku_id, week_start).
    """
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Monday-aligned week key
    df["week_start"] = df["date"] - pd.to_timedelta(df["date"].dt.dayofweek, unit="D")

    # Build aggregation dict
    agg_dict: dict = {
        "units_sold":    "sum",
        "revenue":       "sum",
        "promo_flag":    "max",
        "is_holiday":    "max",
        "on_hand_units": "last",
        "on_order_units": "last",
        "lead_time_days": "first",
        "reorder_point":  "first",
        # static SKU attributes
        "category":      "first",
        "subcategory":   "first",
        "unit_cost":     "first",
        "list_price":    "first",
        "season":        "first",   # carry the most common season in the week
    }

    weekly = (
        df.groupby(["sku_id", "week_start"], sort=False)
        .agg(agg_dict)
        .reset_index()
    )

    # Integer-encode category once, here, so it is available to features and model
    weekly["category"] = weekly["category"].astype("category")
    weekly["category_code"] = weekly["category"].cat.codes.astype(int)

    weekly.sort_values(["sku_id", "week_start"], inplace=True)
    weekly.reset_index(drop=True, inplace=True)

    logger.info(
        "aggregate_to_weekly complete: %d rows, %d unique SKUs, %d unique weeks.",
        len(weekly),
        weekly["sku_id"].nunique(),
        weekly["week_start"].nunique(),
    )
    return weekly


def add_weekly_features(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag, rolling, and calendar features to the weekly panel.

    All features are computed per sku_id group (sorted by week_start) so they
    never bleed across SKUs — the fundamental leakage guard.

    Lag features:
        lag_1w   units_sold shifted 1 week
        lag_2w   units_sold shifted 2 weeks
        lag_4w   units_sold shifted 4 weeks
        lag_52w  units_sold shifted 52 weeks  (same week last year)

    Rolling features (min_periods respected per spec):
        rolling_mean_4w   4-week rolling mean  (min_periods=1)
        rolling_mean_12w  12-week rolling mean (min_periods=1)
        rolling_std_4w    4-week rolling std   (min_periods=2, NaN → 0)

    Calendar features:
        week_of_year  ISO week number (int)
        month         month number (int)
        season_code   Winter=0, Spring=1, Summer=2, Autumn=3
        price_ratio   list_price / unit_cost

    Rows where lag_52w is NaN (first 52 weeks of history for each SKU) are
    dropped because the model cannot learn the seasonal signal without them.
    """
    df = weekly.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df.sort_values(["sku_id", "week_start"], inplace=True)

    # ---- per-SKU lag / rolling ------------------------------------------------
    grp = df.groupby("sku_id", sort=False)["units_sold"]

    df["lag_1w"]  = grp.shift(1)
    df["lag_2w"]  = grp.shift(2)
    df["lag_4w"]  = grp.shift(4)
    df["lag_52w"] = grp.shift(SEASON_LENGTH_WEEKS)

    df["rolling_mean_4w"]  = grp.transform(lambda s: s.shift(1).rolling(4,  min_periods=1).mean())
    df["rolling_mean_12w"] = grp.transform(lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    df["rolling_std_4w"]   = (
        grp.transform(lambda s: s.shift(1).rolling(4, min_periods=2).std())
        .fillna(0.0)
    )

    # ---- calendar features ---------------------------------------------------
    iso_week = df["week_start"].dt.isocalendar().week
    df["week_of_year"] = iso_week.astype(int)
    df["month"]        = df["week_start"].dt.month.astype(int)
    df["season_code"]  = df["season"].map(_SEASON_MAP).fillna(0).astype(int)
    df["price_ratio"]  = df["list_price"] / df["unit_cost"].replace(0, np.nan)
    df["price_ratio"]  = df["price_ratio"].fillna(1.0)

    # ---- drop rows without year-ago lag --------------------------------------
    before = len(df)
    df.dropna(subset=["lag_52w"], inplace=True)
    after = len(df)
    dropped = before - after
    logger.info(
        "add_weekly_features: dropped %d rows without lag_52w (first 52 weeks per SKU).",
        dropped,
    )

    df.reset_index(drop=True, inplace=True)
    return df


def get_weekly_feature_columns() -> list[str]:
    """
    Exact feature column list consumed by the LightGBM model.
    Used in train_model(), rolling_origin_backtest(), and generate_final_forecast().
    Order matters — must be consistent everywhere.
    """
    return [
        "lag_1w",
        "lag_2w",
        "lag_4w",
        "lag_52w",
        "rolling_mean_4w",
        "rolling_mean_12w",
        "rolling_std_4w",
        "week_of_year",
        "month",
        "season_code",
        "promo_flag",
        "is_holiday",
        "unit_cost",
        "list_price",
        "price_ratio",
        "category_code",
        "on_hand_units",
        "lead_time_days",
    ]


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    objective: str = "regression",
    alpha: float | None = None,
) -> lgb.LGBMRegressor:
    """
    Train a LightGBM model.

    Parameters
    ----------
    X_train   : feature matrix (rows without NaN)
    y_train   : target series aligned with X_train
    objective : "regression" for the main model, "quantile" for interval models
    alpha     : quantile level (0.10 or 0.90) — only used when objective="quantile"

    Returns the fitted LGBMRegressor.
    """
    params = dict(LIGHTGBM_PARAMS)  # shallow copy so we don't mutate the module constant
    params["objective"] = objective
    if objective == "quantile":
        if alpha is None:
            raise ValueError("alpha must be provided when objective='quantile'.")
        params["alpha"] = alpha
        params["metric"] = "quantile"

    model = lgb.LGBMRegressor(**params)
    # category_code is already an integer; pass its position index to LightGBM
    feature_cols = list(X_train.columns)
    cat_features = ["category_code"] if "category_code" in feature_cols else "auto"
    model.fit(X_train, y_train, categorical_feature=cat_features)
    return model


def rolling_origin_backtest(weekly: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling-origin cross-validation: LightGBM vs seasonal-naive baseline.

    5 folds.  Each fold expands the training window by ~5 % of total weeks.
    Training cutoffs range from 60 % to 80 % of available weeks, with the
    test window being the 8 weeks immediately after each cutoff.

    LEAKAGE GUARANTEE: X_train uses only rows where week_start <= cutoff_date.
    Features in X_test were computed from history that predates their week_start,
    because add_weekly_features() always shifts by at least 1 before rolling.

    Returns a DataFrame with columns:
        fold (int), cutoff_date (Timestamp), sku_id (str), week_start (Timestamp),
        actual (float), lgb_yhat (float), baseline_yhat (float)
    """
    FEATURE_COLS = get_weekly_feature_columns()
    all_weeks = sorted(weekly["week_start"].unique())
    n_weeks = len(all_weeks)

    # Build fold cutoffs: 60 %, 65 %, 70 %, 75 %, 80 % of total weeks
    fold_fractions = [0.60, 0.65, 0.70, 0.75, 0.80]
    cutoff_indices = [max(0, int(n_weeks * f) - 1) for f in fold_fractions]
    cutoff_dates   = [all_weeks[i] for i in cutoff_indices]

    results: list[pd.DataFrame] = []

    for fold_num, cutoff_date in enumerate(cutoff_dates, start=1):
        test_end = cutoff_date + pd.Timedelta(weeks=FORECAST_HORIZON_WEEKS)

        train_df = weekly[weekly["week_start"] <= cutoff_date].copy()
        test_df  = weekly[
            (weekly["week_start"] >  cutoff_date) &
            (weekly["week_start"] <= test_end)
        ].copy()

        if test_df.empty:
            logger.warning("Fold %d: no test rows after cutoff %s — skipping.", fold_num, cutoff_date)
            continue

        # ---- prepare train ---------------------------------------------------
        train_clean = train_df[FEATURE_COLS + ["units_sold"]].dropna()
        if train_clean.empty:
            logger.warning("Fold %d: train set empty after dropna — skipping.", fold_num)
            continue
        X_train = train_clean[FEATURE_COLS]
        y_train = train_clean["units_sold"]

        # ---- prepare test ----------------------------------------------------
        test_meta   = test_df[["sku_id", "week_start", "units_sold"]].copy()
        test_clean  = test_df[FEATURE_COLS + ["units_sold", "sku_id", "week_start"]].dropna(subset=FEATURE_COLS)
        X_test      = test_clean[FEATURE_COLS]
        y_test      = test_clean["units_sold"]

        # ---- train & predict -------------------------------------------------
        model  = train_model(X_train, y_train)
        preds  = np.clip(model.predict(X_test), 0, None)

        # ---- baseline forecast for the test SKUs at this cutoff --------------
        # Build a lightweight daily-grain panel proxy from weekly (for baseline function)
        # The baseline only needs sku_id, date (as week_start), units_sold
        baseline_panel = (
            train_df[["sku_id", "week_start", "units_sold"]]
            .rename(columns={"week_start": "date"})
            .copy()
        )
        test_skus = test_clean["sku_id"].unique().tolist()
        baseline_df = run_baseline_all_skus(
            baseline_panel,
            as_of_date=pd.Timestamp(cutoff_date),
            horizon_weeks=FORECAST_HORIZON_WEEKS,
        )
        # Align baseline to test rows on (sku_id, week_start)
        baseline_df = baseline_df.rename(columns={"week_start": "week_start"})
        test_with_baseline = test_clean[["sku_id", "week_start"]].copy()
        test_with_baseline["lgb_yhat"] = preds
        test_with_baseline["actual"]   = y_test.values
        test_with_baseline = test_with_baseline.merge(
            baseline_df[["sku_id", "week_start", "baseline_yhat"]],
            on=["sku_id", "week_start"],
            how="left",
        )
        # Fill any unmatched baseline rows with lag_52w as rough fallback
        test_with_baseline["baseline_yhat"] = test_with_baseline["baseline_yhat"].fillna(
            test_with_baseline["lgb_yhat"]   # last resort: keep model pred as baseline too
        )
        test_with_baseline["fold"]        = fold_num
        test_with_baseline["cutoff_date"] = cutoff_date

        # ---- per-fold metrics ------------------------------------------------
        fold_lgb_wape  = wape(test_with_baseline["actual"].values,
                              test_with_baseline["lgb_yhat"].values)
        fold_base_wape = wape(
            test_with_baseline["actual"].values,
            test_with_baseline["baseline_yhat"].values,
        )
        logger.info(
            "Fold %d | cutoff=%s | LGB WAPE=%.2f%% | Baseline WAPE=%.2f%%",
            fold_num,
            cutoff_date.date(),
            fold_lgb_wape  * 100,
            fold_base_wape * 100,
        )

        results.append(test_with_baseline)

    if not results:
        raise RuntimeError("Backtest produced no results — check that panel has sufficient history.")

    output = pd.concat(results, ignore_index=True)
    output = output[["fold", "cutoff_date", "sku_id", "week_start",
                      "actual", "lgb_yhat", "baseline_yhat"]]
    return output


def generate_final_forecast(
    panel: pd.DataFrame,
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train on ALL available weekly data, then forecast the next FORECAST_HORIZON_WEEKS
    weeks for every SKU.

    Generates:
        yhat            point forecast (main regression model)
        yhat_lower_80   10th-percentile quantile model
        yhat_upper_80   90th-percentile quantile model
        baseline_yhat   seasonal-naive comparison

    The returned DataFrame carries an is_future flag:
        True  = the 8 future forecast weeks
        False = historical weeks (in-sample fitted values, for chart context)

    Saves the result to FORECAST_PATH as parquet.
    """
    FEATURE_COLS = get_weekly_feature_columns()

    # ---- load calendar for promo / holiday look-ups --------------------------
    raw_calendar_path = Path("data/raw/calendar.csv")
    if raw_calendar_path.exists():
        calendar = pd.read_csv(raw_calendar_path, parse_dates=["date"])
        calendar["promo_flag"] = (calendar["promo_event"].fillna("") != "").astype(int)
        calendar["is_holiday"] = calendar["is_holiday"].fillna(0).astype(int)
        calendar["week_start"] = (
            calendar["date"] - pd.to_timedelta(calendar["date"].dt.dayofweek, unit="D")
        )
        # weekly promo / holiday: 1 if any day in that week is flagged
        cal_weekly = (
            calendar.groupby("week_start")
            .agg(promo_flag=("promo_flag", "max"), is_holiday=("is_holiday", "max"))
            .reset_index()
        )
    else:
        logger.warning("calendar.csv not found — future promo/holiday flags set to 0.")
        cal_weekly = pd.DataFrame(columns=["week_start", "promo_flag", "is_holiday"])

    # ---- train on full history -----------------------------------------------
    train_clean = weekly[FEATURE_COLS + ["units_sold"]].dropna()
    X_all = train_clean[FEATURE_COLS]
    y_all = train_clean["units_sold"]

    logger.info("Training main model on %d rows across %d SKUs...",
                len(X_all), weekly["sku_id"].nunique())
    model_main  = train_model(X_all, y_all, objective="regression")
    model_lower = train_model(X_all, y_all, objective="quantile", alpha=0.10)
    model_upper = train_model(X_all, y_all, objective="quantile", alpha=0.90)
    logger.info("All three LightGBM models trained (main + lower-80 + upper-80).")

    # ---- in-sample fitted values (historical context for the dashboard) ------
    hist_meta = weekly[["sku_id", "week_start"]].copy()
    hist_valid = weekly.dropna(subset=FEATURE_COLS).copy()
    hist_X = hist_valid[FEATURE_COLS]
    hist_valid["yhat"]           = np.clip(model_main.predict(hist_X),  0, None)
    hist_valid["yhat_lower_80"]  = np.clip(model_lower.predict(hist_X), 0, None)
    hist_valid["yhat_upper_80"]  = np.clip(model_upper.predict(hist_X), 0, None)
    hist_valid["is_future"]      = False

    # Seasonal-naive baseline for historical weeks
    baseline_panel = (
        panel[["sku_id", "date", "units_sold"]]
        .rename(columns={"date": "date"})
        .copy()
    )
    hist_as_of     = pd.Timestamp(weekly["week_start"].max())
    hist_baseline  = run_baseline_all_skus(baseline_panel, as_of_date=hist_as_of,
                                           horizon_weeks=FORECAST_HORIZON_WEEKS)
    # For historical rows just fill with NaN — baseline is shown for future only
    hist_valid["baseline_yhat"] = np.nan

    # ---- build future feature rows -------------------------------------------
    last_date        = pd.Timestamp(weekly["week_start"].max())
    future_starts    = [last_date + pd.Timedelta(weeks=w) for w in range(1, FORECAST_HORIZON_WEEKS + 1)]
    sku_list         = weekly["sku_id"].unique().tolist()

    # Per-SKU static lookups from the most recent weekly row
    latest = weekly.sort_values("week_start").groupby("sku_id", sort=False).last()

    future_rows: list[dict] = []

    for sku in sku_list:
        if sku not in latest.index:
            continue
        sku_hist = weekly[weekly["sku_id"] == sku].sort_values("week_start")
        last_row = latest.loc[sku]

        for w_idx, week_start in enumerate(future_starts):
            # Calendar signals
            cal_row    = cal_weekly[cal_weekly["week_start"] == week_start]
            promo_flag = int(cal_row["promo_flag"].values[0]) if not cal_row.empty else 0
            is_holiday = int(cal_row["is_holiday"].values[0]) if not cal_row.empty else 0

            # Lags from most recent history
            lag_1w  = float(sku_hist["units_sold"].iloc[-1]) if len(sku_hist) >= 1 else 0.0
            lag_2w  = float(sku_hist["units_sold"].iloc[-2]) if len(sku_hist) >= 2 else lag_1w
            lag_4w  = float(sku_hist["units_sold"].iloc[-4]) if len(sku_hist) >= 4 else lag_1w

            # lag_52w: look up same week 52 weeks ago
            target_52_date = week_start - pd.Timedelta(weeks=SEASON_LENGTH_WEEKS)
            same_week_rows = sku_hist[sku_hist["week_start"] == target_52_date]
            lag_52w = (
                float(same_week_rows["units_sold"].iloc[0])
                if not same_week_rows.empty
                else float(sku_hist["units_sold"].mean())
            )

            # Rolling stats from last known history (not from future rows)
            recent_4  = sku_hist["units_sold"].iloc[-4:].values  if len(sku_hist) >= 1 else np.array([0.0])
            recent_12 = sku_hist["units_sold"].iloc[-12:].values if len(sku_hist) >= 1 else np.array([0.0])
            rolling_mean_4w  = float(np.mean(recent_4))
            rolling_mean_12w = float(np.mean(recent_12))
            rolling_std_4w   = float(np.std(recent_4))  if len(recent_4) >= 2 else 0.0

            # Calendar encoding
            iso_week     = int(week_start.isocalendar()[1])
            month        = int(week_start.month)
            season_name  = {12: "Winter", 1: "Winter", 2: "Winter",
                             3: "Spring", 4: "Spring", 5: "Spring",
                             6: "Summer", 7: "Summer", 8: "Summer",
                             9: "Autumn", 10: "Autumn", 11: "Autumn"}[month]
            season_code  = _SEASON_MAP[season_name]

            future_rows.append({
                "sku_id":           sku,
                "week_start":       week_start,
                "lag_1w":           lag_1w,
                "lag_2w":           lag_2w,
                "lag_4w":           lag_4w,
                "lag_52w":          lag_52w,
                "rolling_mean_4w":  rolling_mean_4w,
                "rolling_mean_12w": rolling_mean_12w,
                "rolling_std_4w":   rolling_std_4w,
                "week_of_year":     iso_week,
                "month":            month,
                "season_code":      season_code,
                "promo_flag":       promo_flag,
                "is_holiday":       is_holiday,
                "unit_cost":        float(last_row["unit_cost"]),
                "list_price":       float(last_row["list_price"]),
                "price_ratio":      float(last_row["list_price"]) / max(float(last_row["unit_cost"]), 1e-6),
                "category_code":    int(last_row["category_code"]),
                "on_hand_units":    float(last_row["on_hand_units"]),
                "lead_time_days":   float(last_row["lead_time_days"]),
            })

    future_df = pd.DataFrame(future_rows)
    X_future  = future_df[FEATURE_COLS]

    future_df["yhat"]          = np.clip(model_main.predict(X_future),  0, None)
    future_df["yhat_lower_80"] = np.clip(model_lower.predict(X_future), 0, None)
    future_df["yhat_upper_80"] = np.clip(model_upper.predict(X_future), 0, None)
    future_df["is_future"]     = True

    # Seasonal-naive baseline for future weeks
    future_baseline = run_baseline_all_skus(
        baseline_panel,
        as_of_date=pd.Timestamp(weekly["week_start"].max()),
        horizon_weeks=FORECAST_HORIZON_WEEKS,
    )
    future_df = future_df.merge(
        future_baseline[["sku_id", "week_start", "baseline_yhat"]],
        on=["sku_id", "week_start"],
        how="left",
    )
    future_df["baseline_yhat"] = future_df["baseline_yhat"].fillna(future_df["yhat"])

    # ---- combine historical + future ----------------------------------------
    keep_cols = ["sku_id", "week_start", "yhat", "yhat_lower_80", "yhat_upper_80",
                 "baseline_yhat", "is_future"]

    hist_out = hist_valid[keep_cols].copy()
    futr_out = future_df[keep_cols].copy()

    output = pd.concat([hist_out, futr_out], ignore_index=True)
    output.sort_values(["sku_id", "week_start"], inplace=True)
    output.reset_index(drop=True, inplace=True)

    # ---- save ----------------------------------------------------------------
    Path(FORECAST_PATH).parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(FORECAST_PATH, index=False)

    n_future_rows = int(future_df.groupby("sku_id").ngroups * FORECAST_HORIZON_WEEKS)
    logger.info(
        "Forecast complete. %d SKUs × %d weeks saved to %s",
        future_df["sku_id"].nunique(),
        FORECAST_HORIZON_WEEKS,
        FORECAST_PATH,
    )
    return output


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Full pipeline:
      1. Load panel.parquet
      2. Aggregate to weekly
      3. Add weekly features
      4. Rolling-origin backtest  → save backtest_results.parquet
      5. Print overall WAPE summary
      6. Generate final forecast  → save forecasts.parquet
    """
    logger.info("Loading panel from %s ...", PANEL_PATH)
    panel = pd.read_parquet(PANEL_PATH)

    weekly = aggregate_to_weekly(panel)
    weekly = add_weekly_features(weekly)

    # ---- backtest -----------------------------------------------------------
    logger.info("Starting rolling-origin backtest (%d folds)...", BACKTEST_N_SPLITS)
    backtest_results = rolling_origin_backtest(weekly)

    Path(BACKTEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    backtest_results.to_parquet(BACKTEST_PATH, index=False)
    logger.info("Backtest results saved to %s", BACKTEST_PATH)

    # ---- overall WAPE summary -----------------------------------------------
    overall_lgb_wape  = wape(backtest_results["actual"].values,
                              backtest_results["lgb_yhat"].values)
    overall_base_wape = wape(backtest_results["actual"].values,
                              backtest_results["baseline_yhat"].values)
    overall_lgb_bias  = bias(backtest_results["actual"].values,
                              backtest_results["lgb_yhat"].values)

    logger.info("=" * 55)
    logger.info("BACKTEST SUMMARY (all folds combined)")
    logger.info("  LightGBM  WAPE : %.2f%%", overall_lgb_wape  * 100)
    logger.info("  Baseline  WAPE : %.2f%%", overall_base_wape * 100)
    logger.info("  LightGBM  Bias : %.2f units/week", overall_lgb_bias)
    if overall_lgb_wape < overall_base_wape:
        logger.info("  ✅  LightGBM BEATS the seasonal-naive baseline.")
    else:
        logger.info("  ⚠️   LightGBM does NOT beat the baseline — shipping baseline.")
    logger.info("=" * 55)

    # ---- final forecast -----------------------------------------------------
    logger.info("Generating final 8-week forecast for all SKUs...")
    generate_final_forecast(panel, weekly)


if __name__ == "__main__":
    run()
