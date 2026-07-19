"""
src/config.py
-------------
Central configuration for Project FORESIGHT.

All file paths, modelling constants, and data-cleaning maps live here.
No other module should hardcode paths or magic numbers — import them from here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
DATA_RAW       = Path("data/raw")
DATA_PROCESSED = Path("data/processed")

# ---------------------------------------------------------------------------
# Raw input file paths
# ---------------------------------------------------------------------------
RAW_SALES      = DATA_RAW / "sales_daily.csv"
RAW_SKU_MASTER = DATA_RAW / "sku_master.csv"
RAW_CALENDAR   = DATA_RAW / "calendar.csv"
RAW_INVENTORY  = DATA_RAW / "inventory_snapshots.csv"

# ---------------------------------------------------------------------------
# Processed output file paths
# ---------------------------------------------------------------------------
PANEL_PATH    = DATA_PROCESSED / "panel.parquet"
FORECAST_PATH = DATA_PROCESSED / "forecasts.parquet"
RISK_PATH     = DATA_PROCESSED / "risk_scores.parquet"
BACKTEST_PATH = DATA_PROCESSED / "backtest_results.parquet"

# ---------------------------------------------------------------------------
# Modelling constants
# ---------------------------------------------------------------------------
FORECAST_HORIZON_WEEKS  = 8      # weeks ahead to forecast
RANDOM_SEED             = 42     # fixed for reproducibility everywhere
SERVICE_LEVEL           = 0.90   # 90% — underpins safety stock calculation
OVERSTOCK_WINDOW_WEEKS  = 8      # forward window for overstock risk check
SAFETY_STOCK_Z          = 1.65   # z-score for 90% service level
SEASON_LENGTH_WEEKS     = 52     # one year of weekly data = one season
BACKTEST_N_SPLITS       = 5      # rolling-origin CV folds

# ---------------------------------------------------------------------------
# Data-cleaning maps
# ---------------------------------------------------------------------------
# Maps every observed dirty category label to its canonical form.
# Labels already in canonical form are NOT listed here — they need no mapping.
CATEGORY_LABEL_MAP: dict[str, str] = {
    "furniture":       "Furniture",
    "DECOR":           "Decor",
    "decor":           "Decor",
    "Small Appliance": "Small Appliances",
    "Bedding and Bath":"Bedding & Bath",
}

# The four canonical category values that must appear in the clean dataset.
CANONICAL_CATEGORIES: list[str] = [
    "Furniture",
    "Decor",
    "Small Appliances",
    "Bedding & Bath",
]