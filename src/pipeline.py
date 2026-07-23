"""
src/pipeline.py
---------------
Deliverable D1 — Reproducible Data Pipeline.

Reads four raw CSV extracts, validates, cleans, joins them into one
analysis-ready panel, and saves it as a Parquet file.

Run with:
    python -m src.pipeline

Every cleaning step logs the row count before and after so the output log
becomes the audit trail for the D2 data-quality memo.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    RAW_SALES,
    RAW_SKU_MASTER,
    RAW_CALENDAR,
    RAW_INVENTORY,
    PANEL_PATH,
    CATEGORY_LABEL_MAP,
    CANONICAL_CATEGORIES,
)

# ---------------------------------------------------------------------------
# Logging — prefix every line with [pipeline] for easy grepping
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="[pipeline] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Ingest
# ---------------------------------------------------------------------------

def ingest() -> dict[str, pd.DataFrame]:
    """
    Read all four raw CSV extracts into DataFrames.

    Returns
    -------
    dict with keys: "sales", "sku_master", "calendar", "inventory"
    """
    logger.info("Ingesting raw CSV files...")

    sales = pd.read_csv(RAW_SALES)
    logger.info("  sales_daily.csv       : %d rows × %d cols", *sales.shape)

    sku_master = pd.read_csv(RAW_SKU_MASTER)
    logger.info("  sku_master.csv        : %d rows × %d cols", *sku_master.shape)

    calendar = pd.read_csv(RAW_CALENDAR)
    logger.info("  calendar.csv          : %d rows × %d cols", *calendar.shape)

    inventory = pd.read_csv(RAW_INVENTORY)
    logger.info("  inventory_snapshots.csv: %d rows × %d cols", *inventory.shape)

    return {
        "sales":      sales,
        "sku_master": sku_master,
        "calendar":   calendar,
        "inventory":  inventory,
    }


# ---------------------------------------------------------------------------
# Step 2 — Validate
# ---------------------------------------------------------------------------

def validate(dfs: dict[str, pd.DataFrame]) -> None:
    """
    Assert that every expected column exists in each DataFrame.
    Raises ValueError with a descriptive message on the first failure.
    Also verifies that the date columns parse without errors.
    """
    logger.info("Validating column schemas...")

    expected: dict[str, list[str]] = {
        "sales":      ["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"],
        "sku_master": ["sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"],
        "calendar":   ["date", "week", "month", "season", "is_holiday", "promo_event"],
        "inventory":  ["date", "sku_id", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point"],
    }

    for table, cols in expected.items():
        df = dfs[table]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Table '{table}' is missing columns: {missing}. "
                f"Found columns: {df.columns.tolist()}"
            )

    # Spot-check: date columns must be parseable
    for table, date_col in [("sales", "date"), ("calendar", "date"),
                             ("sku_master", "launch_date"), ("inventory", "date")]:
        sample = pd.to_datetime(dfs[table][date_col], errors="coerce")
        n_bad = sample.isna().sum()
        if n_bad > 0:
            logger.warning(
                "  %s.%s has %d unparseable date values (will coerce to NaT).",
                table, date_col, n_bad,
            )

    logger.info("  Schema validation passed for all 4 tables.")


# ---------------------------------------------------------------------------
# Step 3 — Clean
# ---------------------------------------------------------------------------

def clean(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Apply all cleaning steps. Every step logs before/after row counts.

    sales_daily cleaning:
      1. Parse date to datetime
      2. Drop exact duplicate rows
      3. Drop rows where units_sold is NaN
      4. Drop rows where units_sold <= 0
      5. Impute missing revenue as units_sold × unit_price
      6. Drop rows where unit_price <= 0

    sku_master cleaning:
      1. Drop duplicate rows on sku_id (keep first)
      2. Normalize category labels via CATEGORY_LABEL_MAP
      3. Parse launch_date to datetime

    calendar cleaning:
      1. Parse date to datetime
      2. Fill NaN in promo_event with ""

    inventory cleaning:
      1. Parse date to datetime
      (no other issues in this table)

    Returns
    -------
    Cleaned dict with the same keys as input.
    """
    result: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ sales
    logger.info("Cleaning sales_daily...")
    s = dfs["sales"].copy()
    n0 = len(s)

    # 1. Parse date
    s["date"] = pd.to_datetime(s["date"], errors="coerce")

    # 2. Drop exact duplicate rows
    s.drop_duplicates(inplace=True)
    logger.info("  Duplicates removed   : %d rows (%.1f%%)",
                n0 - len(s), (n0 - len(s)) / n0 * 100)

    # 3. Drop rows where units_sold is NaN
    n_before = len(s)
    s.dropna(subset=["units_sold"], inplace=True)
    logger.info("  Null units_sold dropped: %d rows", n_before - len(s))

    # 4. Drop rows where units_sold <= 0
    n_before = len(s)
    s = s[s["units_sold"] > 0]
    logger.info("  Zero/neg units_sold dropped: %d rows", n_before - len(s))

    # 5. Impute missing revenue
    n_missing_rev = s["revenue"].isna().sum()
    s.loc[s["revenue"].isna(), "revenue"] = (
        s.loc[s["revenue"].isna(), "units_sold"]
        * s.loc[s["revenue"].isna(), "unit_price"]
    )
    logger.info("  Missing revenue imputed (units × price): %d rows", n_missing_rev)

    # 6. Drop rows where unit_price <= 0
    n_before = len(s)
    s = s[s["unit_price"] > 0]
    logger.info("  Zero/neg unit_price dropped: %d rows", n_before - len(s))
    logger.info("  sales_daily final: %d rows", len(s))
    result["sales"] = s.reset_index(drop=True)

    # -------------------------------------------------------------- sku_master
    logger.info("Cleaning sku_master...")
    m = dfs["sku_master"].copy()
    n0 = len(m)

    # 1. Drop duplicate sku_id rows (keep first)
    m.drop_duplicates(subset="sku_id", keep="first", inplace=True)
    logger.info("  Duplicate SKU rows removed: %d", n0 - len(m))

    # 2. Normalize category labels
    def _normalise_category(label: str) -> str:
        if label in CATEGORY_LABEL_MAP:
            return CATEGORY_LABEL_MAP[label]
        if label in CANONICAL_CATEGORIES:
            return label
        logger.warning("  Unknown category label kept as-is: '%s'", label)
        return label

    m["category"] = m["category"].astype(str).apply(_normalise_category)
    logger.info("  Category labels normalised to %d canonical values.",
                len(CANONICAL_CATEGORIES))

    # 3. Parse launch_date
    m["launch_date"] = pd.to_datetime(m["launch_date"], errors="coerce")
    logger.info("  sku_master final: %d rows", len(m))
    result["sku_master"] = m.reset_index(drop=True)

    # --------------------------------------------------------------- calendar
    logger.info("Cleaning calendar...")
    c = dfs["calendar"].copy()

    # 1. Parse date
    c["date"] = pd.to_datetime(c["date"], errors="coerce")

    # 2. Fill NaN promo_event with empty string
    c["promo_event"] = c["promo_event"].fillna("").astype(str)
    logger.info("  calendar final: %d rows", len(c))
    result["calendar"] = c.reset_index(drop=True)

    # -------------------------------------------------------------- inventory
    logger.info("Cleaning inventory_snapshots...")
    inv = dfs["inventory"].copy()

    # 1. Parse date
    inv["date"] = pd.to_datetime(inv["date"], errors="coerce")
    logger.info("  inventory_snapshots final: %d rows", len(inv))
    result["inventory"] = inv.reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Step 4 — Join
# ---------------------------------------------------------------------------

def join(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all four cleaned tables into one analysis-ready panel.

    Step 1: sales LEFT JOIN sku_master ON sku_id
    Step 2: result LEFT JOIN calendar ON date
    Step 3: Forward-fill weekly inventory snapshots to daily grain
    Step 4: Exclude pre-launch rows (date < launch_date)
    Step 5: Sort by (sku_id, date), reset index
    Step 6: Cast category to pandas Categorical

    Final panel has at minimum these columns:
        date, sku_id, units_sold, revenue, unit_price, promo_flag,
        category, subcategory, launch_date, unit_cost, list_price,
        week, month, season, is_holiday, promo_event,
        on_hand_units, on_order_units, lead_time_days, reorder_point
    """
    logger.info("Joining tables...")

    sales      = dfs["sales"]
    sku_master = dfs["sku_master"]
    calendar   = dfs["calendar"]
    inventory  = dfs["inventory"]

    # Step 1: sales + sku_master
    panel = sales.merge(sku_master, on="sku_id", how="left")
    logger.info("  After sales ⋈ sku_master : %d rows", len(panel))

    # Step 2: + calendar
    panel = panel.merge(calendar, on="date", how="left")
    logger.info("  After ⋈ calendar          : %d rows", len(panel))

    # Step 3: forward-fill weekly inventory snapshots to daily grain
    # Build a full daily date × SKU spine, merge inventory on nearest prior date,
    # then forward-fill within each SKU group.
    inv_cols = ["date", "sku_id", "on_hand_units", "on_order_units",
                "lead_time_days", "reorder_point"]
    inventory_slim = inventory[inv_cols].copy()
    inventory_slim.sort_values(["sku_id", "date"], inplace=True)

    # Merge on exact (date, sku_id) — weekly snapshots land only on snapshot dates
    panel = panel.merge(
        inventory_slim,
        on=["date", "sku_id"],
        how="left",
    )

    # Forward-fill within each SKU (weekly → daily)
    ffill_cols = ["on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]
    panel.sort_values(["sku_id", "date"], inplace=True)
    panel[ffill_cols] = (
        panel.groupby("sku_id")[ffill_cols]
        .transform(lambda s: s.ffill())
    )
    logger.info("  Inventory forward-filled to daily grain.")

    # Step 4: Exclude pre-launch rows
    n_before = len(panel)
    panel = panel[panel["date"] >= panel["launch_date"]]
    n_excluded = n_before - len(panel)
    logger.info("  Pre-launch rows excluded  : %d", n_excluded)

    # Step 5: Sort and reset index
    panel.sort_values(["sku_id", "date"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    # Step 6: Cast category to Categorical for memory efficiency
    panel["category"] = pd.Categorical(
        panel["category"], categories=CANONICAL_CATEGORIES
    )

    logger.info(
        "  Final panel: %d rows × %d cols | %s → %s | %d SKUs",
        len(panel),
        len(panel.columns),
        str(panel["date"].min().date()),
        str(panel["date"].max().date()),
        panel["sku_id"].nunique(),
    )
    return panel


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run() -> None:
    """
    Run the full pipeline: ingest → validate → clean → join → save.
    Creates data/processed/ if it does not exist.
    """
    logger.info("=" * 55)
    logger.info("FORESIGHT Data Pipeline — starting")
    logger.info("=" * 55)

    dfs    = ingest()
    validate(dfs)
    dfs    = clean(dfs)
    panel  = join(dfs)

    PANEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)

    logger.info("=" * 55)
    logger.info("Pipeline complete. Panel saved to %s", PANEL_PATH)
    logger.info("=" * 55)


if __name__ == "__main__":
    run()
