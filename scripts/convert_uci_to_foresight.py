"""
scripts/convert_uci_to_foresight.py
------------------------------------
Converts the UCI Online Retail II dataset into the 4-table schema
that FORESIGHT's pipeline.py expects.

INPUT  : data/raw/online_retail_II.xlsx   (or any name listed in POSSIBLE_XLS)
OUTPUTS: data/raw/sales_daily.csv
         data/raw/sku_master.csv
         data/raw/calendar.csv
         data/raw/inventory_snapshots.csv

Run from the repo ROOT folder (where requirements.txt lives):
    python scripts/convert_uci_to_foresight.py

This takes 2-4 minutes because xlsx loading is slow.
"""

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Logging — configured BEFORE any logger calls
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[convert] %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Working directory guard
# ---------------------------------------------------------------------------
cwd = Path.cwd()
if not (cwd / "requirements.txt").exists():
    print(
        "\n ERROR: Run this script from the repo ROOT folder, not from scripts/.\n"
        f"   Current folder: {cwd}\n"
        "   Correct commands:\n"
        "       cd C:\\Zidio\\foresight\n"
        "       python scripts\\convert_uci_to_foresight.py\n"
    )
    sys.exit(1)

logger.info("Working directory: %s", cwd)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = cwd / "data" / "raw"

POSSIBLE_XLS = [
    RAW_DIR / "online_retail_II.xlsx",
    RAW_DIR / "Online Retail II.xlsx",
    RAW_DIR / "online-retail-II.xlsx",
    RAW_DIR / "archive.xlsx",
    RAW_DIR / "online_retail_II.csv",
    RAW_DIR / "Online Retail II.csv",
]

OUT_SALES     = RAW_DIR / "sales_daily.csv"
OUT_SKU       = RAW_DIR / "sku_master.csv"
OUT_CALENDAR  = RAW_DIR / "calendar.csv"
OUT_INVENTORY = RAW_DIR / "inventory_snapshots.csv"


def load_raw() -> pd.DataFrame:
    found = None
    for p in POSSIBLE_XLS:
        if p.exists():
            found = p
            break

    if found is None:
        names = "\n    ".join(str(p.name) for p in POSSIBLE_XLS[:5])
        print(
            f"\n ERROR: UCI dataset file not found in {RAW_DIR}\n"
            f"   Looked for:\n    {names}\n\n"
            "   Fix:\n"
            "   1. Download from https://www.kaggle.com/datasets/cgrymn/online-retail-ii-uci-dataset\n"
            "   2. Extract the zip\n"
            f"   3. Place the xlsx file in: {RAW_DIR}\n"
            "   4. Rename it to: online_retail_II.xlsx\n"
            "   5. Re-run this script\n"
        )
        sys.exit(1)

    logger.info("Found: %s", found)

    if found.suffix.lower() in (".xlsx", ".xls"):
        logger.info("Loading Excel — please wait 2-4 minutes...")
        try:
            df_a = pd.read_excel(found, sheet_name="Year 2009-2010", dtype=str)
            df_b = pd.read_excel(found, sheet_name="Year 2010-2011", dtype=str)
            df = pd.concat([df_a, df_b], ignore_index=True)
            logger.info("  Loaded 2 sheets: %d rows total", len(df))
        except Exception:
            df = pd.read_excel(found, dtype=str)
            logger.info("  Loaded single sheet: %d rows", len(df))
    else:
        logger.info("Loading CSV...")
        try:
            df = pd.read_csv(found, dtype=str, encoding="unicode_escape")
        except Exception:
            df = pd.read_csv(found, dtype=str, encoding="utf-8", errors="replace")
        logger.info("  Loaded %d rows", len(df))

    logger.info("Columns: %s", df.columns.tolist())
    return df


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning (%d rows)...", len(df))
    df.columns = df.columns.str.strip()
    df.rename(columns={"Invoice": "InvoiceNo", "Price": "UnitPrice",
                        "Customer ID": "CustomerID"}, inplace=True)

    required = {"InvoiceNo", "StockCode", "Description", "Quantity",
                "InvoiceDate", "UnitPrice"}
    missing = required - set(df.columns)
    if missing:
        print(f"\n ERROR: Columns missing: {missing}")
        print(f"   Found: {df.columns.tolist()}")
        sys.exit(1)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Quantity"]    = pd.to_numeric(df["Quantity"],    errors="coerce")
    df["UnitPrice"]   = pd.to_numeric(df["UnitPrice"],   errors="coerce")
    n0 = len(df)

    df = df[~df["InvoiceNo"].astype(str).str.upper().str.startswith("C")]
    logger.info("  Cancellations removed      : %d", n0 - len(df))
    n = len(df); df.dropna(subset=["InvoiceDate"], inplace=True)
    logger.info("  Missing date dropped       : %d", n - len(df))
    n = len(df); df = df[df["Quantity"] > 0]
    logger.info("  Non-positive Quantity      : %d", n - len(df))
    n = len(df); df = df[df["UnitPrice"] > 0]
    logger.info("  Non-positive UnitPrice     : %d", n - len(df))
    n = len(df); df.dropna(subset=["StockCode", "Description"], inplace=True)
    logger.info("  Missing StockCode/Desc     : %d", n - len(df))
    n = len(df); df = df[df["StockCode"].astype(str).str.match(r"^\d{5}[A-Za-z]?$")]
    logger.info("  Non-product codes removed  : %d", n - len(df))

    df["StockCode"]   = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip().str.title()
    logger.info("  Clean rows remaining       : %d", len(df))
    return df.reset_index(drop=True)


def build_sku_master(df: pd.DataFrame) -> tuple:
    logger.info("Building sku_master (top 200 SKUs)...")
    df["sku_id"] = "SKU_" + df["StockCode"].astype(str)
    top_skus = (
        df.groupby("sku_id")["Quantity"].sum()
        .sort_values(ascending=False).head(200).index.tolist()
    )
    df_top = df[df["sku_id"].isin(top_skus)].copy()

    desc  = df_top.groupby("sku_id")["Description"].agg(
        lambda x: x.value_counts().index[0]).reset_index().rename(
        columns={"Description": "description"})
    price = df_top.groupby("sku_id")["UnitPrice"].median().reset_index().rename(
        columns={"UnitPrice": "list_price"})
    price["list_price"] = price["list_price"].round(2)
    price["unit_cost"]  = (price["list_price"] * 0.55).round(2)
    launch = df_top.groupby("sku_id")["InvoiceDate"].min().reset_index().rename(
        columns={"InvoiceDate": "launch_date"})
    launch["launch_date"] = launch["launch_date"].dt.strftime("%Y-%m-%d")

    sku = desc.merge(price, on="sku_id").merge(launch, on="sku_id")

    def _cat(d):
        d = d.upper()
        if any(k in d for k in ["HEART","CANDLE","HOLDER","LANTERN","LIGHT",
                                  "CLOCK","FRAME","PICTURE","MIRROR","VASE",
                                  "SIGN","BUNTING","BANNER","GARLAND","WREATH"]):
            return "Decor", "Wall & Lighting"
        if any(k in d for k in ["BAG","TOTE","PURSE","WALLET","SATCHEL","SHOPPER"]):
            return "Bags & Accessories", "Bags"
        if any(k in d for k in ["MUG","CUP","PLATE","BOWL","JAR","TIN",
                                  "KITCHEN","CAKE","BAKING","TEAPOT","COFFEE"]):
            return "Kitchen & Dining", "Kitchenware"
        if any(k in d for k in ["CARD","WRAP","RIBBON","BOW","STICKER",
                                  "TAG","TISSUE","GIFT","ENVELOPE"]):
            return "Stationery & Gifting", "Gift Wrap"
        if any(k in d for k in ["CUSHION","PILLOW","THROW","BLANKET","DUVET"]):
            return "Home Textiles", "Cushions & Throws"
        if any(k in d for k in ["CHALK","PAINT","CRAFT","SEWING","NEEDLE"]):
            return "Craft & Hobby", "Craft Supplies"
        return "General Gifts", "Miscellaneous"

    cats = sku["description"].apply(_cat)
    sku["category"]    = [c[0] for c in cats]
    sku["subcategory"] = [c[1] for c in cats]
    logger.info("  %d SKUs, %d categories", len(sku), sku["category"].nunique())
    return sku[["sku_id","category","subcategory","launch_date",
                "unit_cost","list_price"]], top_skus


def build_sales_daily(df: pd.DataFrame, top_skus: list) -> pd.DataFrame:
    logger.info("Building sales_daily...")
    df["sku_id"] = "SKU_" + df["StockCode"].astype(str)
    df = df[df["sku_id"].isin(top_skus)].copy()
    df["date"]      = df["InvoiceDate"].dt.strftime("%Y-%m-%d")
    df["TotalLine"] = df["Quantity"] * df["UnitPrice"]
    daily = (df.groupby(["date","sku_id"])
               .agg(units_sold=("Quantity","sum"),
                    revenue=("TotalLine","sum"),
                    unit_price=("UnitPrice","mean"))
               .reset_index())
    daily["promo_flag"] = 0
    daily["revenue"]    = daily["revenue"].round(2)
    daily["unit_price"] = daily["unit_price"].round(2)
    logger.info("  %d rows | %s → %s", len(daily),
                daily["date"].min(), daily["date"].max())
    return daily[["date","sku_id","units_sold","revenue","unit_price","promo_flag"]]


def build_calendar(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building calendar...")
    d0 = pd.Timestamp(df["InvoiceDate"].min().date())
    d1 = pd.Timestamp(df["InvoiceDate"].max().date())
    cal = pd.DataFrame({"date": pd.date_range(d0, d1, freq="D")})
    cal["week"]   = cal["date"].dt.isocalendar().week.astype(int)
    cal["month"]  = cal["date"].dt.month.astype(int)
    cal["season"] = cal["month"].map({
        12:"Winter",1:"Winter",2:"Winter",
        3:"Spring",4:"Spring",5:"Spring",
        6:"Summer",7:"Summer",8:"Summer",
        9:"Autumn",10:"Autumn",11:"Autumn"})
    hols = {"2009-12-25","2009-12-28","2010-01-01","2010-04-02","2010-04-05",
            "2010-05-03","2010-05-31","2010-08-30","2010-12-25","2010-12-27",
            "2011-01-03","2011-04-22","2011-04-25","2011-04-29","2011-05-02",
            "2011-05-30","2011-08-29","2011-12-25","2011-12-26"}
    cal["is_holiday"] = cal["date"].dt.strftime("%Y-%m-%d").isin(hols).astype(int)
    promos = [
        ("2009-12-01","2009-12-24","Christmas Sale"),
        ("2010-02-12","2010-02-14","Valentines"),
        ("2010-03-28","2010-04-04","Easter Sale"),
        ("2010-10-25","2010-10-31","Halloween"),
        ("2010-11-26","2010-11-28","Black Friday"),
        ("2010-12-01","2010-12-24","Christmas Sale"),
        ("2011-02-12","2011-02-14","Valentines"),
        ("2011-04-17","2011-04-24","Easter Sale"),
        ("2011-04-29","2011-04-29","Royal Wedding"),
        ("2011-10-25","2011-10-31","Halloween"),
        ("2011-11-25","2011-11-27","Black Friday"),
        ("2011-12-01","2011-12-09","Christmas Sale"),
    ]
    cal["promo_event"] = ""
    for s, e, nm in promos:
        mask = (cal["date"] >= pd.Timestamp(s)) & (cal["date"] <= pd.Timestamp(e))
        cal.loc[mask, "promo_event"] = nm
    cal["date"] = cal["date"].dt.strftime("%Y-%m-%d")
    logger.info("  %d rows", len(cal))
    return cal


def build_inventory_snapshots(sales_daily, sku_master) -> pd.DataFrame:
    logger.info("Building inventory_snapshots (simulated — UCI has no inventory data)...")
    np.random.seed(42)
    sales_daily["date"] = pd.to_datetime(sales_daily["date"])
    snap_dates = pd.date_range(sales_daily["date"].min(),
                               sales_daily["date"].max(), freq="7D")
    rows = []
    for sku in sku_master["sku_id"].tolist():
        ss = sales_daily[sales_daily["sku_id"]==sku].set_index("date")["units_sold"]
        avg_w = float(ss.mean() if len(ss) > 0 else 1.0) * 7
        lt    = int(np.random.choice([7,10,14,21,30], p=[0.2,0.25,0.3,0.15,0.1]))
        rp    = max(5, round(avg_w * (lt/7) * 1.3))
        stk   = max(10, round(avg_w * 8 * np.random.uniform(0.8, 1.2)))
        for sd in snap_dates:
            we   = sd + pd.Timedelta(days=6)
            sold = float(ss[(ss.index>=sd)&(ss.index<=we)].sum())
            stk  = max(0, stk - sold)
            oo   = 0
            if np.random.rand() < 0.40:
                rs  = round(avg_w * np.random.uniform(4,8))
                oo  = rs; stk += rs
            rows.append({"date": sd.strftime("%Y-%m-%d"), "sku_id": sku,
                         "on_hand_units": max(0,int(stk)),
                         "on_order_units": int(oo),
                         "lead_time_days": lt, "reorder_point": rp})
    inv = pd.DataFrame(rows)
    logger.info("  %d rows", len(inv))
    return inv


def run():
    logger.info("=" * 55)
    logger.info("UCI -> FORESIGHT schema conversion")
    logger.info("=" * 55)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw  = load_raw()
    cln  = clean_raw(raw)
    sku, top_skus = build_sku_master(cln)
    sd   = build_sales_daily(cln, top_skus)
    cal  = build_calendar(cln)
    inv  = build_inventory_snapshots(sd, sku)
    sd.to_csv(OUT_SALES,    index=False)
    sku.to_csv(OUT_SKU,     index=False)
    cal.to_csv(OUT_CALENDAR,index=False)
    inv.to_csv(OUT_INVENTORY,index=False)
    logger.info("=" * 55)
    logger.info("Conversion complete. Files in data/raw/:")
    logger.info("  sales_daily.csv          %d rows", len(sd))
    logger.info("  sku_master.csv           %d rows", len(sku))
    logger.info("  calendar.csv             %d rows", len(cal))
    logger.info("  inventory_snapshots.csv  %d rows", len(inv))
    logger.info("")
    logger.info("Next: python -m src.pipeline")
    logger.info("=" * 55)


if __name__ == "__main__":
    run()
