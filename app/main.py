"""
app/main.py
-----------
FORESIGHT — Streamlit entry point and page router.

Run locally with:
    streamlit run app/main.py

This file contains ZERO business logic.
Its only job is to configure the page, render the sidebar, and delegate
to the correct page module based on the user's navigation choice.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pages import overview


import streamlit as st
import pandas as pd
from utils.loaders import load_risk_scores

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FORESIGHT — Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

def render_risk_alerts(risk_df):
    """Sidebar panel showing critical/high risk SKUs."""
    st.sidebar.markdown("## 🚨 Inventory Alerts")
    critical = risk_df[risk_df["quadrant"] == "reorder_now"]   # adapt to your label
    high = risk_df[risk_df["quadrant"] == "watch_volatile"]    # adapt if needed

    if critical.empty and high.empty:
        st.sidebar.success("✅ All SKUs within safe thresholds")
        return

    for _, row in critical.iterrows():
        st.sidebar.error(
            f"🔴 **{row['sku_id']}** — STOCKOUT RISK: {row['stockout_risk']*100:.1f}%\n"
            f"Reorder {row['forecast_8w_total']:.0f} units NOW"   # or recommended_order_qty if exists
        )
    for _, row in high.iterrows():
        st.sidebar.warning(
            f"🟡 **{row['sku_id']}** — High Risk: {row['stockout_risk']*100:.1f}%\n"
            f"ROP: {row.get('reorder_point', 'N/A')} units"
        )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📦 FORESIGHT")
st.sidebar.caption("NorthBay Living · Demand & Inventory Intelligence")
st.sidebar.markdown("---")

try:
    risk_data = load_risk_scores()
    if not risk_data.empty:
        render_risk_alerts(risk_data)
    else:
        st.sidebar.info("No risk data loaded yet.")
except Exception:
    st.sidebar.warning("Could not load risk data.")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Overview",
        "📈 Forecast",
        "⚠️ Risk & Actions",
        "🔍 Score a SKU",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by LightGBM + Streamlit")

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------
if page.startswith("🏠"):
    from pages import overview
    overview.render()

elif page.startswith("📈"):
    from pages import forecast_page
    forecast_page.render()

elif page.startswith("⚠️"):
    from pages import risk_page
    risk_page.render()

else:
    from pages import scoring_page
    scoring_page.render()
