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

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FORESIGHT — Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📦 FORESIGHT")
st.sidebar.caption("NorthBay Living · Demand & Inventory Intelligence")
st.sidebar.markdown("---")

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
    from app.pages import overview
    overview.render()

elif page.startswith("📈"):
    from app.pages import forecast_page
    forecast_page.render()

elif page.startswith("⚠️"):
    from app.pages import risk_page
    risk_page.render()

else:
    from app.pages import scoring_page
    scoring_page.render()