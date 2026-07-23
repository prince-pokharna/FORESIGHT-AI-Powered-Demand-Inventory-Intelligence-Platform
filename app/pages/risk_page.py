import streamlit as st

st.write("Forecast page loaded")


"""
app/pages/risk_page.py
----------------------
Page 3 — Risk & Actions.

Presents every SKU's risk classification in four tabs — one per decisioning
quadrant — so the Ops team can immediately pull an action list for their
planning meeting without sorting through a single flat table.

Layout:
  1. KPI strip (count per quadrant)
  2. Category filter
  3. Four tabs:
       🔴 Reorder Now      — sort by ₹ at risk, download CSV
       🔵 Markdown / Clear — sort by ₹ at risk, download CSV
       🟠 Watch / Volatile — sort by ₹ at risk, table only
       🟢 Healthy          — sorted alphabetically, compact view
"""

import streamlit as st

from utils.loaders import load_risk_scores


# Columns to show inside each tab
DISPLAY_COLS = [
    "sku_id",
    "category",
    "recommended_action",
    "value_at_stake_inr",
    "on_hand_units",
    "forecast_8w_total",
    "lead_time_days",
    "stockout_risk",
    "overstock_risk",
]


def render() -> None:
    """Render the Risk & Actions page. Called by app/main.py."""

    # ------------------------------------------------------------------
    # STEP 1 — Load data
    # ------------------------------------------------------------------
    risk = load_risk_scores()

    # ------------------------------------------------------------------
    # STEP 2 — KPI strip (4 quadrant counts — full dataset, before filter)
    # ------------------------------------------------------------------
    st.title("⚠️ Risk & Actions")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "🔴 Reorder Now",
        int((risk["quadrant"] == "reorder_now").sum()),
    )
    k2.metric(
        "🔵 Markdown / Clear",
        int((risk["quadrant"] == "markdown_clear").sum()),
    )
    k3.metric(
        "🟠 Watch / Volatile",
        int((risk["quadrant"] == "watch_volatile").sum()),
    )
    k4.metric(
        "🟢 Healthy",
        int((risk["quadrant"] == "healthy").sum()),
    )

    st.divider()

    # ------------------------------------------------------------------
    # STEP 3 — Category filter (applied to tabs below)
    # ------------------------------------------------------------------
    categories   = ["All"] + sorted(risk["category"].unique().tolist())
    selected_cat = st.selectbox("Filter by Category", categories)

    if selected_cat != "All":
        risk = risk[risk["category"] == selected_cat]

    # ------------------------------------------------------------------
    # STEP 4 — Four tabs
    # ------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔴 Reorder Now",
            "🔵 Markdown / Clear",
            "🟠 Watch / Volatile",
            "🟢 Healthy",
        ]
    )

    # ---- Tab 1: Reorder Now ----
    with tab1:
        df = (
            risk[risk["quadrant"] == "reorder_now"]
            .sort_values("value_at_stake_inr", ascending=False)
        )
        st.caption(
            f"{len(df)} SKUs need reorder. "
            "Sorted by ₹ at risk — act on these first."
        )
        if df.empty:
            st.success("No SKUs currently need reordering. ✅")
        else:
            st.dataframe(df[DISPLAY_COLS], use_container_width=True, hide_index=True)
            st.download_button(
                label="📥 Download Reorder Plan",
                data=df.to_csv(index=False),
                file_name="reorder_plan.csv",
                mime="text/csv",
                key="dl_reorder",
            )

    # ---- Tab 2: Markdown / Clear ----
    with tab2:
        df = (
            risk[risk["quadrant"] == "markdown_clear"]
            .sort_values("value_at_stake_inr", ascending=False)
        )
        st.caption(
            f"{len(df)} SKUs are overstocked. "
            "Promotions or discounts could free locked capital."
        )
        if df.empty:
            st.success("No SKUs are currently overstocked. ✅")
        else:
            st.dataframe(df[DISPLAY_COLS], use_container_width=True, hide_index=True)
            st.download_button(
                label="📥 Download Markdown Candidates",
                data=df.to_csv(index=False),
                file_name="markdown_candidates.csv",
                mime="text/csv",
                key="dl_markdown",
            )

    # ---- Tab 3: Watch / Volatile ----
    with tab3:
        df = (
            risk[risk["quadrant"] == "watch_volatile"]
            .sort_values("value_at_stake_inr", ascending=False)
        )
        st.caption(
            f"{len(df)} SKUs show volatile demand — "
            "review manually before placing orders."
        )
        if df.empty:
            st.success("No volatile SKUs detected. ✅")
        else:
            st.dataframe(df[DISPLAY_COLS], use_container_width=True, hide_index=True)

    # ---- Tab 4: Healthy ----
    with tab4:
        df = risk[risk["quadrant"] == "healthy"].sort_values("sku_id")
        st.caption(
            f"{len(df)} SKUs are healthy — no action needed."
        )
        if df.empty:
            st.info("No SKUs in healthy status for the current filter.")
        else:
            st.dataframe(
                df[["sku_id", "category", "on_hand_units", "forecast_8w_total"]],
                use_container_width=True,
                hide_index=True,
            )
