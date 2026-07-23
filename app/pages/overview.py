import streamlit as st

st.write("Forecast page loaded")


"""
app/pages/overview.py
---------------------
Page 2 — Inventory Intelligence Overview.

This is the first page the Ops team sees.
It must immediately answer: "What do I act on today?"

Layout:
  1. Category filter
  2. KPI strip (4 metrics)
  3. SKU decisioning scatter grid  (mirrors brief Figure 6)
  4. Priority action table         (sorted by ₹ value at stake)
  5. Download button
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.loaders import load_risk_scores


# ---------------------------------------------------------------------------
# Colour palette — one colour per quadrant, consistent across the whole app
# ---------------------------------------------------------------------------
COLOUR_MAP: dict[str, str] = {
    "reorder_now":    "#E74C3C",   # red
    "markdown_clear": "#3498DB",   # blue
    "watch_volatile": "#F39C12",   # orange
    "healthy":        "#27AE60",   # green
}

LABEL_MAP: dict[str, str] = {
    "reorder_now":    "Reorder Now",
    "markdown_clear": "Markdown / Clear",
    "watch_volatile": "Watch / Volatile",
    "healthy":        "Healthy",
}


def render() -> None:
    """Render the Overview page. Called by app/main.py."""

    # ------------------------------------------------------------------
    # STEP 1 — Load data
    # ------------------------------------------------------------------
    risk = load_risk_scores()

    # ------------------------------------------------------------------
    # STEP 2 — Page title + category filter
    # ------------------------------------------------------------------
    st.subheader("🏠 Inventory Intelligence Overview")
    st.caption(
        "Real-time demand risk across all 200 NorthBay Living SKUs. "
        "Data refreshes when the pipeline is re-run."
    )

    categories = ["All"] + sorted(risk["category"].unique().tolist())
    selected_cat = st.selectbox("Filter by Category", categories, index=0)

    if selected_cat != "All":
        risk = risk[risk["category"] == selected_cat]

    # ------------------------------------------------------------------
    # STEP 3 — KPI strip (4 columns)
    # ------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="SKUs Needing Reorder",
        value=int((risk["quadrant"] == "reorder_now").sum()),
    )
    col2.metric(
        label="SKUs to Markdown/Clear",
        value=int((risk["quadrant"] == "markdown_clear").sum()),
    )
    col3.metric(
        label="₹ Sales at Risk (Stockout)",
        value=(
            f"₹{risk[risk['quadrant'].isin(['reorder_now', 'watch_volatile'])]['value_at_stake_inr'].sum():,.0f}"
        ),
    )
    col4.metric(
        label="₹ Capital Locked (Overstock)",
        value=(
            f"₹{risk[risk['quadrant'].isin(['markdown_clear', 'watch_volatile'])]['value_at_stake_inr'].sum():,.0f}"
        ),
    )

    st.divider()

    # ------------------------------------------------------------------
    # STEP 4 — Decisioning scatter grid (brief Figure 6)
    # ------------------------------------------------------------------
    st.subheader("📊 SKU Decisioning Grid")
    st.caption("Each bubble is one SKU. Size = ₹ value at stake. Hover for details.")

    # Work on a copy so we don't mutate the cached DataFrame
    plot_df = risk.copy()

    # Guard: if value_at_stake_inr is all zero (e.g., only healthy SKUs),
    # set a uniform minimum bubble size to avoid division by zero.
    max_val = plot_df["value_at_stake_inr"].max()
    if max_val > 0:
        plot_df["bubble_size"] = (
            (plot_df["value_at_stake_inr"] / max_val * 50) + 5
        ).clip(5, 55)
    else:
        plot_df["bubble_size"] = 10.0

    plot_df["colour"] = plot_df["quadrant"].map(COLOUR_MAP)

    fig = go.Figure()

    for quadrant in ["reorder_now", "markdown_clear", "watch_volatile", "healthy"]:
        sub = plot_df[plot_df["quadrant"] == quadrant]
        if sub.empty:
            continue

        hover_text = (
            sub["sku_id"]
            + "<br>" + sub["category"]
            + "<br>₹" + sub["value_at_stake_inr"].apply(lambda x: f"{x:,.0f}")
            + "<br>Action: " + sub["recommended_action"]
        )

        fig.add_trace(
            go.Scatter(
                x=sub["overstock_risk"],
                y=sub["stockout_risk"],
                mode="markers",
                name=LABEL_MAP[quadrant],
                marker=dict(
                    size=sub["bubble_size"],
                    color=COLOUR_MAP[quadrant],
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Quadrant divider lines at x=0.5, y=0.5
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey", opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dash", line_color="grey", opacity=0.5)

    # Quadrant label annotations
    fig.add_annotation(
        x=0.25, y=0.95, text="REORDER NOW", showarrow=False,
        font=dict(color="#E74C3C", size=11),
    )
    fig.add_annotation(
        x=0.75, y=0.95, text="WATCH / VOLATILE", showarrow=False,
        font=dict(color="#F39C12", size=11),
    )
    fig.add_annotation(
        x=0.25, y=0.05, text="HEALTHY", showarrow=False,
        font=dict(color="#27AE60", size=11),
    )
    fig.add_annotation(
        x=0.75, y=0.05, text="MARKDOWN / CLEAR", showarrow=False,
        font=dict(color="#3498DB", size=11),
    )

    fig.update_layout(
        xaxis_title="Overstock Risk →",
        yaxis_title="↑ Stockout Risk",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------------
    # STEP 5 — Priority action table
    # ------------------------------------------------------------------
    st.subheader("📋 Priority Action List")
    st.caption("Sorted by ₹ value at stake — highest impact SKUs first.")

    display_cols = [
        "sku_id",
        "category",
        "quadrant",
        "recommended_action",
        "value_at_stake_inr",
        "on_hand_units",
        "forecast_8w_total",
        "lead_time_days",
    ]

    display_df = (
        risk[display_cols]
        .sort_values("value_at_stake_inr", ascending=False)
        .copy()
    )
    display_df["value_at_stake_inr"] = display_df["value_at_stake_inr"].apply(
        lambda x: f"₹{x:,.0f}"
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # Download button — full risk report as CSV
    # ------------------------------------------------------------------
    csv_data = risk.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Risk Report (CSV)",
        data=csv_data,
        file_name="foresight_risk_report.csv",
        mime="text/csv",
    )
