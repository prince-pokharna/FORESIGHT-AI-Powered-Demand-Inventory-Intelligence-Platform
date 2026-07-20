import streamlit as st

st.write("Forecast page loaded")


"""
app/pages/forecast_page.py
--------------------------
Page 1 — Per-SKU Demand Forecast.

Shows the LightGBM 8-week forecast for any selected SKU, alongside:
  - 12 weeks of actual historical demand
  - 80 % prediction interval (shaded band)
  - Seasonal-naive baseline (dashed) for comparison
  - Vertical marker at the forecast start date

Layout:
  1. Category filter + SKU selector
  2. KPI strip (quadrant, 8-week total, on-hand stock, ₹ at stake)
  3. Forecast chart (Plotly — mirrors brief Figure 5)
  4. Forecast data table
  5. Download button
"""

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.loaders import (
    load_forecasts,
    load_panel_history,
    load_risk_scores,
    get_sku_list,
)


def render() -> None:
    """Render the Forecast page. Called by app/main.py."""

    st.title("📈 Demand Forecast")
    st.caption(
        "LightGBM 8-week weekly forecast per SKU, with 80 % prediction interval "
        "and seasonal-naive baseline for comparison."
    )

    # ------------------------------------------------------------------
    # STEP 1 — Load data
    # ------------------------------------------------------------------
    risk    = load_risk_scores()
    fcst    = load_forecasts()
    history = load_panel_history(last_n_weeks=12)
    sku_list = get_sku_list(risk)   # most impactful SKUs first

    # ------------------------------------------------------------------
    # STEP 2 — Controls (category filter + SKU selector)
    # ------------------------------------------------------------------
    col1, col2 = st.columns(2)

    categories   = ["All"] + sorted(risk["category"].unique().tolist())
    selected_cat = col2.selectbox("Filter by Category", categories, index=0)

    if selected_cat != "All":
        filtered_skus = (
            risk[risk["category"] == selected_cat]
            .sort_values("value_at_stake_inr", ascending=False)["sku_id"]
            .unique()
            .tolist()
        )
        selected_sku = col1.selectbox(
            "Select SKU", filtered_skus, key="sku_cat_filtered"
        )
    else:
        selected_sku = col1.selectbox("Select SKU", sku_list)

    # ------------------------------------------------------------------
    # STEP 3 — KPI strip for the selected SKU
    # ------------------------------------------------------------------
    sku_risk_rows = risk[risk["sku_id"] == selected_sku]
    if sku_risk_rows.empty:
        st.warning(f"No risk data found for {selected_sku}.")
        return
    sku_risk = sku_risk_rows.iloc[0]

    sku_fcst_future = fcst[
        (fcst["sku_id"] == selected_sku) & (fcst["is_future"] == True)
    ]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Risk Quadrant",
        sku_risk["quadrant"].replace("_", " ").title(),
    )
    k2.metric(
        "8-Week Forecast",
        f"{sku_fcst_future['yhat'].sum():,.0f} units",
    )
    k3.metric(
        "On-Hand Stock",
        f"{sku_risk['on_hand_units']:,.0f} units",
    )
    k4.metric(
        "₹ Value at Stake",
        f"₹{sku_risk['value_at_stake_inr']:,.0f}",
    )

    st.divider()

    # ------------------------------------------------------------------
    # STEP 4 — Forecast chart
    # ------------------------------------------------------------------
    st.subheader(f"📈 Demand Forecast — {selected_sku}")

    # Historical actual demand (last 12 weeks)
    sku_history = (
        history[history["sku_id"] == selected_sku]
        .sort_values("week_start")
    )

    # Future forecast weeks only
    sku_future = (
        fcst[(fcst["sku_id"] == selected_sku) & (fcst["is_future"] == True)]
        .sort_values("week_start")
    )

    if sku_future.empty:
        st.warning(
            f"No forecast data found for {selected_sku}. "
            "Ensure `python -m src.forecast` has been run."
        )
        return

    fig = go.Figure()

    # Trace 1 — Historical actual demand
    fig.add_trace(
        go.Scatter(
            x=sku_history["week_start"],
            y=sku_history["units_sold_weekly"],
            name="Actual (Historical)",
            mode="lines+markers",
            line=dict(color="#2C3E50", width=2),
            marker=dict(size=5),
        )
    )

    # Trace 2 — 80 % prediction interval (shaded band)
    # Add BEFORE the forecast line so the line renders on top of the fill.
    fig.add_trace(
        go.Scatter(
            x=pd.concat(
                [sku_future["week_start"], sku_future["week_start"].iloc[::-1]],
                ignore_index=True,
            ),
            y=pd.concat(
                [sku_future["yhat_upper_80"], sku_future["yhat_lower_80"].iloc[::-1]],
                ignore_index=True,
            ),
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.15)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            name="80% Confidence Interval",
            showlegend=True,
            hoverinfo="skip",
        )
    )

    # Trace 3 — Seasonal-naive baseline (dashed grey)
    fig.add_trace(
        go.Scatter(
            x=sku_future["week_start"],
            y=sku_future["baseline_yhat"],
            name="Seasonal-Naive Baseline",
            mode="lines",
            line=dict(color="#95A5A6", width=1.5, dash="dash"),
        )
    )

    # Trace 4 — LightGBM forecast (solid blue, rendered last = on top)
    fig.add_trace(
        go.Scatter(
            x=sku_future["week_start"],
            y=sku_future["yhat"],
            name="LightGBM Forecast",
            mode="lines+markers",
            line=dict(color="#2980B9", width=2.5),
            marker=dict(size=6),
        )
    )

    # Vertical dashed line — boundary between history and forecast
    if not sku_history.empty:
        forecast_start = sku_history["week_start"].max()
        fig.add_vline(
            x=forecast_start,
            line_dash="dot",
            line_color="#E74C3C",
            annotation_text="Forecast Start",
            annotation_position="top right",
        )

    fig.update_layout(
        xaxis_title="Week",
        yaxis_title="Units Sold (Weekly)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        margin=dict(l=20, r=20, t=20, b=60),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # STEP 5 — Forecast data table
    # ------------------------------------------------------------------
    st.subheader("📋 Forecast Data")

    display_cols = [
        "week_start",
        "yhat",
        "yhat_lower_80",
        "yhat_upper_80",
        "baseline_yhat",
    ]
    st.dataframe(
        sku_future[display_cols].round(1),
        use_container_width=True,
        hide_index=True,
    )

    # Download button
    csv_data = sku_future.to_csv(index=False)
    st.download_button(
        label=f"📥 Download Forecast for {selected_sku}",
        data=csv_data,
        file_name=f"{selected_sku}_forecast.csv",
        mime="text/csv",
    )
