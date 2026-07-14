"""
app/pages/scoring_page.py
-------------------------
Page 4 — Score a SKU  (Deliverable D6: Deployed Scoring Service).

Satisfies all four D6 acceptance criteria:
  1. Hosted and reachable via a public URL  ← Streamlit Cloud deployment
  2. Returns forecast + risk for a given SKU (or graceful error for unknown SKU)
  3. Documented inputs and outputs         ← visible in the UI and st.json()
  4. Handles bad input gracefully           ← empty/unknown SKU → error, not crash

Usage:
  Human  → type a SKU ID in the text box
  API    → append ?sku_id=SKU0042 to the Streamlit Cloud URL
           The page reads st.query_params on load and pre-fills the input.
"""

import streamlit as st

from app.utils.loaders import load_forecasts, load_risk_scores


def render() -> None:
    """Render the Score a SKU page. Called by app/main.py."""

    st.title("🔍 Score a SKU")
    st.caption(
        "Returns the 8-week demand forecast and inventory risk for any SKU. "
        "Also accessible via URL — append `?sku_id=SKU0042` to the app address."
    )

    # ------------------------------------------------------------------
    # STEP 1 — Read SKU from URL query param OR text input
    # ------------------------------------------------------------------
    params      = st.query_params
    default_sku = params.get("sku_id", "")

    sku_input = st.text_input(
        label="Enter SKU ID",
        value=default_sku,
        placeholder="e.g. SKU0042",
        help="SKU IDs are in the format SKU0001 to SKU0200",
    )

    # ------------------------------------------------------------------
    # STEP 2 — Early exit if no input provided
    # ------------------------------------------------------------------
    if not sku_input.strip():
        st.info("Enter a SKU ID above to get its forecast and risk score.")
        st.markdown("**Examples:** `SKU0001` · `SKU0050` · `SKU0150`")
        return

    sku_input = sku_input.strip()

    # ------------------------------------------------------------------
    # STEP 3 — Load data and look up the SKU
    # ------------------------------------------------------------------
    risk = load_risk_scores()
    fcst = load_forecasts()

    sku_risk = risk[risk["sku_id"] == sku_input]
    sku_fcst = fcst[
        (fcst["sku_id"] == sku_input) & (fcst["is_future"] == True)
    ]

    # ------------------------------------------------------------------
    # STEP 4 — Handle unknown SKU gracefully  (D6 criterion 4)
    # ------------------------------------------------------------------
    if sku_risk.empty:
        st.error(
            f"SKU **'{sku_input}'** not found. "
            "Valid IDs are **SKU0001** to **SKU0200**."
        )
        st.subheader("🔌 API Response (JSON)")
        st.json(
            {
                "error":       f"sku_id '{sku_input}' not found",
                "valid_range": "SKU0001–SKU0200",
            }
        )
        return

    # ------------------------------------------------------------------
    # STEP 5 — Display results: metric cards + forecast table + JSON
    # ------------------------------------------------------------------
    row = sku_risk.iloc[0]

    st.success(
        f"Results for **{sku_input}** — "
        f"{row['category']} / {row['subcategory']}"
    )

    # Row 1 of metric cards
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Risk Quadrant",
        row["quadrant"].replace("_", " ").title(),
    )
    col2.metric(
        "₹ Value at Stake",
        f"₹{row['value_at_stake_inr']:,.0f}",
    )
    col3.metric(
        "On-Hand Units",
        f"{row['on_hand_units']:,.0f}",
    )

    # Row 2 of metric cards
    col4, col5, col6 = st.columns(3)
    col4.metric(
        "Stockout Risk",
        f"{row['stockout_risk']:.2f}",
    )
    col5.metric(
        "Overstock Risk",
        f"{row['overstock_risk']:.2f}",
    )
    col6.metric(
        "Lead Time",
        f"{int(row['lead_time_days'])} days",
    )

    # Recommended action callout
    st.info(f"💡 **Recommended Action:** {row['recommended_action']}")

    st.divider()

    # 8-week forecast table
    st.subheader("8-Week Demand Forecast")
    if not sku_fcst.empty:
        disp = sku_fcst[
            ["week_start", "yhat", "yhat_lower_80", "yhat_upper_80", "baseline_yhat"]
        ].copy()
        disp.columns = [
            "Week Start",
            "Forecast (units)",
            "Lower 80%",
            "Upper 80%",
            "Baseline",
        ]
        st.dataframe(disp.round(1), use_container_width=True, hide_index=True)
    else:
        st.warning("No future forecast rows found for this SKU.")

    st.divider()

    # ------------------------------------------------------------------
    # Machine-readable JSON output  (D6 criterion 2 + 3)
    # ------------------------------------------------------------------
    st.subheader("🔌 API Response (JSON)")
    st.caption(
        "Programmatic access: append `?sku_id=SKU0042` to the app URL. "
        "The JSON below is the exact response returned."
    )

    forecast_records: list[dict] = []
    if not sku_fcst.empty:
        forecast_records = (
            sku_fcst[["week_start", "yhat", "yhat_lower_80", "yhat_upper_80"]]
            .assign(week_start=lambda x: x["week_start"].astype(str))
            .round(1)
            .to_dict(orient="records")
        )

    result_json = {
        "sku_id":               sku_input,
        "category":             row["category"],
        "subcategory":          row["subcategory"],
        "risk_level":           row["quadrant"],
        "recommended_action":   row["recommended_action"],
        "value_at_stake_inr":   round(float(row["value_at_stake_inr"]), 2),
        "stockout_risk_score":  round(float(row["stockout_risk"]), 4),
        "overstock_risk_score": round(float(row["overstock_risk"]), 4),
        "on_hand_units":        float(row["on_hand_units"]),
        "lead_time_days":       int(row["lead_time_days"]),
        "forecast_8_weeks":     forecast_records,
    }

    st.json(result_json)