
import pytest
import numpy as np
import pandas as pd

from src.risk import (
    compute_stockout_risk_score,
    compute_overstock_risk_score,
    assign_quadrant,
    compute_value_at_stake,
)


class TestStockoutRiskScore:

    def test_no_stockout_when_overstocked(self):
        """Plenty of stock vs modest demand → stockout risk must be 0.0."""
        score = compute_stockout_risk_score(
            on_hand=1000,
            on_order=0,
            lead_time_days=7,
            weekly_forecasts=[50.0] * 8,
            demand_std_weekly=5.0,
        )
        assert score == 0.0, (
            f"Expected 0.0 for well-stocked SKU, got {score}"
        )

    def test_high_stockout_when_empty(self):
        score = compute_stockout_risk_score(
            on_hand=0,
            on_order=0,
            lead_time_days=14,
            weekly_forecasts=[100.0] * 8,
            demand_std_weekly=10.0,
        )
        assert score == 1.0, (
            f"Expected 1.0 for zero-stock, high-demand SKU, got {score}"
        )

    def test_stockout_score_in_range(self):
        score = compute_stockout_risk_score(
            on_hand=50,
            on_order=20,
            lead_time_days=14,
            weekly_forecasts=[80.0] * 8,
            demand_std_weekly=15.0,
        )
        assert 0.0 <= score <= 1.0, (
            f"Stockout risk score out of [0, 1] range: {score}"
        )

    def test_stockout_score_is_float(self):
        score = compute_stockout_risk_score(
            on_hand=100,
            on_order=50,
            lead_time_days=10,
            weekly_forecasts=[30.0] * 8,
            demand_std_weekly=8.0,
        )
        assert isinstance(score, float), f"Expected float, got {type(score)}"

    def test_on_order_reduces_risk(self):
        score_no_order = compute_stockout_risk_score(
            on_hand=20, on_order=0, lead_time_days=14,
            weekly_forecasts=[40.0] * 8, demand_std_weekly=5.0,
        )
        score_with_order = compute_stockout_risk_score(
            on_hand=20, on_order=60, lead_time_days=14,
            weekly_forecasts=[40.0] * 8, demand_std_weekly=5.0,
        )
        assert score_with_order <= score_no_order, (
            "On-order stock should reduce stockout risk, not increase it."
        )

    def test_longer_lead_time_increases_risk(self):
        score_short = compute_stockout_risk_score(
            on_hand=50, on_order=0, lead_time_days=7,
            weekly_forecasts=[30.0] * 8, demand_std_weekly=5.0,
        )
        score_long = compute_stockout_risk_score(
            on_hand=50, on_order=0, lead_time_days=30,
            weekly_forecasts=[30.0] * 8, demand_std_weekly=5.0,
        )
        assert score_long >= score_short, (
            "Longer lead time should produce equal or higher stockout risk."
        )

    def test_zero_demand_no_stockout(self):
        score = compute_stockout_risk_score(
            on_hand=0, on_order=0, lead_time_days=14,
            weekly_forecasts=[0.0] * 8, demand_std_weekly=0.0,
        )
        assert score == 0.0


class TestOverstockRiskScore:

    def test_no_overstock_when_low_stock(self):
        score = compute_overstock_risk_score(
            on_hand=10,
            on_order=0,
            weekly_forecasts=[50.0] * 8,
        )
        assert score == 0.0, (
            f"Expected 0.0 for low-stock SKU, got {score}"
        )

    def test_high_overstock_when_excessive_stock(self):
        score = compute_overstock_risk_score(
            on_hand=10_000,
            on_order=0,
            weekly_forecasts=[1.0] * 8,
        )
        assert score == 1.0, (
            f"Expected 1.0 for massively overstocked SKU, got {score}"
        )

    def test_overstock_score_in_range(self):
        score = compute_overstock_risk_score(
            on_hand=500,
            on_order=100,
            weekly_forecasts=[30.0] * 8,
        )
        assert 0.0 <= score <= 1.0, (
            f"Overstock risk score out of [0, 1] range: {score}"
        )

    def test_overstock_score_is_float(self):
        score = compute_overstock_risk_score(
            on_hand=200,
            on_order=50,
            weekly_forecasts=[20.0] * 8,
        )
        assert isinstance(score, float), f"Expected float, got {type(score)}"

    def test_exact_balance_no_overstock(self):
        weekly_fcst = [10.0] * 8   # total = 80
        score = compute_overstock_risk_score(
            on_hand=80, on_order=0, weekly_forecasts=weekly_fcst,
        )
        assert score == 0.0, (
            f"Exactly-balanced SKU should have 0.0 overstock risk, got {score}"
        )

    def test_more_stock_higher_risk(self):
        score_low = compute_overstock_risk_score(
            on_hand=100, on_order=0, weekly_forecasts=[20.0] * 8,
        )
        score_high = compute_overstock_risk_score(
            on_hand=500, on_order=0, weekly_forecasts=[20.0] * 8,
        )
        assert score_high >= score_low, (
            "More stock should produce equal or higher overstock risk."
        )



class TestAssignQuadrant:

    def test_reorder_now_quadrant(self):
        quadrant, action = assign_quadrant(stockout_risk=0.8, overstock_risk=0.2)
        assert quadrant == "reorder_now", (
            f"Expected 'reorder_now', got '{quadrant}'"
        )
        assert "REORDER" in action.upper(), (
            f"Expected 'REORDER' in action text, got: '{action}'"
        )

    def test_markdown_clear_quadrant(self):
        """Low stockout, high overstock → markdown_clear."""
        quadrant, action = assign_quadrant(stockout_risk=0.2, overstock_risk=0.8)
        assert quadrant == "markdown_clear", (
            f"Expected 'markdown_clear', got '{quadrant}'"
        )
        assert ("MARKDOWN" in action.upper() or "CLEAR" in action.upper()), (
            f"Expected 'MARKDOWN' or 'CLEAR' in action text, got: '{action}'"
        )

    def test_watch_volatile_quadrant(self):
        """High stockout, high overstock → watch_volatile."""
        quadrant, action = assign_quadrant(stockout_risk=0.7, overstock_risk=0.7)
        assert quadrant == "watch_volatile", (
            f"Expected 'watch_volatile', got '{quadrant}'"
        )

    def test_healthy_quadrant(self):
        """Low stockout, low overstock → healthy."""
        quadrant, action = assign_quadrant(stockout_risk=0.1, overstock_risk=0.1)
        assert quadrant == "healthy", (
            f"Expected 'healthy', got '{quadrant}'"
        )

    def test_boundary_stockout_at_threshold(self):
        """stockout_risk == 0.5 is the boundary — should be reorder_now (≥ 0.5)."""
        quadrant, _ = assign_quadrant(stockout_risk=0.5, overstock_risk=0.0)
        assert quadrant == "reorder_now"

    def test_boundary_overstock_at_threshold(self):
        """overstock_risk == 0.5 is the boundary — should be markdown_clear (≥ 0.5)."""
        quadrant, _ = assign_quadrant(stockout_risk=0.0, overstock_risk=0.5)
        assert quadrant == "markdown_clear"

    def test_both_at_threshold_is_watch(self):
        """Both exactly at 0.5 → watch_volatile."""
        quadrant, _ = assign_quadrant(stockout_risk=0.5, overstock_risk=0.5)
        assert quadrant == "watch_volatile"

    def test_return_types(self):
        """Both return values must be strings."""
        quadrant, action = assign_quadrant(stockout_risk=0.6, overstock_risk=0.3)
        assert isinstance(quadrant, str)
        assert isinstance(action, str)

    @pytest.mark.parametrize("so,os,expected_q", [
        (0.9, 0.1, "reorder_now"),
        (0.1, 0.9, "markdown_clear"),
        (0.9, 0.9, "watch_volatile"),
        (0.0, 0.0, "healthy"),
        (0.5, 0.49, "reorder_now"),
        (0.49, 0.5, "markdown_clear"),
    ])
    def test_quadrant_parametrized(self, so, os, expected_q):
        quadrant, _ = assign_quadrant(so, os)
        assert quadrant == expected_q, (
            f"assign_quadrant({so}, {os}) → expected '{expected_q}', got '{quadrant}'"
        )


# ===========================================================================
# compute_value_at_stake
# ===========================================================================

class TestComputeValueAtStake:

    def test_value_at_stake_healthy_is_zero(self):
        """Healthy SKU → value at stake must always be 0.0."""
        val = compute_value_at_stake(
            quadrant="healthy",
            stockout_risk=0.1,
            overstock_risk=0.1,
            weekly_forecasts=[10.0] * 8,
            on_hand=200,
            on_order=0,
            lead_time_days=7,
            unit_cost=100,
            list_price=200,
        )
        assert val == 0.0, f"Expected 0.0 for healthy quadrant, got {val}"

    def test_value_at_stake_positive_for_reorder_now(self):
        """reorder_now with zero stock and high demand → positive value."""
        val = compute_value_at_stake(
            quadrant="reorder_now",
            stockout_risk=0.9,
            overstock_risk=0.1,
            weekly_forecasts=[100.0] * 8,
            on_hand=0,
            on_order=0,
            lead_time_days=14,
            unit_cost=100,
            list_price=200,
        )
        assert val > 0, f"Expected positive value_at_stake for reorder_now, got {val}"

    def test_value_at_stake_positive_for_markdown_clear(self):
        """markdown_clear with massive excess stock → positive value."""
        val = compute_value_at_stake(
            quadrant="markdown_clear",
            stockout_risk=0.1,
            overstock_risk=0.9,
            weekly_forecasts=[5.0] * 8,
            on_hand=1000,
            on_order=0,
            lead_time_days=7,
            unit_cost=150,
            list_price=300,
        )
        assert val > 0, f"Expected positive value_at_stake for markdown_clear, got {val}"

    def test_value_at_stake_non_negative_all_quadrants(self):
        """Value at stake must be non-negative for every possible quadrant."""
        for quadrant in ["reorder_now", "markdown_clear", "watch_volatile", "healthy"]:
            val = compute_value_at_stake(
                quadrant=quadrant,
                stockout_risk=0.5,
                overstock_risk=0.5,
                weekly_forecasts=[20.0] * 8,
                on_hand=50,
                on_order=10,
                lead_time_days=10,
                unit_cost=200,
                list_price=400,
            )
            assert val >= 0.0, (
                f"Negative value_at_stake={val} for quadrant='{quadrant}'"
            )

    def test_reorder_now_uses_list_price(self):
        """
        For reorder_now, lost revenue = units_at_risk × list_price (not unit_cost).
        With list_price=300 vs unit_cost=100, the value must scale with list_price.
        """
        val_high_price = compute_value_at_stake(
            quadrant="reorder_now",
            stockout_risk=0.9, overstock_risk=0.1,
            weekly_forecasts=[50.0] * 8,
            on_hand=0, on_order=0, lead_time_days=7,
            unit_cost=100, list_price=300,
        )
        val_low_price = compute_value_at_stake(
            quadrant="reorder_now",
            stockout_risk=0.9, overstock_risk=0.1,
            weekly_forecasts=[50.0] * 8,
            on_hand=0, on_order=0, lead_time_days=7,
            unit_cost=100, list_price=100,
        )
        assert val_high_price > val_low_price, (
            "Higher list_price should produce higher stockout value_at_stake."
        )

    def test_markdown_clear_uses_unit_cost(self):
        """
        For markdown_clear, locked capital = excess_units × unit_cost (not list_price).
        With unit_cost=300 vs 100, the value must scale with unit_cost.
        """
        val_high_cost = compute_value_at_stake(
            quadrant="markdown_clear",
            stockout_risk=0.1, overstock_risk=0.9,
            weekly_forecasts=[2.0] * 8,
            on_hand=500, on_order=0, lead_time_days=7,
            unit_cost=300, list_price=600,
        )
        val_low_cost = compute_value_at_stake(
            quadrant="markdown_clear",
            stockout_risk=0.1, overstock_risk=0.9,
            weekly_forecasts=[2.0] * 8,
            on_hand=500, on_order=0, lead_time_days=7,
            unit_cost=100, list_price=600,
        )
        assert val_high_cost > val_low_cost, (
            "Higher unit_cost should produce higher overstock value_at_stake."
        )

    def test_return_type_is_float(self):
        val = compute_value_at_stake(
            quadrant="reorder_now",
            stockout_risk=0.8, overstock_risk=0.2,
            weekly_forecasts=[30.0] * 8,
            on_hand=10, on_order=0, lead_time_days=14,
            unit_cost=200, list_price=400,
        )
        assert isinstance(val, float), f"Expected float, got {type(val)}"
