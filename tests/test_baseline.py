"""
tests/test_baseline.py
----------------------
Unit tests for src/baseline.py — seasonal_naive_forecast().

The most important test is test_no_future_leakage: it verifies that the
baseline function NEVER reads data after as_of_date, which is the
non-negotiable requirement of this engagement (brief Section 7.1).

Run with:
    pytest tests/test_baseline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.baseline import seasonal_naive_forecast


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_panel(n_weeks: int = 60, units_per_day: float = 10.0) -> pd.DataFrame:
    """
    Create a minimal 1-SKU daily panel with constant demand.

    Parameters
    ----------
    n_weeks       : total history length in weeks
    units_per_day : constant daily demand to assign
    """
    dates = pd.date_range("2024-01-01", periods=n_weeks * 7, freq="D")
    return pd.DataFrame(
        {
            "date":       dates,
            "sku_id":     "SKU0001",
            "units_sold": float(units_per_day),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_future_leakage():
    """
    Baseline forecast must NEVER use data after as_of_date.

    We 'poison' all future rows with units_sold=9999.
    If any leakage occurs, baseline_yhat will equal 9999 × 7 = 69,993.
    """
    panel = make_panel(n_weeks=60)
    as_of_date = pd.Timestamp("2024-12-01")

    # Poison future rows
    panel.loc[panel["date"] > as_of_date, "units_sold"] = 9999.0

    result = seasonal_naive_forecast(panel, "SKU0001", as_of_date, horizon_weeks=4)

    assert (result["baseline_yhat"] != 9999 * 7).all(), (
        "Leakage detected! The baseline used data after as_of_date."
    )


def test_output_columns():
    """Output DataFrame must contain all four required columns."""
    panel = make_panel(n_weeks=60)
    as_of_date = pd.Timestamp("2024-12-01")
    result = seasonal_naive_forecast(panel, "SKU0001", as_of_date)

    required = {"week_number", "week_start", "sku_id", "baseline_yhat"}
    assert required.issubset(set(result.columns)), (
        f"Missing columns: {required - set(result.columns)}"
    )


def test_output_rows():
    """Output must have exactly horizon_weeks rows."""
    panel = make_panel(n_weeks=60)
    as_of_date = pd.Timestamp("2024-12-01")
    result = seasonal_naive_forecast(panel, "SKU0001", as_of_date, horizon_weeks=8)
    assert len(result) == 8, f"Expected 8 rows, got {len(result)}"


def test_constant_demand_prediction():
    """
    With constant demand of 10 units/day, the baseline should predict
    10 × 7 = 70 units per week (within abs tolerance of 1.0 to allow
    for any week that falls on a partial history boundary).
    """
    panel = make_panel(n_weeks=60, units_per_day=10.0)
    as_of_date = pd.Timestamp("2024-12-01")
    result = seasonal_naive_forecast(panel, "SKU0001", as_of_date, horizon_weeks=4)

    assert (result["baseline_yhat"] == pytest.approx(70.0, abs=1.0)).all(), (
        f"Expected ~70.0 units/week for constant demand. Got: {result['baseline_yhat'].tolist()}"
    )