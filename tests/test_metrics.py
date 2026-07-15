"""
tests/test_metrics.py
---------------------
Unit tests for src/metrics.py.

Run with:
    pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from src.metrics import wape, mape, bias


def test_wape_perfect_forecast():
    """Perfect forecast → WAPE must be exactly 0.0."""
    actual   = np.array([10.0, 20.0, 30.0])
    forecast = np.array([10.0, 20.0, 30.0])
    assert wape(actual, forecast) == 0.0


def test_wape_known_value():
    """
    Hand-computed case:
        |10-8| + |20-25| + |30-30| = 2 + 5 + 0 = 7
        sum(actual) = 60
        WAPE = 7/60
    """
    actual   = np.array([10.0, 20.0, 30.0])
    forecast = np.array([8.0,  25.0, 30.0])
    expected = 7.0 / 60.0
    assert abs(wape(actual, forecast) - expected) < 1e-9


def test_wape_zero_actual_raises():
    """WAPE is undefined when all actual values are zero — must raise ValueError."""
    with pytest.raises(ValueError):
        wape(np.array([0.0, 0.0]), np.array([1.0, 2.0]))


def test_bias_positive():
    """Model over-forecasts → bias must be positive."""
    actual   = np.array([10.0, 10.0])
    forecast = np.array([12.0, 14.0])
    # mean(forecast - actual) = mean([2, 4]) = 3.0
    assert bias(actual, forecast) == pytest.approx(3.0)


def test_bias_negative():
    """Model under-forecasts → bias must be negative."""
    actual   = np.array([10.0, 10.0])
    forecast = np.array([8.0,  6.0])
    # mean(forecast - actual) = mean([-2, -4]) = -3.0
    assert bias(actual, forecast) == pytest.approx(-3.0)


def test_mape_excludes_zeros():
    """
    Rows where actual == 0 must be excluded from MAPE.
    Hand-computed:
        Row 0: actual=0 → excluded
        Row 1: |10-12|/10 = 0.2
        Row 2: |20-18|/20 = 0.1
        mean([0.2, 0.1]) = 0.15
    """
    actual   = np.array([0.0, 10.0, 20.0])
    forecast = np.array([5.0, 12.0, 18.0])
    assert abs(mape(actual, forecast) - 0.15) < 1e-9