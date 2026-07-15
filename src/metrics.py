"""
src/metrics.py
--------------
Forecast accuracy metrics for Project FORESIGHT.

All functions are pure (no side effects, no I/O) and accept numpy arrays.
They are imported by src/baseline.py, src/forecast.py, and the test suite.

Primary metric:   WAPE  (per brief Appendix B)
Secondary metric: MAPE
Diagnostic:       Bias
"""

import numpy as np


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.

    Formula
    -------
        WAPE = sum(|actual - forecast|) / sum(actual)

    Robust to low-volume SKUs where MAPE explodes near zero.
    Primary accuracy metric for this engagement (brief Appendix B).

    Parameters
    ----------
    actual   : array of true demand values
    forecast : array of predicted demand values, same length

    Returns
    -------
    float in [0, ∞)  — lower is better; 0 = perfect forecast

    Raises
    ------
    ValueError if sum(actual) == 0  (WAPE is undefined when total demand is zero)
    """
    actual   = np.asarray(actual,   dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    total_actual = actual.sum()
    if total_actual == 0.0:
        raise ValueError(
            "WAPE is undefined: sum(actual) == 0. "
            "Check that the actual array contains non-zero demand values."
        )

    return float(np.abs(actual - forecast).sum() / total_actual)


def mape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error.

    Formula
    -------
        MAPE = mean(|actual - forecast| / actual)   for rows where actual > 0

    Rows where actual == 0 are excluded to avoid division by zero.
    Use WAPE as the primary metric — MAPE is reported as a secondary,
    intuitive figure but is unreliable when demand is near zero (brief Appendix B).

    Parameters
    ----------
    actual   : array of true demand values
    forecast : array of predicted demand values, same length

    Returns
    -------
    float in [0, ∞)  — lower is better; returns 0.0 if no valid rows remain
    """
    actual   = np.asarray(actual,   dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    mask = actual > 0.0
    if not mask.any():
        return 0.0

    return float(np.mean(np.abs(actual[mask] - forecast[mask]) / actual[mask]))


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Mean signed forecast error (Bias).

    Formula
    -------
        Bias = mean(forecast - actual)

    Interpretation
    --------------
    Positive → model over-forecasts on average (tends to predict too high)
    Negative → model under-forecasts on average (tends to predict too low)
    Zero     → unbiased (errors cancel out on average)

    Parameters
    ----------
    actual   : array of true demand values
    forecast : array of predicted demand values, same length

    Returns
    -------
    float — signed, units match input arrays (units per week for weekly data)
    """
    actual   = np.asarray(actual,   dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    return float(np.mean(forecast - actual))