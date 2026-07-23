# tests/test_risk_engine.py

import pytest
import pandas as pd
import numpy as np
from src.inventory.risk_engine import InventoryRiskEngine


@pytest.fixture
def engine():
    return InventoryRiskEngine(service_level="97%", lead_time_days=7)


@pytest.fixture
def sample_demand():
    np.random.seed(42)
    return pd.Series(np.random.normal(loc=50, scale=10, size=365).clip(min=0))


def test_safety_stock_positive(engine, sample_demand):
    ss = engine.compute_safety_stock(
        demand_std=sample_demand.std(),
        avg_demand=sample_demand.mean()
    )
    assert ss > 0, "Safety stock must be positive"


def test_reorder_point_greater_than_safety_stock(engine, sample_demand):
    avg = sample_demand.mean()
    ss = engine.compute_safety_stock(sample_demand.std(), avg_demand=avg)
    rop = engine.compute_reorder_point(avg, ss)
    assert rop > ss


def test_stockout_prob_range(engine, sample_demand):
    prob = engine.compute_stockout_probability(
        current_stock=200,
        avg_demand=sample_demand.mean(),
        demand_std=sample_demand.std()
    )
    assert 0.0 <= prob <= 1.0


def test_zero_stock_high_stockout(engine, sample_demand):
    prob = engine.compute_stockout_probability(
        current_stock=0,
        avg_demand=sample_demand.mean(),
        demand_std=sample_demand.std()
    )
    assert prob > 0.5, "Zero stock should produce high stockout probability"


def test_eoq_positive(engine):
    eoq = engine.compute_eoq(annual_demand=10000, order_cost=50, holding_cost_per_unit=2)
    assert eoq > 0


def test_evaluate_sku_returns_report(engine, sample_demand):
    report = engine.evaluate_sku("SKU-001", current_stock=100, demand_history=sample_demand)
    assert report.sku_id == "SKU-001"
    assert report.risk_label in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert report.reorder_point > 0


def test_batch_evaluate(engine, sample_demand):
    inventory_df = pd.DataFrame([
        {"sku_id": "SKU-001", "current_stock": 50, "order_cost": 50, "holding_cost": 2},
        {"sku_id": "SKU-002", "current_stock": 500, "order_cost": 75, "holding_cost": 3},
    ])
    demand_df = pd.DataFrame({
        "sku_id": ["SKU-001"] * 365 + ["SKU-002"] * 365,
        "date": list(pd.date_range("2024-01-01", periods=365)) * 2,
        "demand": list(sample_demand) + list(sample_demand * 2),
    })
    result = engine.batch_evaluate(inventory_df, demand_df)
    assert len(result) == 2
    assert "risk_label" in result.columns