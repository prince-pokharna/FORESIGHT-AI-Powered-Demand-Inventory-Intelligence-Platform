# src/inventory/risk_engine.py

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class InventoryRiskReport:
    sku_id: str
    safety_stock: float
    reorder_point: float
    stockout_probability: float
    overstock_units: float
    risk_label: str  # "CRITICAL", "HIGH", "MEDIUM", "LOW"
    recommended_order_qty: float


class InventoryRiskEngine:
    """
    Computes safety stock, reorder point, stockout probability,
    and EOQ (Economic Order Quantity) per SKU.
    Uses the Z-score method for safety stock under demand uncertainty.
    """

    SERVICE_LEVEL_Z = {
        "95%": 1.645,
        "97%": 1.88,
        "99%": 2.326,
        "99.9%": 3.09,
    }

    def __init__(self, service_level: str = "97%", lead_time_days: int = 7):
        self.z = self.SERVICE_LEVEL_Z.get(service_level, 1.88)
        self.lead_time = lead_time_days

    def compute_safety_stock(self, demand_std: float, lead_time_std: float = 0.0,
                              avg_demand: float = 0.0) -> float:
        """
        Safety Stock = Z * sqrt(L * σ_d² + d² * σ_L²)
        where L=lead_time, σ_d=demand_std, d=avg_demand, σ_L=lead_time_std
        """
        variance = (self.lead_time * demand_std ** 2) + (avg_demand ** 2 * lead_time_std ** 2)
        return round(self.z * np.sqrt(variance), 2)

    def compute_reorder_point(self, avg_daily_demand: float, safety_stock: float) -> float:
        return round(avg_daily_demand * self.lead_time + safety_stock, 2)

    def compute_eoq(self, annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
        """Wilson EOQ formula."""
        if holding_cost_per_unit <= 0 or annual_demand <= 0:
            return 0.0
        return round(np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit), 2)

    def compute_stockout_probability(self, current_stock: float, avg_demand: float,
                                      demand_std: float) -> float:
        """P(demand > current_stock) during lead time."""
        lead_demand_mean = avg_demand * self.lead_time
        lead_demand_std = demand_std * np.sqrt(self.lead_time)
        if lead_demand_std == 0:
            return 0.0 if current_stock >= lead_demand_mean else 1.0
        prob = 1 - stats.norm.cdf(current_stock, loc=lead_demand_mean, scale=lead_demand_std)
        return round(float(prob), 4)

    def _classify_risk(self, stockout_prob: float, overstock_units: float) -> str:
        if stockout_prob > 0.25:
            return "CRITICAL"
        elif stockout_prob > 0.10:
            return "HIGH"
        elif overstock_units > 100:
            return "MEDIUM"
        return "LOW"

    def evaluate_sku(self, sku_id: str, current_stock: float,
                     demand_history: pd.Series,
                     order_cost: float = 50.0,
                     holding_cost_per_unit: float = 2.0) -> InventoryRiskReport:
        avg_demand = demand_history.mean()
        demand_std = demand_history.std()
        annual_demand = avg_demand * 365

        safety_stock = self.compute_safety_stock(demand_std, avg_demand=avg_demand)
        rop = self.compute_reorder_point(avg_demand, safety_stock)
        stockout_prob = self.compute_stockout_probability(current_stock, avg_demand, demand_std)
        eoq = self.compute_eoq(annual_demand, order_cost, holding_cost_per_unit)

        expected_lead_demand = avg_demand * self.lead_time
        overstock = max(0.0, current_stock - expected_lead_demand - safety_stock)

        risk_label = self._classify_risk(stockout_prob, overstock)

        return InventoryRiskReport(
            sku_id=sku_id,
            safety_stock=safety_stock,
            reorder_point=rop,
            stockout_probability=stockout_prob,
            overstock_units=round(overstock, 2),
            risk_label=risk_label,
            recommended_order_qty=eoq,
        )

    def batch_evaluate(self, inventory_df: pd.DataFrame,
                        demand_df: pd.DataFrame) -> pd.DataFrame:
        """
        inventory_df: columns [sku_id, current_stock, order_cost, holding_cost]
        demand_df: columns [sku_id, date, demand]
        Returns DataFrame of InventoryRiskReport fields.
        """
        results = []
        for _, row in inventory_df.iterrows():
            sku = row["sku_id"]
            sku_demand = demand_df[demand_df["sku_id"] == sku]["demand"]
            if sku_demand.empty:
                continue
            report = self.evaluate_sku(
                sku_id=sku,
                current_stock=row.get("current_stock", 0),
                demand_history=sku_demand,
                order_cost=row.get("order_cost", 50.0),
                holding_cost_per_unit=row.get("holding_cost", 2.0),
            )
            results.append(report.__dict__)
        return pd.DataFrame(results)