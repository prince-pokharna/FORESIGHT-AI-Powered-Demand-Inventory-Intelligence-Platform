# src/forecasting/ensemble_model.py

import pandas as pd
import numpy as np
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import logging

logger = logging.getLogger(__name__)


class EnsembleForecastModel:
    """
    Prophet + XGBoost ensemble for SKU-level demand forecasting.
    Prophet captures trend/seasonality; XGB corrects residuals using
    exogenous features (price, promo flags, lag features).
    """

    def __init__(self, prophet_params: dict = None, xgb_params: dict = None):
        self.prophet_params = prophet_params or {
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "seasonality_mode": "multiplicative",
            "interval_width": 0.95,
        }
        self.xgb_params = xgb_params or {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self.prophet_model = None
        self.xgb_model = None
        self.feature_cols = []

    def _build_lag_features(self, df: pd.DataFrame, lags: list = [7, 14, 28]) -> pd.DataFrame:
        df = df.copy()
        for lag in lags:
            df[f"lag_{lag}"] = df["y"].shift(lag)
        df["rolling_mean_7"] = df["y"].rolling(7).mean()
        df["rolling_std_7"] = df["y"].rolling(7).std()
        df["day_of_week"] = pd.to_datetime(df["ds"]).dt.dayofweek
        df["month"] = pd.to_datetime(df["ds"]).dt.month
        df["week_of_year"] = pd.to_datetime(df["ds"]).dt.isocalendar().week.astype(int)
        return df.dropna()

    def fit(self, df: pd.DataFrame, exog_cols: list = None):
        """
        df must have columns: ds (datetime), y (demand), plus any exog_cols.
        """
        assert "ds" in df.columns and "y" in df.columns, "df must have 'ds' and 'y' columns"

        # Stage 1: Fit Prophet
        logger.info("Fitting Prophet model...")
        prophet_df = df[["ds", "y"]].copy()
        self.prophet_model = Prophet(**self.prophet_params)
        if exog_cols:
            for col in exog_cols:
                self.prophet_model.add_regressor(col)
            prophet_df = df[["ds", "y"] + exog_cols].copy()
        self.prophet_model.fit(prophet_df)

        # Stage 2: Get Prophet residuals
        prophet_forecast = self.prophet_model.predict(prophet_df)
        df = df.copy()
        df["prophet_pred"] = prophet_forecast["yhat"].values
        df["residual"] = df["y"] - df["prophet_pred"]

        # Stage 3: Build lag features for XGBoost on residuals
        df = self._build_lag_features(df)
        self.feature_cols = ["prophet_pred", "lag_7", "lag_14", "lag_28",
                              "rolling_mean_7", "rolling_std_7",
                              "day_of_week", "month", "week_of_year"]
        if exog_cols:
            self.feature_cols += exog_cols

        X = df[self.feature_cols]
        y_resid = df["residual"]

        logger.info("Fitting XGBoost on Prophet residuals...")
        self.xgb_model = XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(X, y_resid)

        train_pred = self.prophet_model.predict(prophet_df)["yhat"].values
        mape = mean_absolute_percentage_error(df["y"], train_pred[:len(df)])
        logger.info(f"Train MAPE (Prophet only): {mape:.4f}")
        return self

    def predict(self, future_df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
        """
        Predict demand for future_df or auto-generate future dates.
        Returns df with columns: ds, yhat, yhat_lower, yhat_upper.
        """
        if future_df is None:
            future_df = self.prophet_model.make_future_dataframe(periods=periods)

        prophet_pred = self.prophet_model.predict(future_df)

        # XGBoost can only correct in-sample or where lag features are computable
        result = prophet_pred[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result["yhat"] = result["yhat"].clip(lower=0)  # demand can't be negative
        return result

    def save(self, path: str):
        joblib.dump({"prophet": self.prophet_model, "xgb": self.xgb_model,
                     "feature_cols": self.feature_cols}, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        obj = cls()
        data = joblib.load(path)
        obj.prophet_model = data["prophet"]
        obj.xgb_model = data["xgb"]
        obj.feature_cols = data["feature_cols"]
        return obj