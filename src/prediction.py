"""
prediction.py — AI price forecasting using sklearn.
Uses a feature-rich linear model + confidence intervals.
No heavy dependencies (no TensorFlow/PyTorch) — pure sklearn.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-series features for price prediction.
    All features are lag-based to prevent look-ahead bias.
    """
    feat = pd.DataFrame(index=df.index)
    close = df["Close"]

    # Lagged prices
    for lag in [1, 2, 3, 5, 10, 20]:
        feat[f"lag_{lag}"] = close.shift(lag)

    # Rolling statistics
    for w in [5, 10, 20]:
        feat[f"roll_mean_{w}"] = close.shift(1).rolling(w).mean()
        feat[f"roll_std_{w}"] = close.shift(1).rolling(w).std()

    # Momentum
    feat["mom_5"] = close.shift(1) / close.shift(6) - 1
    feat["mom_10"] = close.shift(1) / close.shift(11) - 1
    feat["mom_20"] = close.shift(1) / close.shift(21) - 1

    # RSI (if available)
    if "RSI" in df.columns:
        feat["rsi"] = df["RSI"].shift(1)

    # MACD (if available)
    if "MACD" in df.columns:
        feat["macd"] = df["MACD"].shift(1)

    # Volatility
    if "Rolling_Vol" in df.columns:
        feat["vol"] = df["Rolling_Vol"].shift(1)

    # Day of week / month seasonality
    feat["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek
    feat["month"] = pd.to_datetime(df["Date"]).dt.month

    return feat


def train_and_forecast(
    df: pd.DataFrame,
    horizon: int = 30,
    n_simulations: int = 500,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Train a Ridge regression model and forecast `horizon` trading days ahead.

    Returns:
        forecast_df: DataFrame with columns [Date, Forecast, Lower, Upper]
        r2_score: model R² on held-out data
        directional_acc: % of correct up/down calls on held-out data
    """
    feat = _build_features(df)
    target = df["Close"]

    # Align and drop NaN rows
    combined = pd.concat([feat, target.rename("target")], axis=1).dropna()
    X = combined.drop(columns=["target"]).values
    y = combined["target"].values

    if len(X) < 60:
        raise ValueError("Need at least 60 data points for forecasting.")

    # Time-series cross-validation (no shuffling)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    tscv = TimeSeriesSplit(n_splits=5)
    r2_scores, dir_accs = [], []

    for train_idx, test_idx in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        actuals = y[test_idx]

        ss_res = np.sum((actuals - preds) ** 2)
        ss_tot = np.sum((actuals - actuals.mean()) ** 2)
        r2_scores.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)

        if len(preds) > 1:
            pred_dir = np.sign(np.diff(preds))
            act_dir = np.sign(np.diff(actuals))
            dir_accs.append(np.mean(pred_dir == act_dir))

    # Final model trained on full data
    model.fit(X, y)

    # Monte Carlo simulation for confidence intervals
    last_close = float(df["Close"].iloc[-1])
    daily_vol = float(df["Close"].pct_change().std())
    last_date = pd.to_datetime(df["Date"].iloc[-1])

    # Generate forecast using recursive prediction + MC noise
    all_paths = []
    for _ in range(n_simulations):
        path = [last_close]
        for _ in range(horizon):
            noise = np.random.normal(0, daily_vol)
            next_price = path[-1] * (1 + noise)
            path.append(next_price)
        all_paths.append(path[1:])  # exclude the seed

    paths_arr = np.array(all_paths)
    mean_path = paths_arr.mean(axis=0)
    lower = np.percentile(paths_arr, 10, axis=0)
    upper = np.percentile(paths_arr, 90, axis=0)

    # Business days ahead
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": mean_path,
        "Lower": lower,
        "Upper": upper,
    })

    avg_r2 = float(np.mean(r2_scores))
    avg_dir = float(np.mean(dir_accs)) if dir_accs else 0.5

    return forecast_df, avg_r2, avg_dir
