"""
processing.py — All financial calculations: MAs, volatility, returns.
All functions are pure: take a DataFrame, return a new DataFrame with added columns.
"""

import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    df = df.copy()
    df[f"SMA_{short}"] = df["Close"].rolling(short).mean()
    df[f"SMA_{long}"] = df["Close"].rolling(long).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    df = df.copy()
    mid = df["Close"].rolling(period).mean()
    std = df["Close"].rolling(period).std()
    df["BB_Upper"] = mid + 2 * std
    df["BB_Mid"] = mid
    df["BB_Lower"] = mid - 2 * std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Mid"]
    return df


def add_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Rolling_Vol"] = df["Daily_Return"].rolling(window).std() * np.sqrt(252) * 100  # annualised %
    return df


def add_signals(df: pd.DataFrame, short: int, long: int) -> pd.DataFrame:
    """MA crossover signals: 1 = bullish, -1 = bearish, 0 = neutral."""
    df = df.copy()
    short_col = f"SMA_{short}"
    long_col = f"SMA_{long}"
    df["Signal"] = 0
    valid = df[short_col].notna() & df[long_col].notna()
    df.loc[valid & (df[short_col] > df[long_col]), "Signal"] = 1
    df.loc[valid & (df[short_col] < df[long_col]), "Signal"] = -1
    df["Position"] = df["Signal"].diff()
    return df


def run_backtest(df: pd.DataFrame, initial: float = 10_000.0) -> pd.DataFrame:
    """
    Simulate a simple MA crossover strategy.
    Ensures alignment between signal and return series to avoid NaN equity.
    """
    df = df.copy()
    if "Daily_Return" not in df.columns:
        df["Daily_Return"] = df["Close"].pct_change()

    # Shift signal by 1 to avoid look-ahead bias
    df["Strategy_Return"] = df["Signal"].shift(1) * df["Daily_Return"]

    # Fill NaN returns with 0 (no trade)
    df["Strategy_Return"] = df["Strategy_Return"].fillna(0)
    df["BuyHold_Return"] = df["Daily_Return"].fillna(0)

    df["Equity_Strategy"] = (1 + df["Strategy_Return"]).cumprod() * initial
    df["Equity_BuyHold"] = (1 + df["BuyHold_Return"]).cumprod() * initial

    return df


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Return a dict of key portfolio statistics."""
    close = df["Close"]
    daily_ret = df.get("Daily_Return", close.pct_change())

    total_return = (close.iloc[-1] / close.iloc[0] - 1) * 100
    ann_vol = daily_ret.std() * np.sqrt(252) * 100
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    max_dd = ((close / close.cummax()) - 1).min() * 100

    return {
        "latest_price": float(close.iloc[-1]),
        "total_return": float(total_return),
        "ann_volatility": float(ann_vol),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "data_points": len(df),
    }
