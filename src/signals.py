"""
signals.py — Rule-based signal detection and insight generation.
Returns structured dicts that app.py can render — no Streamlit imports here.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


def detect_crossovers(df: pd.DataFrame, short: int, long: int) -> List[Dict]:
    """Return a list of golden/death cross events with dates."""
    events = []
    pos = df.get("Position")
    if pos is None:
        return events

    crosses = df[pos.isin([2, -2])].copy()
    for _, row in crosses.iterrows():
        kind = "Golden Cross 🟡" if row["Position"] == 2 else "Death Cross 💀"
        events.append({
            "date": row["Date"].strftime("%Y-%m-%d"),
            "type": kind,
            "price": f"${row['Close']:.2f}",
        })
    return events[-10:]  # last 10 events


def generate_insights(df: pd.DataFrame, stats: dict, short: int, long: int) -> List[Dict]:
    """
    Generate human-readable insight cards.
    Each insight has: title, body, sentiment ('positive'|'negative'|'neutral').
    """
    insights = []
    close = df["Close"]
    short_col = f"SMA_{short}"
    long_col = f"SMA_{long}"

    # Trend
    if short_col in df.columns and long_col in df.columns:
        last_short = df[short_col].dropna().iloc[-1]
        last_long = df[long_col].dropna().iloc[-1]
        if last_short > last_long:
            insights.append({
                "title": "Bullish Trend",
                "body": f"The {short}-day MA (${last_short:.2f}) is above the {long}-day MA (${last_long:.2f}), indicating bullish momentum.",
                "sentiment": "positive",
            })
        else:
            insights.append({
                "title": "Bearish Trend",
                "body": f"The {short}-day MA (${last_short:.2f}) is below the {long}-day MA (${last_long:.2f}), indicating bearish momentum.",
                "sentiment": "negative",
            })

    # RSI
    if "RSI" in df.columns:
        rsi = df["RSI"].dropna().iloc[-1]
        if rsi > 70:
            insights.append({
                "title": "Overbought (RSI)",
                "body": f"RSI is {rsi:.1f} — above 70 suggests the stock may be overbought. Consider risk management.",
                "sentiment": "negative",
            })
        elif rsi < 30:
            insights.append({
                "title": "Oversold (RSI)",
                "body": f"RSI is {rsi:.1f} — below 30 may indicate oversold conditions and a potential reversal.",
                "sentiment": "positive",
            })
        else:
            insights.append({
                "title": "RSI Neutral",
                "body": f"RSI at {rsi:.1f} is in neutral territory (30–70), no extreme signal.",
                "sentiment": "neutral",
            })

    # Volatility spike
    if "Rolling_Vol" in df.columns:
        vol_series = df["Rolling_Vol"].dropna()
        current_vol = vol_series.iloc[-1]
        avg_vol = vol_series.mean()
        if current_vol > avg_vol * 1.5:
            insights.append({
                "title": "High Volatility Alert",
                "body": f"Current annualised volatility ({current_vol:.1f}%) is significantly above the average ({avg_vol:.1f}%). Elevated risk.",
                "sentiment": "negative",
            })

    # Sharpe
    sharpe = stats.get("sharpe_ratio", 0)
    if sharpe > 1.5:
        insights.append({
            "title": "Strong Risk-Adjusted Return",
            "body": f"Sharpe ratio of {sharpe:.2f} indicates excellent risk-adjusted performance.",
            "sentiment": "positive",
        })
    elif sharpe < 0:
        insights.append({
            "title": "Negative Sharpe Ratio",
            "body": f"Sharpe ratio of {sharpe:.2f} suggests returns are not compensating for the risk taken.",
            "sentiment": "negative",
        })

    # Max drawdown
    max_dd = stats.get("max_drawdown", 0)
    if max_dd < -30:
        insights.append({
            "title": "Severe Drawdown Detected",
            "body": f"Maximum drawdown of {max_dd:.1f}% observed — significant peak-to-trough decline in this period.",
            "sentiment": "negative",
        })

    return insights
