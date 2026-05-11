"""
data_loader.py — Data ingestion, validation, and cleaning.
Supports yfinance live fetch and CSV upload (Yahoo Finance format).
"""

import pandas as pd
import yfinance as yf
import logging
from typing import Optional

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Date", "Close"}


def fetch_live(ticker: str, period: str) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 10:
        raise ValueError(f"Invalid ticker symbol: '{ticker}'")
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception as e:
        raise ConnectionError(f"Could not fetch data for '{ticker}'. Check your connection.")
    if raw.empty:
        raise ValueError(f"No data for '{ticker}'. It may be delisted or misspelled.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.reset_index(inplace=True)
    return _clean(raw)


def load_csv(file) -> pd.DataFrame:
    """
    Load Yahoo Finance CSV export.
    Handles the known yfinance double-header format:
      Row 0: Date, Close, High, Low, Open, Volume   (headers)
      Row 1: <empty>, AAPL, AAPL, AAPL, AAPL, AAPL  (ticker row — drop this)
      Row 2+: actual data
    """
    try:
        df = pd.read_csv(file, header=0)
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")

    # Normalise column names first
    df.columns = [str(c).strip().title() for c in df.columns]

    # Drop the yfinance ticker-name row — it has a null/empty Date
    # and non-numeric values in Close column
    if len(df) > 0:
        first_date = str(df.iloc[0].get("Date", "")).strip()
        first_close = str(df.iloc[0].get("Close", "")).strip()
        # Row is a ticker row if Date is empty OR Close is a ticker symbol (letters only)
        if (first_date == "" or first_date == "nan" or
                (first_close.isalpha() and first_close.isupper())):
            df = df.iloc[1:].reset_index(drop=True)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. "
            f"Need at minimum: Date and Close."
        )

    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    if len(df) < 20:
        raise ValueError("Not enough data (need at least 20 rows).")
    return df
