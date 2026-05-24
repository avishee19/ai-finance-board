import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = {"Date", "Close"}

def fetch_live(ticker, period):
    ticker = ticker.strip().upper()
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    except Exception as e:
        raise ConnectionError(f"Could not fetch {ticker}")
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    if "Date" not in raw.columns:
        for c in ["Datetime", "index", "datetime"]:
            if c in raw.columns:
                raw = raw.rename(columns={c: "Date"})
                break
    return _clean(raw)

def load_csv(file):
    try:
        df = pd.read_csv(file, header=0)
    except Exception as e:
        raise ValueError(f"Could not parse CSV: {e}")
    df.columns = [str(c).strip().title() for c in df.columns]
    if len(df) > 0:
        fd = str(df.iloc[0].get("Date", "")).strip()
        fc = str(df.iloc[0].get("Close", "")).strip()
        if fd in ("", "nan") or (fc.isalpha() and fc.isupper()):
            df = df.iloc[1:].reset_index(drop=True)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing: {missing}")
    return _clean(df)

def _clean(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    for col in ["Close", "Open", "High", "Low", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close"])
    if len(df) < 20:
        raise ValueError("Not enough data")
    return df
