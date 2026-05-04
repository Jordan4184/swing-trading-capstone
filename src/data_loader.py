"""
Data loader for swing trading capstone.

Reponsibilities:
- Define the trading universe (which stocks)
- Download historical OHLCV (Open/High/Low/Close/Volume) from Yahoo Finance
- Cache data locally so we don't re-download on every run

Author: Jordan Donaldson
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

# Universe Definition

UNIVERSE = [
    # Tech
    "AAPL",
    "TSLA",
    "NVDA",
    "META",
    # Finance
    "JPM",
    # Healthcare
    "JNJ",
    "UNH",
    "PFE",
    # Consumer
    "AMZN",
    "MCD",
    # Benchmark ETF
    "SPY",
]


def get_universe() -> list[str]:
    """Returns list of tickers traded"""
    return UNIVERSE.copy()


# Data download + caching


DATA_DIR = Path(__file__).parent.parent / "Data" / "raw"


def download_data(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Downloaded daily OHLCV data from Yahoo Finance.

    Returned a long-format DataFrame with columns:
        date, ticker, open, high, low, close, volume

    Long format (one row per ticker-date) is easier
    to work with than yfinance's default  wide format
    (multi-index columns).
    """

    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # adjusts for splits or dividends
        progress=False,
        group_by="ticker",
    )

    # Reshape from wide multi-index columns to long format

    frames = []
    for ticker in tickers:
        if ticker not in raw.columns.get_level_values(0):
            print(f"WARNING: no data for {ticker}, skipping")
            continue
        df = raw[ticker].copy()
        df["ticker"] = ticker
        df = df.reset_index()
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.lower() for c in out.columns]
    out = out[["date", "ticker", "open", "high", "low", "close", "volume"]]
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"Downloaded {len(out):,} rows.")
    return out


def load_data(
    tickers: list[str] | None = None,
    start: str = "2018-01-01",
    end: str = "2025-12-31",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data, using cached parquet file if it exists,

    Pipeline calls function. Allows for iterating features wihtout re-downloading
    """

    if tickers is None:
        tickers = get_universe()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = DATA_DIR / "ohlcv.parquet"

    if use_cache and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    df = download_data(tickers, start, end)
    df.to_parquet(cache_path, index=False)
    print(f"Saved cache to {cache_path}")
    return df


# CLI entry point:

if __name__ == "__main__":
    df = load_data(use_cache=False)
    print("\n=== Sample ===")
    print(df.head())
    print("\n=== Shape ===")
    print(df.shape)
    print("\n=== Tickers ===")
    print(df["ticker"].unique())
    print("\n=== Date range ===")
    print(f"{df['date'].min()} → {df['date'].max()}")
