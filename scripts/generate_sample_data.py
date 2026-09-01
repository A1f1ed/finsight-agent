"""Generate simulated HK daily OHLCV data (portfolio demo only, not real market data).

Usage:
    python scripts/generate_sample_data.py

Produces data/hk_stocks_sample.csv with ~6 months of daily OHLCV for 3 fictional
HK stocks; prices follow a geometric random walk and the output is reproducible
(fixed random seed).
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
TRADING_DAYS = 126  # roughly 6 months of trading days

# Fictional stocks: (code, name, start price, annual drift, annual volatility)
STOCKS = [
    ("HK0001", "HKTech Holdings", 120.0, 0.25, 0.35),
    ("HK0002", "Pearl River Bank", 45.0, 0.08, 0.20),
    ("HK0003", "Orient Green Energy", 18.0, -0.05, 0.45),
]


def generate_one(code: str, name: str, start_price: float, mu: float, sigma: float,
                 dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Generate daily data for a single stock via geometric random walk."""
    dt = 1 / 252
    daily_ret = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(len(dates)))
    close = start_price * np.cumprod(daily_ret)

    # Open = previous close + small gap; high/low oscillate around open/close
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + 0.004 * rng.standard_normal(len(dates)))
    high = np.maximum(open_, close) * (1 + np.abs(0.008 * rng.standard_normal(len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(0.008 * rng.standard_normal(len(dates))))
    volume = (2_000_000 * (1 + 0.6 * np.abs(rng.standard_normal(len(dates))))).astype(int)

    return pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Symbol": code,
        "Name": name,
        "Open": open_.round(2),
        "High": high.round(2),
        "Low": low.round(2),
        "Close": close.round(2),
        "Volume": volume,
    })


def main() -> None:
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2025-01-02", periods=TRADING_DAYS)  # business days only, approximating the trading calendar

    frames = [generate_one(*stock, dates, rng) for stock in STOCKS]
    df = pd.concat(frames, ignore_index=True).sort_values(["Date", "Symbol"])

    out = Path(__file__).resolve().parent.parent / "data" / "hk_stocks_sample.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {out}: {len(df)} rows, {df['Symbol'].nunique()} stocks, "
          f"{df['Date'].min()} ~ {df['Date'].max()}")


if __name__ == "__main__":
    main()
