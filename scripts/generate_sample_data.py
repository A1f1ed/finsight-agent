"""生成模拟港股日线行情数据（作品集演示用，非真实行情）。

用法:
    python scripts/generate_sample_data.py

生成 data/hk_stocks_sample.csv，包含 3 只虚构港股约 6 个月的日线
OHLCV 数据，价格用几何随机游走模拟，保证可复现（固定随机种子）。
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
TRADING_DAYS = 126  # 约 6 个月交易日

# 虚构股票: (代码, 名称, 起始价, 年化漂移, 年化波动率)
STOCKS = [
    ("HK0001", "HKTech Holdings", 120.0, 0.25, 0.35),
    ("HK0002", "Pearl River Bank", 45.0, 0.08, 0.20),
    ("HK0003", "Orient Green Energy", 18.0, -0.05, 0.45),
]


def generate_one(code: str, name: str, start_price: float, mu: float, sigma: float,
                 dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """用几何随机游走生成单只股票的日线数据。"""
    dt = 1 / 252
    daily_ret = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(len(dates)))
    close = start_price * np.cumprod(daily_ret)

    # 开盘价 = 前收盘 + 小幅跳空；高低价围绕开收波动
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
    dates = pd.bdate_range("2025-01-02", periods=TRADING_DAYS)  # 仅工作日，近似交易日历

    frames = [generate_one(*stock, dates, rng) for stock in STOCKS]
    df = pd.concat(frames, ignore_index=True).sort_values(["Date", "Symbol"])

    out = Path(__file__).resolve().parent.parent / "data" / "hk_stocks_sample.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Generated {out}: {len(df)} rows, {df['Symbol'].nunique()} stocks, "
          f"{df['Date'].min()} ~ {df['Date'].max()}")


if __name__ == "__main__":
    main()
