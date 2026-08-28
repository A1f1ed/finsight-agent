# Technical Indicators Reference

A practical reference for common technical analysis indicators used in equity markets.
All formulas assume daily closing prices unless stated otherwise.

## Moving Average (MA)

The simple moving average smooths price noise over a lookback window of N days:

MA(N) = (P1 + P2 + ... + PN) / N

Common windows: 5, 10, 20, 60, 120, 250 days. The 250-day MA is often called the
"annual line" in Asian markets. A price crossing above a rising MA is commonly read
as bullish momentum; crossing below is read as bearish.

## Exponential Moving Average (EMA)

EMA weights recent prices more heavily:

EMA(t) = alpha * P(t) + (1 - alpha) * EMA(t-1),  alpha = 2 / (N + 1)

EMA reacts faster to price changes than MA, at the cost of more false signals.

## Relative Strength Index (RSI)

RSI measures the speed and magnitude of recent price changes on a 0-100 scale:

RSI = 100 - 100 / (1 + RS),  RS = average gain over N days / average loss over N days

Standard setting is N = 14. RSI above 70 suggests an overbought condition; RSI below
30 suggests oversold. RSI can stay extreme for long periods in strong trends, so it
works best in range-bound markets.

## MACD (Moving Average Convergence Divergence)

MACD line = EMA(12) - EMA(26)
Signal line = EMA(9) of the MACD line
Histogram = MACD line - Signal line

A MACD line crossing above the signal line is a bullish signal; crossing below is
bearish. Divergence between MACD and price (price makes a new high but MACD does not)
often precedes trend exhaustion.

## Bollinger Bands

Middle band = MA(20)
Upper band = MA(20) + 2 * standard deviation of the last 20 closes
Lower band = MA(20) - 2 * standard deviation of the last 20 closes

Bands widen when volatility rises and squeeze when volatility compresses. Prices
touching the upper band are relatively high; touching the lower band are relatively
low. Band-width expansion after a squeeze often marks the start of a directional move.

## Average True Range (ATR)

True Range = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
ATR = moving average of True Range over N days (commonly 14)

ATR measures volatility in absolute price units. It is widely used for position
sizing and stop-loss placement (e.g., stop = entry - 2 * ATR).

## Practical Notes

- Indicators are derived from past prices; they describe, not predict.
- Combining trend indicators (MA, MACD) with oscillators (RSI) reduces false signals.
- Always state the lookback window when reporting an indicator value.
