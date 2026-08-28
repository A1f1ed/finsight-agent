# Risk and Return Metrics Reference

Definitions and computation conventions for common portfolio risk metrics.
Conventions below assume daily returns r_t = P_t / P_(t-1) - 1 and 252 trading
days per year unless stated otherwise.

## Annualized Return

Annualized return converts a period return to a yearly rate:

Annualized return = (1 + total return) ^ (252 / number of trading days) - 1

## Annualized Volatility

Volatility is the standard deviation of returns, scaled to a year:

Annualized volatility = std(daily returns) * sqrt(252)

Higher volatility means wider price swings. Equity volatility commonly ranges from
15% (large stable companies) to 50%+ (small or speculative names).

## Sharpe Ratio

Risk-adjusted return relative to a risk-free rate rf:

Sharpe = (annualized return - rf) / annualized volatility

Interpretation: below 0 is worse than the risk-free asset; 0.5-1.0 is moderate;
above 1.0 is good; above 2.0 is excellent. For short sample periods the estimate
is noisy and should be reported with the sample length.

## Sortino Ratio

Like Sharpe but penalizes only downside volatility:

Sortino = (annualized return - rf) / (std of negative daily returns * sqrt(252))

Useful when returns are skewed and upside volatility should not be punished.

## Maximum Drawdown (MDD)

The largest peak-to-trough decline over the sample:

MDD = max over t of (peak value up to t - value at t) / peak value up to t

Reported as a positive percentage. MDD describes worst-case loss experience and is
often more intuitive for investors than volatility.

## Value at Risk (VaR)

VaR at confidence level c (e.g., 95%) is the loss threshold not exceeded with
probability c over a given horizon. Historical simulation takes the (1-c) quantile
of the return distribution:

Historical VaR(95%) = 5th percentile of daily returns

Parametric VaR assumes normality: VaR = mean - 1.645 * std (for 95%).
VaR ignores the shape of the tail beyond the threshold; Expected Shortfall (CVaR)
averages losses beyond VaR and is more conservative.

## Beta

Sensitivity of an asset's returns to the market's returns:

Beta = Cov(asset return, market return) / Var(market return)

Beta > 1 amplifies market moves; beta < 1 dampens them; negative beta moves against
the market. Beta from a short window (under 120 days) is unstable.

## Practical Notes

- Always report the sample window together with any risk metric.
- Annualization assumes returns are independent; clustered volatility biases
  annualized volatility downward slightly.
- Risk-free rate for HKD is commonly proxied by HK Treasury bills or the USD
  risk-free rate when the reporting currency is USD.
