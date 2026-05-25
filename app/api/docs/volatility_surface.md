Fetch the live implied volatility surface for a crypto or equity asset and return
the calibrated surface with discount curves and forward prices.

Data sources:

- **BTC, ETH**: live option chain from Deribit (inverse, crypto-quoted contracts).
- **SPY, AAPL, NVDA**: option chain from Yahoo Finance (standard equity contracts).

The surface is calibrated using Black-Scholes implied volatilities. Outlier options
(bid-ask spread too wide, or IV outside plausible bounds) are disabled before
the surface is returned.

Two discount curves are calibrated from the option data:

- **quote_curve**: discount curve for the numeraire (USD for equity, crypto for inverse).
- **asset_curve**: discount curve for the underlying asset.

The forward curve and per-maturity implied forwards from put-call parity are also
included, which are useful for detecting curve arbitrage or funding dislocations.

Responses are cached; live data may be up to a few minutes old.
