Fetch the live implied volatility surface for a crypto or equity asset and return
the calibrated surface with discount curves and forward prices.

Data sources:

- **BTC, ETH**: live option chain from Deribit (inverse, crypto-quoted contracts).
- **SPY, AAPL, NVDA**: option chain from Yahoo Finance (standard equity contracts).

The surface is calibrated using Black-Scholes implied volatilities. Outlier options
(bid-ask spread too wide, or IV outside plausible bounds) are disabled before
the surface is returned.

The forward of each maturity is calibrated from put-call parity first, and
the surface prices options directly off those forwards. The `pcp_forwards`
field of the response contains the calibrated forward term structure.

Two discount curves are then derived from the calibrated forwards:

- **quote_curve**: discount curve for the numeraire (USD for equity, crypto for inverse).
- **asset_curve**: discount curve for the underlying asset.

The asset curve is always an interpolated curve fitted through the discount
factors implied by the calibrated forwards. The quote curve is fixed at no
discounting for crypto assets, whose inverse options settle without
discounting, and fitted as an interpolated curve for equity assets. The
curves provide discounting and rates only; the pricing forwards come from
parity regardless.

The forward curve and per-maturity implied forwards from put-call parity are also
included, which are useful for detecting curve arbitrage or funding dislocations.

Responses are cached; live data may be up to a few minutes old.
