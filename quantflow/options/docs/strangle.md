# Strangle

A strangle is a call and put at different strikes, both OTM, same expiry, with the same signed quantity.

## Structure

- quantity put at strike K_low (below forward)
- quantity call at strike K_high (above forward)

A positive quantity is a long strangle (long vol). A negative quantity is a short strangle (short vol).

## Greeks

- Delta: near zero for log-symmetric strikes
- Gamma: positive when long, negative when short
- Vega: positive when long, negative when short; lower magnitude than a straddle for the same notional

## Use case

A long strangle is cheaper than a straddle but requires a larger move to profit.
It profits when realized volatility significantly exceeds implied volatility.
A short strangle profits from low realized volatility with a wider breakeven range
than a short straddle.
