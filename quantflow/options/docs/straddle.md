# Straddle

A straddle is a call and put at the same strike and expiry, with the same signed quantity.

## Structure

- quantity call at strike K
- quantity put at strike K

A positive quantity is a long straddle (long vol). A negative quantity is a short straddle (short vol).

## Greeks

- Delta: near zero at inception (ATM)
- Gamma: positive when long, negative when short
- Vega: positive when long, negative when short

## Use case

A long straddle profits when realized volatility exceeds implied volatility,
regardless of direction. A short straddle profits when realized volatility
is below implied volatility.
