# Spread (Vertical Spread)

A spread (formally a vertical spread) combines two options of the same type and expiry
at different strikes. The term "vertical" refers to the strikes being at different levels
on the same expiry column of an options chain.

## Call Spread

- Long call at K_low (lower strike)
- Short call at K_high (higher strike)

A long call spread profits from a rise in the underlying above K_low, with profit capped
at K_high. The short call at K_high finances the long call at K_low.

## Put Spread

- Long put at K_high (higher strike)
- Short put at K_low (lower strike)

A long put spread profits from a fall in the underlying below K_high, with profit capped
at K_low. The short put at K_low finances the long put at K_high.

## Quantity

A positive quantity is long the spread (debit spread). A negative quantity is short (credit spread).

## Greeks

- Delta: positive for a long call spread, negative for a long put spread
- Gamma: changes sign across the body of the spread
- Vega: low net vega, the two legs largely offset

## Use case

Spreads are directional trades with defined risk and reward, cheaper than outright options
because the sold leg partially finances the bought leg.
