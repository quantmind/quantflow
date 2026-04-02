# Calendar Spread

A calendar spread (also known as a time spread or horizontal spread) is the same option
type and strike at two different maturities: long the far maturity, short the near maturity
(long calendar) or the reverse (short calendar).

## Structure

- quantity option at strike K, maturity T_far
- -quantity option at strike K, maturity T_near

A positive quantity is a long calendar spread. A negative quantity is a short calendar spread.

## Greeks

### Call calendar

Long far call, short near call. When in the money (K < F), the far call has less delta than
the near call (more time value means less sensitivity to the underlying) for the same value
of implied volatility. The correct cutoff point depends on the term structure of implied volatility.

In this case, net delta is negative when long. The opposite is true when out of the money
(K > F): the far call has more delta than the near call, so net delta is positive when long.
There is a crossover point where the net delta is zero, near the strike price, where the
delta of the far call equals the delta of the near call.

### Put calendar

Long far put, short near put. The sign mirrors the call calendar with moneyness reversed
(ITM for puts means K > F):

- ITM put (K > F): far put has more negative delta than near put. Net delta is negative when long.
- OTM put (K < F): far put has less negative delta than near put. Net delta is positive when long.
- ATM: net delta near zero, crossover where far and near deltas are equal.

### Summary

- Delta: near zero at ATM, sign depends on moneyness and option type (see above)
- Gamma: negative at the near expiry strike when long (short gamma near term)
- Vega: positive when long — the far leg has more vega than the near leg

## Use case

A long calendar spread profits from the near-term option decaying faster than the far-term
option (theta play), and from an increase in implied volatility of the far leg relative to
the near leg (forward vol play).
It is sensitive to the term structure of implied volatility rather than the level.
