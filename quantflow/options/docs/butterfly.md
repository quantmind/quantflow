# Butterfly

A butterfly consists of three strikes: a lower wing, a body, and an upper wing with equal
log-spacing. It is constructed by buying the wings and selling twice the body (long butterfly)
or the reverse (short butterfly).

## Structure

- quantity option at K_low (lower wing)
- -2 * quantity option at K_mid (body)
- quantity option at K_high (upper wing)

The three strikes are symmetric in log space: log(K_mid/K_low) = log(K_high/K_mid).

A positive quantity is a long butterfly. A negative quantity is a short butterfly.

## Call vs Put construction

By put-call parity, a butterfly built entirely with calls is equivalent in price to one built
entirely with puts. The choice is purely a liquidity consideration:

- Body above ATM (moneyness > 0): use calls, which are more liquid OTM on the upside
- Body below ATM (moneyness < 0): use puts, which are more liquid OTM on the downside
- Body at ATM (moneyness = 0): either works

## Greeks

- Delta: near zero for log-symmetric strikes around ATM
- Gamma: small and negative when long, small and positive when short. The gamma of the wings
  and body largely cancel out, leaving low net exposure.
- Vega: small and negative when long, small and positive when short. The vega of the three
  legs nearly offsets, so the butterfly has limited sensitivity to parallel shifts in implied
  volatility.

The low vega and gamma distinguish the butterfly from outright vol strategies such as straddles
and strangles. The butterfly is primarily sensitive to the curvature of the vol smile across
strikes, not to the overall level of volatility.

## Use case

A long butterfly profits when the underlying stays close to the body strike at expiry.
It is a relative value trade on the shape of the vol smile: it is cheap when the smile is
steep (wings are expensive relative to the body) and expensive when the smile is flat.
A short butterfly profits from large moves away from the body and from smile flattening.
