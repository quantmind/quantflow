# Butterfly

A butterfly consists of three strikes: a lower wing, a body, and an upper wing.
It is constructed by buying the wings and selling twice the body (long butterfly)
or the reverse (short butterfly).

## Structure

All three legs share the same maturity and the same option type. Only the strikes differ.

- quantity option at K_low (lower wing)
- -2 * quantity option at K_mid (body)
- quantity option at K_high (upper wing)

A positive quantity is a long butterfly. A negative quantity is a short butterfly.

## Wing symmetry

The strikes are usually equally spaced in price, so that the body sits midway between
the wings:

    K_mid - K_low = K_high - K_mid

This is the conventional construction, but it is not enforced. Any three increasing strikes
are accepted, which allows unequal wing widths (a structure sometimes called a broken wing
or skip strike butterfly).

The difference between the two wing widths drives the behaviour of the unbalanced case:

    residual = (K_mid - K_low) - (K_high - K_mid)

This is zero when the wings are equally spaced, and two properties of the butterfly hold
only in that case.

The first is that the payoff closes. Outside the wings a balanced butterfly expires worthless,
giving the familiar tent shaped payoff peaking at K_mid. With unequal wings a constant value
equal to the residual remains beyond the outer strikes: a loss when the upper wing is wider,
a gain when the lower wing is wider.

The second is that the call and put constructions coincide (see below).

## Call vs Put construction

By put-call parity, a butterfly built entirely with calls is equivalent in price to one built
entirely with puts, provided the wings are equally spaced. The price difference between the two
constructions is exactly the residual defined above, so it vanishes only in the balanced case.

For a balanced butterfly the choice is therefore purely a liquidity consideration:

- Body above ATM (moneyness > 0): use calls, which are more liquid OTM on the upside
- Body below ATM (moneyness < 0): use puts, which are more liquid OTM on the downside
- Body at ATM (moneyness = 0): either works

With unequal wings the two constructions no longer price the same, so the option type becomes
a pricing decision rather than a liquidity one and should be chosen explicitly.

## Greeks

The Greeks below describe the balanced case.

- Delta: near zero when the wings are equally spaced and the body sits close to the forward
- Gamma: small and negative when long, small and positive when short. The gamma of the wings
  and body largely cancel out, leaving low net exposure.
- Vega: small and negative when long, small and positive when short. The vega of the three
  legs nearly offsets, so the butterfly has limited sensitivity to parallel shifts in implied
  volatility.

The low vega and gamma distinguish the butterfly from outright vol strategies such as straddles
and strangles. The butterfly is primarily sensitive to the curvature of the vol smile across
strikes, not to the overall level of volatility.

### Gamma near expiry

The cancellation above holds only while the wings sit close to the body relative to the size of
a typical move, that is, while the wing width is small compared with sigma * sqrt(ttm).

When the body sits at the forward and expiry approaches, the wings move far away in those units
and their gamma decays to zero, while the body gamma rises. What remains is the body alone.

A long butterfly then carries the gamma of a short straddle struck at the body, and a short
butterfly the gamma of a long straddle. The near cancellation of the longer dated case is gone.

The counterpart is theta: the same position earns time decay for as long as the underlying stays
pinned to the body.

Away from the body the sign reverses. Close to expiry the position is long gamma near each wing,
where the wing option is itself at the money.

## Use case

A long butterfly profits when the underlying stays close to the body strike at expiry.
It is a relative value trade on the shape of the vol smile: it is cheap when the smile is
steep (wings are expensive relative to the body) and expensive when the smile is flat.
A short butterfly profits from large moves away from the body and from smile flattening.
