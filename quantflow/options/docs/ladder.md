# Ladder

A ladder combines three options of the same type and the same maturity: one bought at the
strike closest to the money and two sold further out of the money. It is sometimes called a
Christmas tree.

The position is net short one option, so a long ladder is short volatility and short gamma.

## Structure

All three legs share the same maturity and the same option type. Only the strikes differ.

A positive quantity is a long ladder. A negative quantity is a short ladder.

### Call ladder

Requires K_low < K_mid <= K_high.

- quantity call at K_low (bought)
- -quantity call at K_mid (sold)
- -quantity call at K_high (sold)

### Put ladder

Requires K_low <= K_mid < K_high.

- quantity put at K_high (bought)
- -quantity put at K_mid (sold)
- -quantity put at K_low (sold)

## Relation to the ratio spread

When the two sold strikes coincide (K_mid = K_high for a call ladder, K_low = K_mid for a put
ladder) the two short legs fall on the same strike and the structure becomes a ratio spread,
quoted as a 1 by 2: one option bought against two sold.

In that case the strategy holds two legs rather than three, the sold leg carrying twice the
quantity, so the position is represented the same way a hand built 1 by 2 would be.

The ladder is therefore the generalisation of the 1 by 2 in which the two short legs are spread
across different strikes, which softens the loss profile beyond the outer strike at the cost of
a lower premium received.

## Payoff

A long call ladder at expiry, ignoring the premium:

- below K_low: zero
- between K_low and K_mid: rises one for one with the underlying
- between K_mid and K_high: flat at its maximum of K_mid - K_low
- above K_high: falls one for one with the underlying, without limit

The upper breakeven sits at K_mid + K_high - K_low. Above that level the position loses without
limit, so a ladder is not a defined risk structure.

The put ladder mirrors this. Its maximum is K_high - K_mid, reached between K_low and K_mid, and
its lower breakeven sits at K_low + K_mid - K_high. When that level is at or below zero the
position stays profitable all the way down.

## Greeks

- Delta: slightly positive for a long call ladder while the underlying sits below K_low, turning
  negative as the underlying rallies through the sold strikes, and approaching -1 once all three
  options are deep in the money. The put ladder mirrors this.
- Gamma: positive near the bought strike, where its gamma peaks, and negative around and beyond
  the two sold strikes. The position is net short gamma overall.
- Vega: negative when long, since two options are sold against one bought. The exposure grows as
  the underlying approaches the sold strikes.
- Theta: positive when long. The ladder collects time decay, which is the counterpart of its
  short gamma.

## Use case

A long call ladder expresses a moderately bullish view with a ceiling: it pays most when the
underlying settles between K_mid and K_high, and it is often opened for a net credit because two
options are sold against one bought. The premium collected is compensation for uncapped risk
above the upper breakeven.

A long put ladder is the mirror image, expressing a moderately bearish view with a floor.

Selling two wing options makes the ladder attractive when the wings of the smile are rich
relative to the strike being bought, so it is sensitive to the slope and the level of the smile,
not only to the level of volatility.
