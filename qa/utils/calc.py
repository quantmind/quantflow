import math


def tolerant_equals(a, b, atol=10e-7, rtol=10e-7):
    return math.fabs(a - b) <= (atol + rtol * math.fabs(b))


# On an order to buy, between .05 below to .95 above a penny, use that penny.
# On an order to sell, between .05 above to .95 below a penny, use that penny.
# buy: [.0095, .0195) -> round to .01, sell: (.0005, .0105] -> round to .01
def round_for_minimum_price_variation(x, is_buy, diff=(0.0095 - .005)):
    # relies on rounding half away from zero, unlike numpy's bankers' rounding
    rounded = round(x - (diff if is_buy else -diff), 2)
    return 0.0 if tolerant_equals(rounded, 0.0) else rounded
