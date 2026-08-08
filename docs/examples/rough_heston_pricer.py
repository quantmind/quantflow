"""Rough Heston: price a short-maturity smile and compare with the Heston model.

The rough model (Hurst H < 1/2) produces a steeper short-maturity skew than the
classical Heston model with the same parameters, without adding jumps.
"""

from quantflow.options.inputs import OptionType
from quantflow.options.pricer import OptionPricer, OptionPricingMethod
from quantflow.sp.heston import Heston
from quantflow.sp.rough_heston import RoughHeston

TTM = 0.1

rough = OptionPricer(
    model=RoughHeston.create(vol=0.5, kappa=1.5, sigma=0.7, rho=-0.6, hurst=0.1),
    method=OptionPricingMethod.COS,
)
heston = OptionPricer(
    model=Heston.create(vol=0.5, kappa=1.5, sigma=0.7, rho=-0.6),
    method=OptionPricingMethod.COS,
)


def implied_vol(pricer: OptionPricer, strike: float) -> float:
    option_type = OptionType.PUT if strike < 100.0 else OptionType.CALL
    price = pricer.price(option_type=option_type, strike=strike, forward=100.0, ttm=TTM)
    return float(price.black.iv)


print(f"Short-maturity smile at ttm={TTM} (forward=100)")
print(f"{'strike':>8}{'rough IV':>12}{'heston IV':>12}")
for strike in (90.0, 95.0, 100.0, 105.0, 110.0):
    print(
        f"{strike:>8.1f}"
        f"{implied_vol(rough, strike):>12.4f}"
        f"{implied_vol(heston, strike):>12.4f}"
    )
