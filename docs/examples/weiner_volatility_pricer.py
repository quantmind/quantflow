from quantflow.options.inputs import OptionType
from quantflow.options.pricer import OptionPricer
from quantflow.sp.weiner import WeinerProcess
from quantflow.utils.distributions import DoubleExponential

# Weiner process with constant volatility
# This produces the same sensitivities as the Black-Scholes model
pricer = OptionPricer(
    model=WeinerProcess(sigma=0.3)
)

# Price an ATM call option at time to maturity 1.0
price = pricer.price(
    option_type=OptionType.call,
    strike=100.0,
    forward=100.0,
    ttm=1.0,
)
print(price.model_dump_json(indent=2))
