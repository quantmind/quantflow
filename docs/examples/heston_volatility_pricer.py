from quantflow.options.inputs import OptionType
from quantflow.options.pricer import OptionPricer
from quantflow.sp.heston import HestonJ
from quantflow.utils.distributions import DoubleExponential

pricer = OptionPricer(
    model=HestonJ.create(
        DoubleExponential,
        vol=0.5,
        kappa=2,
        rho=-0.2,
        sigma=0.8,
        jump_fraction=0.5,
        jump_asymmetry=0.2,
    )
)

# Price an ATM call option at time to maturity 1.0
price = pricer.price(
    option_type=OptionType.call,
    strike=100.0,
    forward=100.0,
    ttm=1.0,
)
print(price.model_dump_json(indent=2))
