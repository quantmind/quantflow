import json

from quantflow.options.heston_calibration import HestonCalibration
from quantflow.options.pricer import OptionPricer
from quantflow.options.surface import VolSurface, VolSurfaceInputs, surface_from_inputs
from quantflow.sp.heston import Heston

# Load a saved volatility surface snapshot and build the surface
with open("docs/examples/volsurface.json") as fp:
    surface: VolSurface = surface_from_inputs(VolSurfaceInputs(**json.load(fp)))

surface.bs()
surface.disable_outliers()

# Create a Heston pricer with initial parameters
pricer = OptionPricer(model=Heston.create(vol=0.5, kappa=1, sigma=0.8, rho=0))

# Set up the calibration, dropping the first (very short) maturity and high-vol wings
calibration = HestonCalibration(
    pricer=pricer,
    vol_surface=surface.trim(len(surface.maturities) - 1),
    moneyness_weight=1.0,
).remove_implied_above(quantile=0.95)

result = calibration.fit()
print(result.message)
params = calibration.get_params()
model = calibration.model
print(f"vol:   {model.variance_process.rate**0.5:.4f}")
print(f"theta: {model.variance_process.theta**0.5:.4f}")
print(f"kappa: {model.variance_process.kappa:.4f}")
print(f"sigma: {model.variance_process.sigma:.4f}")
print(f"rho:   {model.rho:.4f}")
