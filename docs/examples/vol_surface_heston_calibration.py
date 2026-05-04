import json

from docs.examples._utils import assets_path, print_model
from quantflow.options.calibration import HestonCalibration
from quantflow.options.pricer import OptionPricer, OptionPricingMethod
from quantflow.options.surface import VolSurfaceInputs, surface_from_inputs
from quantflow.sp.heston import Heston

# Load a saved volatility surface snapshot and build the surface
with open("docs/examples/volsurface.json") as fp:
    surface = surface_from_inputs(VolSurfaceInputs(**json.load(fp)))

surface.bs()
surface.disable_outliers()

# Create a Heston pricer with initial parameters
pricer = OptionPricer(
    model=Heston.create(vol=0.5, kappa=1, sigma=0.8, rho=0),
    method=OptionPricingMethod.COS,
)

# Set up the calibration, dropping the first (very short) maturity
calibration: HestonCalibration[Heston] = HestonCalibration(
    pricer=pricer,
    vol_surface=surface,
)

result = calibration.fit()
print(result.message)
print_model(calibration.model)

# Plot the calibrated smile for all maturities and save as PNG
fig = calibration.plot_maturities(max_moneyness=1.5, support=101)
fig.update_layout(title="Heston Calibrated Smiles")
fig.write_image(assets_path("heston_calibrated_smile.png"), width=1200)
