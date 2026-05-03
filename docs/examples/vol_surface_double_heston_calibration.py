import json

from docs.examples._utils import assets_path, print_model
from quantflow.options.heston_calibration import DoubleHestonCalibration
from quantflow.options.pricer import OptionPricer
from quantflow.options.surface import VolSurface, VolSurfaceInputs, surface_from_inputs
from quantflow.sp.heston import DoubleHeston, Heston

# Load a saved volatility surface snapshot and build the surface
with open("docs/examples/volsurface.json") as fp:
    surface: VolSurface = surface_from_inputs(VolSurfaceInputs(**json.load(fp)))

surface.bs()
surface.disable_outliers()

# Build a DoubleHeston model: heston1 is the short-maturity process (higher kappa),
# heston2 is the long-maturity process
model = DoubleHeston(
    heston1=Heston.create(vol=0.5, kappa=4, sigma=1.0, rho=-0.3),
    heston2=Heston.create(vol=0.4, kappa=1, sigma=0.6, rho=-0.2),
)

pricer = OptionPricer(model=model, n=256)

calibration: DoubleHestonCalibration[DoubleHeston] = DoubleHestonCalibration(
    pricer=pricer,
    vol_surface=surface,
)

result = calibration.fit()
print(result.message)
print_model(calibration.model)

# Plot the calibrated smile for all maturities and save as PNG
fig = calibration.plot_maturities(max_moneyness=1.5, support=101)
fig.update_layout(title="Double Heston Calibrated Smiles")
fig.write_image(assets_path("double_heston_calibrated_smile.png"), width=1200)
