import json

from docs.examples._utils import assets_path, print_model
from quantflow.options.calibration import BNSCalibration
from quantflow.options.pricer import OptionPricer
from quantflow.options.surface import VolSurface, VolSurfaceInputs, surface_from_inputs
from quantflow.sp.bns import BNS

# Load a saved volatility surface snapshot and build the surface
with open("docs/examples/volsurface.json") as fp:
    surface: VolSurface = surface_from_inputs(VolSurfaceInputs(**json.load(fp)))

surface.bs()
surface.disable_outliers()

# Create a BNS pricer with initial parameters
pricer = OptionPricer(
    model=BNS.create(vol=0.5, kappa=1.0, decay=10.0, rho=-0.2),
)

calibration: BNSCalibration[BNS] = BNSCalibration(
    pricer=pricer,
    vol_surface=surface,
    moneyness_weight=0.5,
)

result = calibration.fit()
print(result.message)
print_model(calibration.model)

# Plot the calibrated smile for all maturities and save as PNG
fig = calibration.plot_maturities(max_moneyness=1.5, support=101)
fig.update_layout(title="BNS Calibrated Smiles")
fig.write_image(assets_path("bns_calibrated_smile.png"), width=1200)
