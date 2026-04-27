import json
from pathlib import Path

from docs.examples._utils import print_model
from quantflow.options.heston_calibration import HestonJCalibration
from quantflow.options.pricer import OptionPricer
from quantflow.options.surface import VolSurface, VolSurfaceInputs, surface_from_inputs
from quantflow.sp.heston import HestonJ
from quantflow.utils.distributions import DoubleExponential

# Load a saved volatility surface snapshot and build the surface
with open("docs/examples/volsurface.json") as fp:
    surface: VolSurface = surface_from_inputs(VolSurfaceInputs(**json.load(fp)))

surface.bs()
surface.disable_outliers()

# Create a HestonJ pricer with initial parameters
pricer = OptionPricer(
    model=HestonJ.create(
        DoubleExponential,
        vol=0.5,
        kappa=2,
        rho=-0.2,
        sigma=0.8,
        jump_fraction=0.3,
        jump_asymmetry=0.2,
    )
)

# Set up the calibration, dropping the first (very short) maturity
calibration: HestonJCalibration[DoubleExponential] = HestonJCalibration(
    pricer=pricer,
    vol_surface=surface,
    moneyness_weight=0.5,
)

result = calibration.fit()
print(result.message)
print_model(calibration.model)

# Plot the calibrated smile for all maturities and save as PNG
fig = calibration.plot_maturities(max_moneyness_ttm=1.5, support=101)
fig.update_layout(title="HestonJ Calibrated Smiles")

out_path = Path("docs/assets/hestonj_calibrated_smile.png")
fig.write_image(str(out_path), width=1200)
