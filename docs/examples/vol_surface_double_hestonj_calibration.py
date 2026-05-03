import json

from docs.examples._utils import assets_path, print_model
from quantflow.options.heston_calibration import DoubleHestonJCalibration
from quantflow.options.pricer import OptionPricer
from quantflow.options.surface import VolSurface, VolSurfaceInputs, surface_from_inputs
from quantflow.sp.heston import DoubleHestonJ, Heston, HestonJ
from quantflow.utils.distributions import DoubleExponential

# Load a saved volatility surface snapshot and build the surface
with open("docs/examples/volsurface.json") as fp:
    surface: VolSurface = surface_from_inputs(VolSurfaceInputs(**json.load(fp)))

surface.bs()
surface.disable_outliers()

# Build a DoubleHestonJ model: heston1 is the short-maturity process with jumps,
# heston2 is the long-maturity diffusion process
model = DoubleHestonJ(
    heston1=HestonJ.create(
        DoubleExponential,
        vol=0.5,
        kappa=4,
        sigma=1.0,
        rho=-0.3,
        jump_fraction=0.3,
        jump_asymmetry=0.2,
    ),
    heston2=Heston.create(vol=0.4, kappa=1, sigma=0.6, rho=-0.2),
)

pricer = OptionPricer(model=model, n=256)

calibration: DoubleHestonJCalibration[DoubleExponential] = DoubleHestonJCalibration(
    pricer=pricer,
    vol_surface=surface,
    moneyness_weight=0.5,
)

result = calibration.fit()
print(result.message)
print_model(calibration.model)

# Plot the calibrated smile for all maturities and save as PNG
fig = calibration.plot_maturities(max_moneyness=1.5, support=101)
fig.update_layout(title="Double HestonJ Calibrated Smiles")
fig.write_image(assets_path("double_hestonj_calibrated_smile.png"), width=1200)
