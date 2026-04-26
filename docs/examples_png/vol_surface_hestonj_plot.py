import json
from pathlib import Path

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

# Create a HestonJ pricer and calibrate
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
calibration = HestonJCalibration(
    pricer=pricer,
    vol_surface=surface.trim(len(surface.maturities) - 1),
    moneyness_weight=1.0,
).remove_implied_above(quantile=0.95)
calibration.fit()

# Plot the calibrated smile for the second maturity and save as PNG
fig = calibration.plot(index=1, max_moneyness_ttm=1.5, support=101)
fig.update_layout(
    xaxis_title="Moneyness / sqrt(T)",
    yaxis_title="Implied Volatility",
    title="HestonJ Calibrated Smile — Maturity 2",
)

out_path = Path("docs/assets/hestonj_calibrated_smile.png")
fig.write_image(str(out_path), width=900, height=500)
print(f"saved {out_path}")
