import json
from pathlib import Path

from quantflow.options.calibration import HestonCalibration
from quantflow.options.pricer import OptionPricer
from quantflow.options.surface import VolSurface, VolSurfaceInputs, surface_from_inputs
from quantflow.sp.heston import Heston

# Load a saved volatility surface snapshot and build the surface
with open("quantflow_tests/volsurface.json") as fp:
    surface: VolSurface = surface_from_inputs(VolSurfaceInputs(**json.load(fp)))

surface.bs()
surface.disable_outliers()

# Create a Heston pricer and calibrate
pricer = OptionPricer(model=Heston.create(vol=0.5, kappa=1, sigma=0.8, rho=0))
calibration = HestonCalibration(
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
    title="Heston Calibrated Smile — Maturity 2",
)

out_path = Path("docs/assets/heston_calibrated_smile.png")
fig.write_image(str(out_path), width=900, height=500)
print(f"saved {out_path}")
