import json

from docs.examples._utils import assets_path, print_model
from quantflow.options.calibration import BNS2Calibration
from quantflow.options.pricer import OptionPricer
from quantflow.options.surface import VolSurface, VolSurfaceInputs, surface_from_inputs
from quantflow.sp.bns import BNS, BNS2

# Load a saved volatility surface snapshot and build the surface
with open("docs/examples/volsurface.json") as fp:
    surface: VolSurface = surface_from_inputs(VolSurfaceInputs(**json.load(fp)))

surface.bs()
surface.disable_outliers()

# Two-factor BNS: a fast factor for short maturities and a slow one for long.
# Opposite-sign leverages lets one factor lift the OTM-call wing (rho>0) while
# the other carries the equity-style downside skew (rho<0).
pricer = OptionPricer(
    model=BNS2(
        bns1=BNS.create(vol=0.4, kappa=20.0, decay=20.0, rho=-0.6),
        bns2=BNS.create(vol=0.5, kappa=0.3, decay=5.0, rho=0.3),
        weight=0.3,
    )
)

calibration: BNS2Calibration[BNS2] = BNS2Calibration(
    pricer=pricer,
    vol_surface=surface,
    moneyness_weight=0.5,
)

result = calibration.fit()
print(result.message)
print_model(calibration.model)

fig = calibration.plot_maturities(max_moneyness=1.5, support=101)
fig.update_layout(title="BNS2 Calibrated Smiles")
fig.write_image(assets_path("bns2_calibrated_smile.png"), width=1200)
