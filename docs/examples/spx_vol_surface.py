import gzip
import json

from docs.examples._utils import FIXTURES, assets_path, print_model
from quantflow.data.yahoo import Yahoo
from quantflow.options.calibration import BNS2Calibration
from quantflow.options.calibration.base import ResidualKind
from quantflow.options.pricer import OptionPricer, OptionPricingMethod
from quantflow.sp.bns import BNS, BNS2

chain = json.loads(gzip.decompress((FIXTURES / "yahoo_spx.json.gz").read_bytes()))
loader = Yahoo.loader_from_chain(chain, exclude_volume=1)
surface = loader.surface()
surface.bs()
surface.disable_outliers()

fig = surface.plot3d()
fig.update_traces(marker=dict(size=3))
fig.update_layout(
    title="SPX implied volatility surface",
    scene=dict(
        xaxis_title="moneyness",
        yaxis_title="time to maturity (log)",
        zaxis_title="implied volatility",
        yaxis=dict(type="log"),
        camera=dict(eye=dict(x=0.6, y=-2.2, z=0.8)),
    ),
)
fig.write_image(assets_path("spx_vol_surface.png"), width=1200, height=800)

# Calibrate a two-factor BNS model to the SPX surface. A fast factor absorbs
# the steep short-dated equity skew; a slow factor anchors the long end.
pricer = OptionPricer(
    model=BNS2(
        bns1=BNS.create(vol=0.2, kappa=20.0, decay=10.0, rho=-0.6),
        bns2=BNS.create(vol=0.2, kappa=0.3, decay=10.0, rho=-0.3),
        weight=0.5,
    ),
    method=OptionPricingMethod.COS,
)
calibration: BNS2Calibration[BNS2] = BNS2Calibration(
    pricer=pricer,
    vol_surface=surface,
    residual_kind=ResidualKind.IV,
)
result = calibration.fit()
print(result.message)
print_model(calibration.model)

smile = calibration.plot_maturities(max_moneyness=0.5, support=101)
smile.update_layout(title="SPX BNS2 Calibrated Smiles")
smile.write_image(assets_path("spx_vol_surface_bns2.png"), width=1200)
