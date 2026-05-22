import gzip
import json

from docs.examples._utils import FIXTURES, assets_path
from quantflow.data.yahoo import Yahoo

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
