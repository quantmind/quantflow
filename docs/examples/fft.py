from docs.examples._utils import assets_path
from quantflow.sp.wiener import WienerProcess
from quantflow.utils import plot

p = WienerProcess(sigma=0.5)
m = p.marginal(0.2)

fig = plot.plot_characteristic(m)
fig.update_layout(title="Wiener Process Characteristic Function")
fig.write_image(assets_path("wiener_characteristic.png"))

fig = plot.plot_marginal_pdf(m, n=128, use_fft=True, max_frequency=20)
fig.update_layout(title="Wiener Process PDF via FFT with n=128")
fig.write_image(assets_path("wiener_fft_128.png"))

fig = plot.plot_marginal_pdf(m, n=128 * 8, use_fft=True, max_frequency=8 * 20)
fig.update_layout(title="Wiener Process PDF via FFT with n=1024")
fig.write_image(assets_path("wiener_fft_1024.png"))

fig = plot.plot_marginal_pdf(m, 64)
fig.update_layout(title="Wiener Process PDF via FRFT with n=64")
fig.write_image(assets_path("wiener_64.png"))
