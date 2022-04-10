# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Option Pricing
#
# We ca use the tooling for characteristic function inversion to price european call options on an underlying $S_t = S_0 e^{x_t}$. A call option is defined as
#
# \begin{align}
# F &= S_0{\mathbb E}_0\left[e^{x_t}\right] \\
# C &= F c_k \\
# k &= \ln\frac{K}{F}\\ 
# c_k &= {\mathbb E}\left[\left(e^x_t - e^k\right)1_{x_t\ge k}\right]
# \end{align}
#
# We follow {cite:p}`carr_madan` and write the Fourier transform of the the call option as
#
# \begin{equation}
# \Psi_u = \int_{-\infty}^\infty e^{i u k} c_k dk
# \end{equation}
#
# Note that $c_k$ tends to $e^x_t$ as $k \to -\infty$, therefore the call price function is not square-integrable. In order to obtain integrability, we choose complex values of $u$ of the form
# \begin{equation}
# u = v - i \alpha
# \end{equation}
# The value of $\alpha$ is a numerical choice we can check later.
#
# It is possible to obtain the analytical expression of $\Psi_k$ in terms of the characteristic function $\Phi_x$. Once we have that expression we can use the Fourier transform tooling presented peviously to calculate option prices.
#
# \begin{align}
# c_k &= \int_0^{\infty} e^{-iuk} \Psi\left(u\right) du \\
#     &= \frac{e^{-\alpha k}}{\pi} \int_0^{\infty} e^{-ivk} \Psi\left(v-i\alpha\right) dv \\
# \end{align}
#
# The analytical expression of $\phi_k$ is given by
#
# \begin{equation}
# \Psi\left(u\right) = \frac{\Phi\left(u-i\right)}{iu \left(iu + 1\right)}
# \end{equation}

# %%
from quantflow.sp.weiner import Weiner
p = Weiner(0.5)
m = p.marginal(1)
m.std()

# %%
import plotly.express as px
N = 64
M = 8
dx = 4/N
alpha = 0.5
r = m.call_option(N, M, dx, alpha=alpha)
fig = px.line(r, x="x", y="y", markers=True)
fig.show()
