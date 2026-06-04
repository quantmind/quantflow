# Kalman Filter

The Kalman filter is a recursive algorithm that estimates the latent state
of a state-space model from noisy observations. Introduced in
[Kalman (1960)](../bibliography.md#kalman), it is optimal for
linear-Gaussian systems. The
[unscented Kalman filter](#unscented-kalman-filter), developed by
[Julier & Uhlmann (1997)](../bibliography.md#julier_uhlmann), extends the
same framework to non-linear transition functions while keeping the Gaussian
noise assumption.

State-space models are a powerful tool in quantitative finance. They
separate a time series into an unobserved *latent state* that captures the
underlying dynamics and an *observation* layer that models how those
dynamics manifest in market data. This decomposition is natural for many
problems: the latent state might represent an unobserved interest-rate
factor, a stochastic volatility process, or an economic regime, while the
observations are bond yields, option prices, or macroeconomic indicators.
The Kalman filter solves the inverse problem of reconstructing the latent
state from the observed data, a task known as *filtering*.

For a comprehensive treatment of state-space methods in time series, see
[Durbin & Koopman (2012)](../bibliography.md#durbin_koopman).

## State-Space Model

A state-space model describes the joint evolution of two stochastic
processes: the **latent state** $x_t$ and the **observation** $y_t$.
The latent state is the core of the model: it is the unobserved quantity
that we would like to estimate, and it evolves according to its own
dynamics. The observation is what we actually measure, and it depends on
the current state, typically with some added noise. This two-layer
structure lets us model complex, unobserved dynamics while maintaining a
tractable link to the data.

In full generality the model makes no distributional assumptions; it simply
factorises the joint law as

\begin{equation}
    \begin{aligned}
        x_t &\sim p(x_t \mid x_{t-1}) \\
        y_t &\sim p(y_t \mid x_t)
    \end{aligned}
\end{equation}

The **transition density** $p(x_t \mid x_{t-1})$ encodes how the latent
state evolves from one time step to the next. It is Markovian by
construction: the next state depends only on the current state, not on the
full history. The **observation density** $p(y_t \mid x_t)$ defines how the
observed data relates to the latent state. No particular parametric form
for either density is required at this level of generality; the
framework accommodates non-Gaussian, non-linear, and even discrete-state
models.

### State-Space Equations

The two-equation structure above is the defining feature of state-space
models. In the probabilistic form used by quantflow, the first is the
**state equation** (or transition equation): it describes how the latent
state evolves. This equation defines a Markov chain: the next state
depends only on the current state, not on any earlier history. The second
is the **observation equation** (or output equation): it describes how the
state manifests in the data. Together the two equations form a
[hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model):
a Markov chain $(x_t)$ that is not directly observed, inferred from
noisy observations $(y_t)$ whose distribution depends on the current
state. This probabilistic form is fully general: it makes no assumption
about the functional form of either equation, nor about the distribution
of any noise terms.

The same two equations are often written in continuous-time form:

\begin{equation}
    \begin{aligned}
        \dot x_t &= A_t\,x_t + B_t\,u_t \\
        y_t      &= C_t\,x_t + D_t\,u_t
    \end{aligned}
\end{equation}

where $u_t$ is an observed exogenous input and $A_t, B_t, C_t, D_t$ are
matrices, generally time-dependent. The input $u_t$ drives the state
dynamics and can also feed directly into the observation. In the
probabilistic formulation it enters as a conditioning variable:
$p(x_t \mid x_{t-1}, u_t)$ and $p(y_t \mid x_t, u_t)$. Quantflow's
models have no exogenous input (for example, in Vasicek calibration the
short rate evolves autonomously and yields depend only on the state), so
$B_t = 0$ and $D_t = 0$, recovering the input-free dynamics above.

### Applications

In finance, common examples include:

* **Interest-rate term structure**: the state is a vector of latent
  factors (level, slope, curvature) driving the yield curve; the
  observations are bond yields at various maturities, measured with
  auction noise.
* **Stochastic volatility**: the state is the unobserved variance
  process; the observations are asset returns, whose variance is
  modulated by the state.
* **Regime-switching**: the state is a discrete variable capturing
  different market regimes (bull, bear, crisis); the observations are
  returns whose distribution depends on the current regime.

### The Filtering Problem

Given a sequence of observations $y_1, \dots, y_T$, the goal is to compute
the **filtering distribution**

\begin{equation}
    p(x_t \mid y_{1:t})
\end{equation}

at each time step. This is the posterior over the latent state given all
observations up to time $t$. It answers the question: given everything we
have observed so far, what is the best estimate of the current latent
state, and how uncertain are we about it?

In the Gaussian setting below this distribution is itself Gaussian,
$\mathcal{N}(\hat x_t, \hat P_t)$, and both filters maintain it via
alternating predict and update steps. The *predict* step propagates the
state distribution through the transition dynamics; the *update* step
incorporates a new observation by Bayes' rule. This recursive structure
makes the filter computationally efficient: each new observation is
processed in constant time, without revisiting earlier data.

## Gaussian State-Space Model

The Kalman filter and the unscented Kalman filter both assume **additive
Gaussian noise**. This assumption is realistic whenever observations are
averages or aggregates of many independent random components, as in bond
yields, futures prices, and many macroeconomic indicators. In this
setting the model takes the form

\begin{equation}
\begin{aligned}
    x_t &= f(x_{t-1}, \Delta t) + \varepsilon_t,
        \quad \varepsilon_t \sim \mathcal{N}(0, Q_t) \\
    y_t &= H_t\, x_t + d_t + \eta_t,
        \quad \eta_t \sim \mathcal{N}(0, R_t)
\end{aligned}
\end{equation}

where $\mathcal{N}(\mu, \Sigma)$ denotes the multivariate normal
distribution with mean $\mu$ and covariance $\Sigma$.

The transition function $f$ is non-linear in general. When it is
linear, $f(x, \Delta t) = F_t x + c_t$, the model is
**linear-Gaussian** and the Kalman filter is exact. The observation
equation is always linear in $x_t$.

The components are:

* $x_t \in \mathbb{R}^{n_x}$ — latent state vector; $n_x$ is the
  state dimension.
* $y_t \in \mathbb{R}^{n_y}$ — observation vector; $n_y$ is the
  number of observed series (e.g. yields at different tenors).
* $f(\cdot, \Delta t) : \mathbb{R}^{n_x} \to \mathbb{R}^{n_x}$ —
  transition function. In the linear case $f(x, \Delta t) = F_t x + c_t$
  with $F_t$ of shape $(n_x, n_x)$ and $c_t \in \mathbb{R}^{n_x}$.
* $Q_t$ of shape $(n_x, n_x)$ — process noise covariance.
* $H_t$ of shape $(n_y, n_x)$ — observation matrix, mapping the latent
  state to the observation space.
* $R_t$ of shape $(n_y, n_y)$ — observation noise covariance.
* $d_t \in \mathbb{R}^{n_y}$ — observation intercept.

The initial state follows $x_0 \sim \mathcal{N}(\mu_0, \Sigma_0)$
with $\mu_0 \in \mathbb{R}^{n_x}$ and $\Sigma_0$ of shape
$(n_x, n_x)$.

## Kalman Filter

The Kalman filter assumes a linear transition $f(x, \Delta t) = F_t x + c_t$.

### Predict Step

The state distribution is propagated through the transition dynamics using
the prior estimate $\hat x_{t-1}, \hat P_{t-1}$:

\begin{equation}
\begin{aligned}
    \hat x_{t \mid t-1} &= F_t\,\hat x_{t-1} + c_t \\
    \hat P_{t \mid t-1} &= F_t\,\hat P_{t-1}\,F_t^\top + Q_t
\end{aligned}
\end{equation}

### Update Step

The prediction is corrected using the new observation $y_t$. Define the
**innovation** (observation residual) and its covariance:

\begin{equation}
\begin{aligned}
    e_t &= y_t - (H_t\,\hat x_{t \mid t-1} + d_t) \\
    S_t &= H_t\,\hat P_{t \mid t-1}\,H_t^\top + R_t
\end{aligned}
\end{equation}

The **Kalman gain** $K_t = \hat P_{t \mid t-1}\,H_t^\top S_t^{-1}$ weights
the innovation against the prediction uncertainty:

\begin{equation}
\begin{aligned}
    \hat x_t &= \hat x_{t \mid t-1} + K_t e_t \\
    \hat P_t &= \hat P_{t \mid t-1} - K_t H_t \hat P_{t \mid t-1}
\end{aligned}
\end{equation}

The log-likelihood contribution of observation $t$ is

\begin{equation}
    \ell_t = -\frac{1}{2}\left(
        n_y \log 2\pi + \log\det S_t + e_t^\top S_t^{-1} e_t
    \right)
\end{equation}

where $n_y$ is the observation dimension, and the total log-likelihood
$\mathcal{L} = \sum_{t=1}^T \ell_t$ is used for parameter estimation via
maximum likelihood.

### Sherman-Morrison Optimisation

When the observation noise is isotropic ($R_t = h^2 I$) and the observation
matrix $H_t$ is a column vector $c$ (which occurs when $n_x = 1$), the
innovation covariance $S_t = h^2 I + P\,c c^\top$ is a rank-1 update to a
scaled identity. The
[Sherman-Morrison identity](../glossary.md#sherman-morrison-identity)
inverts it in $O(n_y)$ instead of $O(n_y^3)$:

\begin{equation}
    S_t^{-1} = \frac{1}{h^2}I -
        \frac{P\,c c^\top}{h^2\left(h^2 + P\,c^\top c\right)}
\end{equation}

The module detects this case automatically and applies the fast path
without user intervention.

## Unscented Kalman Filter

When the transition $f(x, \Delta t)$ is non-linear the Kalman predict step
is no longer exact. The unscented Kalman filter (UKF), introduced by
[Julier & Uhlmann (1997)](../bibliography.md#julier_uhlmann), replaces it
with a **sigma-point** propagation. The observation update follows the same
structure, but builds the innovation covariance $S_t$ and the cross covariance
$C_t$ from the propagated sigma points and solves the gain $K_t = C_t S_t^{-1}$
with a dense Cholesky factorisation. The Sherman-Morrison fast path is specific
to the exact linear filter and is not used here.

### Sigma-Point Predict

Instead of pushing the mean and covariance through $F_t$, the UKF:

1. Generates $2n_x + 1$ *sigma points* around $\hat x_{t-1}$ using the
   Cholesky factor of $(n_x + \lambda)\hat P_{t-1}$.
2. Propagates each sigma point through $f(\cdot, \Delta t)$.
3. Re-estimates the mean and covariance from the propagated points using
   pre-computed weights.

The parameters $\alpha$, $\beta$, and $\kappa$ control the spread of the
sigma points via $\lambda = \alpha^2 (n_x + \kappa) - n_x$.

The UKF maintains a Gaussian approximation:

\begin{equation}
    p(x_t \mid y_{1:t-1}) \approx \mathcal{N}(\hat x_{t \mid t-1}, \hat P_{t \mid t-1})
\end{equation}

After the predict step the update step is identical to the Kalman update
described above.

## Usage

The module is exposed through [quantflow.ta.kalman](../api/ta/kalman.md).
A model implements
[GaussianStateSpaceModel](../api/ta/kalman.md#quantflow.ta.kalman.GaussianStateSpaceModel)
(or the convenience subclass
[LinearGaussianModel](../api/ta/kalman.md#quantflow.ta.kalman.LinearGaussianModel))
and is passed to one of the filters:

```python
from quantflow.ta.kalman import KalmanFilter, LinearGaussianModel

model = LinearGaussianModel(F=..., Q=..., H=..., R=...)
kf = KalmanFilter(model)
states, log_lik = kf.filter(observations, dt)
```

The same interface works for the UKF:

```python
from quantflow.ta.kalman import UnscentedKalmanFilter

ukf = UnscentedKalmanFilter(model, alpha=1e-3, beta=2.0, kappa=0.0)
states, log_lik = ukf.filter(observations, dt)
```

Both return the sequence of filtered state estimates and the total
log-likelihood, which can be used directly in maximum-likelihood parameter
estimation. See the [Vasicek curve calibration](../api/rates/vasicek.md)
for a worked example.

## Distribution-Level Interface

The [StateSpaceModel](../api/ta/kalman.md#quantflow.ta.kalman.StateSpaceModel)
exposes a distribution-level API that mirrors the
[particles](https://github.com/nchopin/particles) library.
These methods return [Distribution](../api/dists/index.md) objects and are
intended for particle-filter algorithms (sequential Monte Carlo). They are
abstract on the base class; [LinearGaussianModel](../api/ta/kalman.md#quantflow.ta.kalman.LinearGaussianModel)
implements them by returning [MvNormal](../api/dists/index.md#quantflow.dists.MvNormal)
instances.

| Method | Signature | Returns |
|--------|-----------|---------|
| ``PX0()`` | $(t)$ | Distribution of $X_0$ at time 0 |
| ``PX(t, xp)`` | $(t, x_p)$ | Distribution of $X_t$ given $X_{t-1}=x_p$ |
| ``PY(t, xp, x)`` | $(t, x_p, x)$ | Distribution of $Y_t$ given $X_{t-1}=x_p, X_t=x$ |

For guided or auxiliary particle filters two additional proposal methods
are available:

| Method | Signature | Returns |
|--------|-----------|---------|
| ``proposal0(data)`` | $(\text{data})$ | Proposal for $X_0$ given the first observation |
| ``proposal(t, xp, data)`` | $(t, x_p, \text{data})$ | Proposal for $X_t$ given $X_{t-1}=x_p$ and the data |

For a linear-Gaussian model each of these returns a
``scipy.stats.multivariate_normal``. Non-Gaussian models (e.g. CIR, BNS)
would return their exact transition distributions
($\chi^2$, Lévy-driven, etc.).
