from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, Field
from scipy import linalg
from typing_extensions import Annotated, Doc

from quantflow.dists import MeanAndCov, MvDistribution, MvNormal
from quantflow.utils.types import FloatArray

# ---------------------------------------------------------------------------
# Abstract state-space model
# ---------------------------------------------------------------------------


class StateSpaceModel(BaseModel, ABC):
    r"""Generic state-space model

    \begin{equation}
        \begin{aligned}
            x_t &\sim p_x\left(x_t \mid x_{t-1}\right) \\
            y_t &\sim p_y\left(y_t \mid x_t\right)
        \end{aligned}
    \end{equation}

    where $p_x$ is the transition distribution of the latent state $x_t$ and
    $p_y$ is the observation distribution of the observable $y_t$ given $x_t$.
    """

    @abstractmethod
    def get_px0(self) -> MvDistribution:
        r"""Distribution $p_x(x_0)$ of the initial state $x_0$."""

    @abstractmethod
    def get_px(
        self,
        t: Annotated[int, Doc("Time index $t$.")],
        xp: Annotated[FloatArray, Doc("State at time $t-1$.")],
    ) -> MvDistribution:
        r"""Distribution $p_x\left(x_t \mid x_{t-1}\right)$
        of $x_t$ given $x_{t-1} = x_p$."""

    @abstractmethod
    def get_py(
        self,
        t: Annotated[int, Doc("Time index $t$.")],
        xp: Annotated[FloatArray, Doc("State at time $t-1$.")],
        x: Annotated[FloatArray, Doc("State at time $t$.")],
    ) -> MvDistribution:
        r"""Distribution of $y_t$ given $x_{t-1}=x_p$ and $x_t=x$."""

    # -- Simulation helpers ----------------------------------------------

    def simulate_given_x(
        self,
        x: Annotated[
            list[FloatArray], Doc(r"State trajectory $x_0, \ldots, x_{T-1}$.")
        ],
    ) -> list:
        r"""Simulate observations given a state trajectory.

        Returns $[y_0, \ldots, y_{T-1}]$ where
        $y_t \sim$ [get_py][..get_py]$(t, x_{t-1}, x_t)$.
        """
        lag_x: list = [None] + x[:-1]
        return [
            self.get_py(t, xp, xt).sample(size=1)
            for t, (xp, xt) in enumerate(zip(lag_x, x))
        ]

    def simulate(
        self,
        T: Annotated[int, Doc("Number of time steps.")],
    ) -> tuple[list, list]:
        r"""Simulate state and observation trajectories of length $T$.

        Returns a tuple of two lists: the state trajectory
        $[x_0, \ldots, x_{T-1}]$ and the observation trajectory
        $[y_0, \ldots, y_{T-1}]$.
        """
        x: list = []
        for t in range(T):
            law = self.get_px0() if t == 0 else self.get_px(t, x[-1])
            x.append(law.sample(size=1))
        y = self.simulate_given_x(x)
        return x, y

    def unscented_filter(
        self,
        y: Annotated[FloatArray, Doc("Observations $y_{1:T}$ of shape $(T, n_y)$.")],
    ) -> UnscentedKalmanFilter:
        """Build an [UnscentedKalmanFilter][..UnscentedKalmanFilter] for this
        model and the given observations."""
        return UnscentedKalmanFilter(model=self, y=y)


# ---------------------------------------------------------------------------
# Linear-Gaussian model
# ---------------------------------------------------------------------------


class LinearGaussianModel(StateSpaceModel, arbitrary_types_allowed=True):
    r"""State-space model with linear-Gaussian dynamics.

    \begin{equation}
        \begin{aligned}
            x_t &= F\, x_{t-1} + \varepsilon_t,
                \quad \varepsilon_t \sim N(0, Q) \\
            y_t &= H\, x_t + \eta_t,
                \quad \eta_t \sim N(0, R)
        \end{aligned}
    \end{equation}
    """

    Q: FloatArray = Field(
        description="Process noise covariance $Q$ of shape $(n_x, n_x)$.",
    )
    R: FloatArray = Field(
        description="Observation noise covariance $R$ of shape $(n_y, n_y)$.",
    )
    F: FloatArray = Field(
        default_factory=lambda: np.eye(1),
        description="State transition matrix $F$ of shape $(n_x, n_x)$.",
    )
    H: FloatArray = Field(
        description=(
            "Observation matrix $H$ of shape $(n_y, n_x)$, or $(n_y,)$ "
            "when $n_x = 1$."
        ),
    )
    mu0: FloatArray = Field(
        default_factory=lambda: np.zeros(1),
        description=r"Prior mean $\mu_0$ of $x$ of shape $(n_x,)$.",
    )
    cov0: FloatArray = Field(
        default_factory=lambda: np.eye(1),
        description=r"Prior covariance $\Sigma_0$ of $x$ of shape $(n_x, n_x)$.",
    )

    def model_post_init(self, context: object) -> None:
        """Coerce the model arrays to their canonical shapes.

        Covariance and transition matrices are promoted to 2d, while the mean
        and observation vector are promoted to at least 1d (``H`` keeps its
        $(n_y,)$ column-vector form when $n_x = 1$).
        """
        self.F = np.atleast_2d(self.F)
        self.Q = np.atleast_2d(self.Q)
        self.R = np.atleast_2d(self.R)
        self.cov0 = np.atleast_2d(self.cov0)
        self.mu0 = np.atleast_1d(self.mu0).ravel()
        self.H = np.atleast_1d(self.H)

    @property
    def n_x(self) -> int:
        """State dimension."""
        return self.mu0.shape[0]

    # -- space-state model API ------------------------------------------------

    def get_px0(self) -> MvNormal:
        r"""Initial state distribution $p(x_0) = \mathcal{N}(\mu_0, \Sigma_0)$."""
        return MvNormal(mean=self.mu0, cov=self.cov0)

    def get_px(
        self,
        t: Annotated[int, Doc("Time index $t$.")],
        xp: Annotated[FloatArray, Doc("State at time $t-1$.")],
    ) -> MvNormal:
        r"""Transition distribution of latent state $x_t$ given $x_{t-1}=x_p$.

        \begin{equation}
            p(x_t \mid x_{t-1}) = \mathcal{N}(F x_{t-1}, Q)
        \end{equation}
        """
        return MvNormal(mean=self.F @ xp, cov=self.Q)

    def get_py(
        self,
        t: Annotated[int, Doc("Time index $t$.")],
        xp: Annotated[FloatArray, Doc("State at time $t-1$.")],
        x: Annotated[FloatArray, Doc("State at time $t$.")],
    ) -> MvNormal:
        return MvNormal(mean=(self.H @ x).ravel(), cov=self.R)

    def kalman_filter(
        self,
        y: Annotated[FloatArray, Doc("Observations $y_{1:T}$ of shape $(T, n_y)$.")],
    ) -> KalmanFilter:
        """Convenience method to create a [KalmanFilter][..KalmanFilter] for
        this model and the given observations."""
        return KalmanFilter(model=self, y=y)


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------


def isotropic_noise(
    R: Annotated[FloatArray, Doc("Observation noise covariance of shape (n_y, n_y).")],
) -> float | None:
    r"""Detect isotropic observation noise of the form $R = h^2 I$.

    Returns the scalar variance $h^2$ when every diagonal entry equals $h^2$
    and all off-diagonal entries are zero, otherwise returns ``None``.

    This is the precondition for the Sherman-Morrison fast path in the Kalman
    update: when the noise is isotropic and the state is one-dimensional, the
    innovation covariance is a rank-1 update to a scaled identity and can be
    inverted in $O(n_y)$ instead of $O(n_y^3)$.
    """
    n = R.shape[0]
    h2 = float(R[0, 0])
    if h2 <= 0.0:
        return None
    if n > 1 and not np.allclose(R, h2 * np.eye(n)):
        return None
    return h2


class KalmanFilter(BaseModel, arbitrary_types_allowed=True):
    r"""Kalman filter for a [LinearGaussianModel][..LinearGaussianModel].

    Starting from the prior $(\mu_0, \Sigma_0)$, the forward pass alternates a
    prediction step with an observation update, accumulating the
    log-likelihood of the observations.

    The state estimate of $x_t$ given observations $y_{1:s}$ is written
    $\hat{x}_{t \mid s}$ with covariance $P_{t \mid s}$.

    The **prediction** propagates the filtered state through the linear
    dynamics of the model:

    \begin{equation}
        \begin{aligned}
            \hat{x}_{0 \mid 0} &= \mu_0 \\
            \hat{x}_{t \mid t-1} &= F\, \hat{x}_{t-1 \mid t-1}, \\
            P_{t \mid t-1} &= F\, P_{t-1 \mid t-1}\, F^\top + Q.
        \end{aligned}
    \end{equation}

    The **update** corrects the prediction with the observation $y_t$, where $S_t$
    is the innovation covariance and $K_t$ the optimal Kalman gain:

    \begin{equation}
        \begin{aligned}
            e_t &= y_t - H\, \hat{x}_{t \mid t-1}, \\
            S_t &= H\, P_{t \mid t-1}\, H^\top + R, \\
            K_t &= P_{t \mid t-1}\, H^\top S_t^{-1}, \\
            \hat{x}_{t \mid t} &= \hat{x}_{t \mid t-1} + K_t\, e_t, \\
            P_{t \mid t} &= P_{t \mid t-1} - K_t\, H\, P_{t \mid t-1}.
        \end{aligned}
    \end{equation}

    Each step adds its contribution to the Gaussian log-likelihood:

    \begin{equation}
        \log L = -\frac{1}{2} \sum_t \left(
            n_y \log 2\pi + \log\det S_t + e_t^\top S_t^{-1} e_t
        \right).
    \end{equation}

    When the state is one-dimensional and the observation noise is isotropic
    ($R = h^2 I$), the innovation covariance $S_t = h^2 I + P_{t \mid t-1}\, c c^\top$
    is a rank-1 update to a scaled identity, where $c = H$ is a column vector.

    The [Sherman-Morrison identity](../../glossary.md#sherman-morrison-identity)
    inverts it in $O(n_y)$ instead of $O(n_y^3)$:

    \begin{equation}
        S_t^{-1} = \frac{1}{h^2}\left(
            I - \frac{P_{t \mid t-1}\, c c^\top}{h^2 + P_{t \mid t-1}\, c^\top c}
        \right).
    \end{equation}

    The update detects this case automatically (see
    [isotropic_noise][..isotropic_noise]) and applies the fast path; otherwise
    it falls back to a dense Cholesky solve.
    """

    model: LinearGaussianModel = Field(description="Linear-Gaussian state-space model.")
    y: FloatArray = Field(description="Observations $y_{1:T}$ of shape $(T, n_y)$.")
    states: list[MeanAndCov] = Field(
        default_factory=list,
        description=(
            "List of filtered state estimates. ``states[t]`` is the Gaussian "
            r"approximation of $p(x_t \mid y_{1:t})$ with mean and covariance."
        ),
    )

    def model_post_init(self, context: object) -> None:
        """Ensure the observations are a 2d array of shape $(T, n_y)$."""
        self.y = np.atleast_2d(self.y)

    # -- public API -----------------------------------------------------

    def filter(self) -> float:
        r"""Run the Kalman forward pass.

        Starts from the model prior, then for each observation applies
        [predict][..predict] followed by [update][..update]. Populates
        [states][..states] with the filtered estimates and returns the total
        log-likelihood of the observations.
        """
        self.states = []
        log_lik = 0.0
        pred = self.model.get_px0().mean_and_cov()
        for t in range(self.y.shape[0]):
            if t > 0:
                pred = self.predict(self.states[-1])
            state, ll_inc = self.update(pred, self.y[t])
            log_lik += ll_inc
            self.states.append(state)
        return log_lik

    # -- Kalman recursion -----------------------------------------------

    def predict(
        self,
        state: Annotated[
            MeanAndCov, Doc(r"Filtered state $(\hat{x}_{t-1}, P_{t-1})$.")
        ],
    ) -> MeanAndCov:
        r"""One-step-ahead state prediction.

        \begin{equation}
            \begin{aligned}
                \hat{x}_{t \mid t-1} &= F\, \hat{x}_{t-1 \mid t-1}, \\
                P_{t \mid t-1} &= F\, P_{t-1 \mid t-1}\, F^\top + Q.
            \end{aligned}
        \end{equation}
        """
        model = self.model
        mean = model.F @ state.mean
        cov = model.F @ state.cov @ model.F.T + model.Q
        return MeanAndCov(mean=mean, cov=cov)

    def update(
        self,
        pred: Annotated[
            MeanAndCov,
            Doc(r"Predicted state $(\hat{x}_{t \mid t-1}, P_{t \mid t-1})$."),
        ],
        y: Annotated[FloatArray, Doc(r"Observation $y_t$ of shape $(n_y,)$.")],
    ) -> tuple[MeanAndCov, float]:
        r"""Correct the prediction with the observation $y_t$.

        Returns the filtered state and the contribution of $y_t$ to the
        Gaussian log-likelihood.

        \begin{equation}
            \begin{aligned}
                e_t &= y_t - H\, \hat{x}_{t \mid t-1}, \\
                S_t &= H\, P_{t \mid t-1}\, H^\top + R, \\
                K_t &= P_{t \mid t-1}\, H^\top S_t^{-1}, \\
                \hat{x}_{t \mid t} &= \hat{x}_{t \mid t-1} + K_t\, e_t, \\
                P_{t \mid t} &= P_{t \mid t-1} - K_t\, H\, P_{t \mid t-1}.
            \end{aligned}
        \end{equation}

        The log-likelihood contribution of $y_t$ is

        \begin{equation}
            -\tfrac{1}{2}\left(
                n_y \log 2\pi + \log\det S_t + e_t^\top S_t^{-1} e_t
            \right).
        \end{equation}
        """
        model = self.model
        H = model.H[:, None] if model.H.ndim == 1 else model.H
        obs = np.atleast_1d(np.asarray(y, dtype=float))
        innov = obs - H @ pred.mean
        h2 = isotropic_noise(model.R)
        if h2 is not None and model.n_x == 1:
            return self._update_isotropic(pred, H[:, 0], innov, h2)
        S = H @ pred.cov @ H.T + model.R
        sign, log_det = np.linalg.slogdet(S)
        if sign <= 0:
            raise RuntimeError("innovation covariance is not positive definite")
        quad = float(innov @ linalg.solve(S, innov, assume_a="pos"))
        log_lik = -0.5 * (obs.shape[0] * np.log(2.0 * np.pi) + log_det + quad)
        gain = linalg.solve(S, H @ pred.cov, assume_a="pos").T  # P H^T S^{-1}
        mean = pred.mean + gain @ innov
        cov = pred.cov - gain @ H @ pred.cov
        return MeanAndCov(mean=mean, cov=cov), log_lik

    def _update_isotropic(
        self,
        pred: MeanAndCov,
        c: FloatArray,
        innov: FloatArray,
        h2: float,
    ) -> tuple[MeanAndCov, float]:
        """Sherman-Morrison fast path of [update][..update] for a 1d state with
        isotropic noise.

        Equivalent to the dense path but inverts the rank-1 innovation
        covariance in closed form. See the class docstring and the theory docs
        for the derivation.
        """
        n_y = innov.shape[0]
        p = float(pred.cov[0, 0])
        d = h2 + p * float(c @ c)  # h^2 + P c^T c
        ce = float(c @ innov)  # c^T e_t
        log_det = (n_y - 1) * np.log(h2) + np.log(d)
        quad = (float(innov @ innov) - p * ce * ce / d) / h2
        log_lik = -0.5 * (n_y * np.log(2.0 * np.pi) + log_det + quad)
        gain = (p / d) * c  # K_t = (P / d) c^T, shape (n_y,)
        mean = pred.mean + gain @ innov
        cov = np.array([[p * h2 / d]])
        return MeanAndCov(mean=mean, cov=cov), log_lik


# ---------------------------------------------------------------------------
# Unscented Kalman filter
# ---------------------------------------------------------------------------


class UnscentedKalmanFilter(BaseModel, arbitrary_types_allowed=True):
    r"""Unscented Kalman filter (UKF) for a [StateSpaceModel][..StateSpaceModel].

    The UKF handles non-linear, non-Gaussian dynamics by propagating
    $2 n_x + 1$ sigma points through the model transition
    ([get_px][..StateSpaceModel.get_px]) and observation
    ([get_py][..StateSpaceModel.get_py]) distributions, matching the first two
    moments of each. Only the conditional mean and covariance of those
    distributions are used, so any model that exposes them is supported.

    The predicted moments combine the spread of the propagated means with the
    average conditional covariance (the law of total variance):

    \begin{equation}
        \begin{aligned}
            \hat{x}_{t \mid t-1} &= \sum_i W^m_i\, \mu(\chi_i), \\
            P_{t \mid t-1} &= \sum_i W^c_i
                \bigl(\mu(\chi_i) - \hat{x}_{t \mid t-1}\bigr)
                \bigl(\mu(\chi_i) - \hat{x}_{t \mid t-1}\bigr)^\top
                + \sum_i W^m_i\, \Sigma(\chi_i),
        \end{aligned}
    \end{equation}

    where $\chi_i$ are the sigma points and $\mu, \Sigma$ are the mean and
    covariance of [get_px][..StateSpaceModel.get_px]. The observation update
    forms the innovation covariance $S_t$ and cross covariance $C_t$ the same
    way through [get_py][..StateSpaceModel.get_py], with gain
    $K_t = C_t S_t^{-1}$. For a [LinearGaussianModel][..LinearGaussianModel] the
    filter reduces to the exact [KalmanFilter][..KalmanFilter].
    """

    model: StateSpaceModel = Field(description="State-space model.")
    y: FloatArray = Field(description="Observations $y_{1:T}$ of shape $(T, n_y)$.")
    states: list[MeanAndCov] = Field(
        default_factory=list,
        description=(
            "List of filtered state estimates. ``states[t]`` is the Gaussian "
            r"approximation of $p(x_t \mid y_{1:t})$ with mean and covariance."
        ),
    )
    alpha: float = Field(default=1e-3, description="Sigma-point spread.")
    beta: float = Field(
        default=2.0, description="Prior knowledge factor (2 is optimal for Gaussians)."
    )
    kappa: float = Field(default=0.0, description="Secondary scaling parameter.")

    def model_post_init(self, context: object) -> None:
        """Ensure the observations are a 2d array of shape $(T, n_y)$."""
        self.y = np.atleast_2d(self.y)

    # -- public API -----------------------------------------------------

    def filter(self) -> float:
        r"""Run the UKF forward pass.

        Populates [states][..states] with the filtered estimates and returns
        the total log-likelihood of the observations.
        """
        model = self.model
        observations = self.y
        prior = model.get_px0().mean_and_cov()
        x: FloatArray = np.atleast_1d(np.asarray(prior.mean, dtype=float))
        P: FloatArray = np.atleast_2d(np.asarray(prior.cov, dtype=float))
        n_x = x.shape[0]
        Wm, Wc, lmda = self._weights(n_x)

        self.states.clear()
        log_lik = 0.0
        for t in range(observations.shape[0]):
            if t > 0:
                x, P = self._predict(t, x, P, lmda, Wm, Wc)
            x, P, ll_inc = self._update(t, x, P, observations[t], lmda, Wm, Wc)
            log_lik += ll_inc
            self.states.append(MeanAndCov(mean=x.copy(), cov=P.copy()))

        return log_lik

    # -- internals ------------------------------------------------------

    def _weights(self, n: int) -> tuple[FloatArray, FloatArray, float]:
        r"""Mean and covariance sigma-point weights for an $n$-dimensional state."""
        lmda = self.alpha * self.alpha * (n + self.kappa) - n
        Wm = np.full(2 * n + 1, 0.5 / (n + lmda))
        Wc = Wm.copy()
        Wm[0] = lmda / (n + lmda)
        Wc[0] = lmda / (n + lmda) + 1.0 - self.alpha * self.alpha + self.beta
        return Wm, Wc, lmda

    def _sigma_points(self, x: FloatArray, P: FloatArray, lmda: float) -> FloatArray:
        r"""Generate the $2 n_x + 1$ sigma points of the Gaussian $(x, P)$."""
        n_x = x.shape[0]
        scaled = (n_x + lmda) * P
        try:
            L = np.linalg.cholesky(scaled)
        except np.linalg.LinAlgError:
            # fallback for a near-singular covariance
            vals, vecs = np.linalg.eigh(scaled)
            L = vecs @ np.diag(np.sqrt(np.maximum(vals, 0.0)))
        sigma = np.empty((2 * n_x + 1, n_x))
        sigma[0] = x
        for i in range(n_x):
            sigma[1 + i] = x + L[:, i]
            sigma[1 + n_x + i] = x - L[:, i]
        return sigma

    def _predict(
        self,
        t: int,
        x: FloatArray,
        P: FloatArray,
        lmda: float,
        Wm: FloatArray,
        Wc: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        r"""Sigma-point prediction through the transition distribution."""
        sigma = self._sigma_points(x, P, lmda)
        laws = [self.model.get_px(t, s).mean_and_cov() for s in sigma]
        means = np.array(
            [np.atleast_1d(np.asarray(law.mean, dtype=float)) for law in laws]
        )
        x_new = Wm @ means
        n_x = x_new.shape[0]
        P_new = np.zeros((n_x, n_x))
        for i, law in enumerate(laws):
            diff = means[i] - x_new
            P_new += Wc[i] * np.outer(diff, diff)
            P_new += Wm[i] * np.atleast_2d(np.asarray(law.cov, dtype=float))
        return x_new, P_new

    def _update(
        self,
        t: int,
        x: FloatArray,
        P: FloatArray,
        y: FloatArray,
        lmda: float,
        Wm: FloatArray,
        Wc: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float]:
        r"""Sigma-point observation update with Gaussian log-likelihood."""
        sigma = self._sigma_points(x, P, lmda)
        laws = [self.model.get_py(t, x, s).mean_and_cov() for s in sigma]
        gamma = np.array(
            [np.atleast_1d(np.asarray(law.mean, dtype=float)) for law in laws]
        )
        y_hat = Wm @ gamma
        n_y = y_hat.shape[0]
        n_x = x.shape[0]
        S = np.zeros((n_y, n_y))
        C = np.zeros((n_x, n_y))
        for i, law in enumerate(laws):
            dy = gamma[i] - y_hat
            S += Wc[i] * np.outer(dy, dy)
            S += Wm[i] * np.atleast_2d(np.asarray(law.cov, dtype=float))
            C += Wc[i] * np.outer(sigma[i] - x, dy)

        innov = np.atleast_1d(np.asarray(y, dtype=float)) - y_hat
        K = linalg.solve(S, C.T, assume_a="pos").T  # C @ S^{-1}
        x_new = x + K @ innov
        P_new = P - K @ S @ K.T

        sign, log_det = np.linalg.slogdet(S)
        if sign <= 0:
            raise RuntimeError("innovation covariance is not positive definite")
        quad = float(innov @ linalg.solve(S, innov, assume_a="pos"))
        ll_inc = -0.5 * (n_y * np.log(2.0 * np.pi) + log_det + quad)
        return x_new, P_new, ll_inc
