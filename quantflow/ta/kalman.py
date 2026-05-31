from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, Field
from scipy import linalg
from typing_extensions import Annotated, Doc

from quantflow.dists import Distribution, MeanAndCov, MvNormal
from quantflow.utils.types import FloatArray

# ---------------------------------------------------------------------------
# Abstract state-space model
# ---------------------------------------------------------------------------


class StateSpaceModel(BaseModel, ABC):
    r"""Generic state-space model with additive Gaussian observation noise.

    \begin{equation}
        \begin{aligned}
            x_t &= f(x_{t-1}, \Delta t) + \varepsilon_t,
                \quad \varepsilon_t \sim N(0, Q_t) \\
            y_t &= H_t\, x_t + d_t + \eta_t,
                \quad \eta_t \sim N(0, R_t)
        \end{aligned}
    \end{equation}
    """

    @abstractmethod
    def get_px0(self) -> Distribution:
        """Distribution of the initial state $x_0$."""

    @abstractmethod
    def get_px(
        self,
        t: Annotated[int, Doc("Time index $t$.")],
        xp: Annotated[FloatArray, Doc("State at time $t-1$.")],
    ) -> Distribution:
        """Distribution of $x_t$ given $x_{t-1} = x_p$."""

    @abstractmethod
    def get_py(
        self,
        t: Annotated[int, Doc("Time index $t$.")],
        xp: Annotated[FloatArray, Doc("State at time $t-1$.")],
        x: Annotated[FloatArray, Doc("State at time $t$.")],
    ) -> Distribution:
        """Distribution of $y_t$ given $x_{t-1}=x_p$ and $x_t=x$."""

    @abstractmethod
    def get_proposal0(
        self,
        data: Annotated[FloatArray, Doc("Observation data up to time $t$.")],
    ) -> Distribution:
        """Get the proposal distribution for $y_0$ given observation data."""

    @abstractmethod
    def get_proposal(
        self,
        t: Annotated[int, Doc("Time index $t$.")],
        xp: Annotated[FloatArray, Doc("State at time $t-1$.")],
        data: Annotated[FloatArray, Doc("Observation data up to time $t$.")],
    ) -> Distribution:
        """Get the proposal distribution for $y_t$ given $x_{t-1}=x_p$
        and observation data."""

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
        return MvNormal(mean=self.F @ xp, cov=self.Q)

    def get_py(
        self,
        t: Annotated[int, Doc("Time index $t$.")],
        xp: Annotated[FloatArray, Doc("State at time $t-1$.")],
        x: Annotated[FloatArray, Doc("State at time $t$.")],
    ) -> MvNormal:
        return MvNormal(mean=(self.H @ x).ravel(), cov=self.R)

    def get_proposal0(
        self,
        data: Annotated[FloatArray, Doc("Observation data of shape $(T, n_y)$.")],
    ) -> MvNormal:
        r"""Optimal proposal for the initial state $x_0$ given $y_0$.

        Applies a single Kalman update to the prior $(\mu_0, \Sigma_0)$ using
        the first observation, returning the exact filtering distribution
        $p(x_0 \mid y_0)$.
        """
        return self._get_proposal(self.mu0, self.cov0, data[0])

    def get_proposal(
        self,
        t: Annotated[int, Doc("Time index $t$.")],
        xp: Annotated[FloatArray, Doc("State at time $t-1$.")],
        data: Annotated[FloatArray, Doc("Observation data of shape $(T, n_y)$.")],
    ) -> MvNormal:
        r"""Optimal proposal for $x_t$ given $x_{t-1}=x_p$ and $y_t$.

        Predicts one step forward with the transition dynamics, then applies a
        Kalman update with the observation at time $t$ to return the exact
        filtering distribution $p(x_t \mid x_{t-1}, y_t)$.
        """
        pred_mean = (self.F @ np.atleast_1d(xp)).ravel()
        return self._get_proposal(pred_mean, self.Q, data[t])

    def _get_proposal(
        self,
        pred_mean: Annotated[FloatArray, Doc("Predicted state mean.")],
        pred_cov: Annotated[FloatArray, Doc("Predicted state covariance.")],
        y: Annotated[FloatArray, Doc("Observation $y_t$ of shape $(n_y,)$.")],
    ) -> MvNormal:
        r"""Kalman update of a Gaussian prediction $(m, P)$ with observation $y$.

        \begin{equation}
            \begin{aligned}
                S &= H P H^\top + R, \quad K = P H^\top S^{-1} \\
                m' &= m + K (y - H m), \quad P' = P - K H P
            \end{aligned}
        \end{equation}
        """
        H = self.H[:, None] if self.H.ndim == 1 else self.H
        S = H @ pred_cov @ H.T + self.R
        residual = np.atleast_1d(y) - H @ pred_mean
        gain = linalg.solve(S, H @ pred_cov, assume_a="pos").T
        mean = pred_mean + gain @ residual
        cov = pred_cov - gain @ H @ pred_cov
        return MvNormal(mean=mean.ravel(), cov=cov)


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------


class KalmanFilter(BaseModel, arbitrary_types_allowed=True):
    r"""Kalman filter for a [LinearGaussianModel][..LinearGaussianModel].

    Starting from the prior $(\mu_0, \Sigma_0)$, the forward pass alternates a
    prediction step with an observation update, accumulating the
    log-likelihood of the observations.

    The state estimate of $x_t$ given observations $y_{1:s}$ is written
    $\hat{x}_{t \mid s}$ with covariance $P_{t \mid s}$. The prediction
    propagates the filtered state through the linear dynamics of the model:

    \begin{equation}
        \begin{aligned}
            \hat{x}_{t \mid t-1} &= F\, \hat{x}_{t-1 \mid t-1}, \\
            P_{t \mid t-1} &= F\, P_{t-1 \mid t-1}\, F^\top + Q.
        \end{aligned}
    \end{equation}

    The update corrects the prediction with the observation $y_t$, where $S_t$
    is the innovation covariance and $K_t$ the Kalman gain:

    \begin{equation}
        \begin{aligned}
            S_t &= H\, P_{t \mid t-1}\, H^\top + R, \\
            K_t &= P_{t \mid t-1}\, H^\top S_t^{-1}, \\
            \hat{x}_{t \mid t} &= \hat{x}_{t \mid t-1}
                + K_t\,(y_t - H\, \hat{x}_{t \mid t-1}), \\
            P_{t \mid t} &= P_{t \mid t-1} - K_t\, H\, P_{t \mid t-1}.
        \end{aligned}
    \end{equation}

    Each step adds its contribution to the Gaussian log-likelihood, with
    innovation $e_t = y_t - H\, \hat{x}_{t \mid t-1}$:

    \begin{equation}
        \log L = -\frac{1}{2} \sum_t \left(
            n_y \log 2\pi + \log\det S_t + e_t^\top S_t^{-1} e_t
        \right).
    \end{equation}

    The update uses the Sherman-Morrison identity for $O(n_y)$ innovation
    inversion when the observation matrix is a column vector and the
    observation covariance is a scaled identity $h^2 I$, and falls back to the
    general update otherwise.
    """

    model: LinearGaussianModel = Field(description="Linear-Gaussian state-space model.")
    data: FloatArray = Field(description="Observation data of shape $(T, n_y)$.")

    def model_post_init(self, context: object) -> None:
        """Ensure the observation data is a 2d array of shape $(T, n_y)$."""
        self.data = np.atleast_2d(self.data)

    # -- public API -----------------------------------------------------

    def filter(
        self,
    ) -> Annotated[
        tuple[list[MeanAndCov], float],
        Doc(
            "Tuple of ``(filtered_states, log_likelihood)``. "
            "``filtered_states[t]`` is the Gaussian approximation of "
            r"$p(x_t \mid y_{1:t})$."
        ),
    ]:
        r"""Run the Kalman forward pass.

        Returns the sequence of filtered state estimates and the total
        log-likelihood of the observations.
        """
        model = self.model
        F, Q = model.F, model.Q
        observations = self.data
        x: FloatArray = np.atleast_1d(np.asarray(model.mu0, dtype=float))
        P: FloatArray = np.atleast_2d(np.asarray(model.cov0, dtype=float))
        n_obs, n_y = observations.shape
        n_x = x.shape[0]
        if P.shape != (n_x, n_x):
            raise ValueError(
                f"covariance shape {P.shape} is inconsistent with state "
                f"dimension {n_x}"
            )
        states: list[MeanAndCov] = []
        log_lik: float = -0.5 * n_obs * n_y * np.log(2.0 * np.pi)

        for t in range(n_obs):
            # predict
            if t > 0:
                x = F @ x
                P = F @ P @ F.T + Q

            # update with the observation
            x, P, ll_inc = self._update(x, P, observations[t], model.H, model.R)
            log_lik += ll_inc
            states.append(MeanAndCov(mean=x.copy(), cov=P.copy()))

        return states, log_lik

    # -- internal update ------------------------------------------------

    @staticmethod
    def _update(
        x: FloatArray,
        P: FloatArray,
        y: FloatArray,
        H: FloatArray,
        R: FloatArray,
        d: FloatArray | float = 0.0,
    ) -> tuple[FloatArray, FloatArray, float]:
        r"""Single Kalman update step.

        Returns ``(x_new, P_new, ll_inc)`` where ``ll_inc`` is the
        contribution to the Gaussian log-likelihood:
        $-0.5(\log\det S + e^\top S^{-1} e)$
        with $e = y - (H x + d)$ and $S = H P H^\top + R$.
        """
        # detect Sherman-Morrison fast-path
        use_sm = KalmanFilter._use_sherman_morrison(H, R)

        # predicted observation mean
        if H.ndim == 1:
            y_pred = H * x.item() + d  # scalar state * column vector
        else:
            y_pred = H @ x + d  # shape (n_y,)
        innov = y - y_pred  # innovation

        if use_sm:
            return KalmanFilter._update_sherman_morrison(x, P, innov, H, R)
        else:
            return KalmanFilter._update_general(x, P, innov, H, R)

    @staticmethod
    def _use_sherman_morrison(H: FloatArray, R: FloatArray) -> bool:
        """True when H is a column vector and R is a scaled identity."""
        if H.ndim != 1:
            return False
        n_y = H.shape[0]
        if R.shape != (n_y, n_y):
            return False
        # check R = h^2 * I
        if not np.allclose(np.diag(np.diag(R)), R):
            return False
        diag = np.diag(R)
        if not np.allclose(diag, diag[0]):
            return False
        return True

    @staticmethod
    def _update_sherman_morrison(
        x: FloatArray,
        P: FloatArray,
        innov: FloatArray,
        H: FloatArray,
        R: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float]:
        r"""Kalman update via the Sherman-Morrison identity.

        When $R = h^2 I$ and $H$ is a column vector $c$, the innovation
        covariance $S = h^2 I + P \, c c^\top$ is a rank-1 update to a
        scaled identity, inverted in $O(n_y)$.
        """
        c = H  # column vector, shape (n_y,)
        h2 = float(R[0, 0])  # h^2
        n_y = c.shape[0]

        p_scalar: float = float(P.item())  # P is (1, 1) or scalar
        cc: float = float(c @ c)  # c^T c
        cv: float = float(c @ innov)  # c^T innov
        denom: float = h2 + p_scalar * cc  # h^2 + P c^T c

        # log |S|
        log_det = (n_y - 1) * np.log(h2) + np.log(denom)
        # e^T S^{-1} e
        quad = (float(innov @ innov) - p_scalar * cv * cv / denom) / h2
        ll_inc = -0.5 * (log_det + quad)

        # state update
        x_new = x + p_scalar * cv / denom
        P_new = p_scalar * h2 / denom

        # preserve array shapes
        return np.atleast_1d(x_new), np.atleast_2d(P_new), ll_inc

    @staticmethod
    def _update_general(
        x: FloatArray,
        P: FloatArray,
        innov: FloatArray,
        H: FloatArray,
        R: FloatArray,
    ) -> tuple[FloatArray, FloatArray, float]:
        r"""Standard Kalman update with full linear algebra.

        $S = H P H^\top + R$, $K = P H^\top S^{-1}$.
        """
        if H.ndim == 1:
            H2 = H[:, None]  # (n_y,) -> (n_y, 1)
        else:
            H2 = H
        S = H2 @ P @ H2.T + R  # innovation covariance
        # log |S| + e^T S^{-1} e
        sign, log_det = np.linalg.slogdet(S)
        if sign <= 0:
            raise RuntimeError("innovation covariance is not positive definite")
        S_inv_innov = linalg.solve(S, innov, assume_a="pos")
        quad = float(innov @ S_inv_innov)
        ll_inc = -0.5 * (log_det + quad)

        # Kalman gain and update
        K = linalg.solve(S, H2 @ P.T, assume_a="pos").T  # P @ H^T @ S^{-1}
        x_new = x + K @ innov
        P_new = P - K @ H2 @ P
        return x_new, P_new, ll_inc


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
    data: FloatArray = Field(description="Observation data of shape $(T, n_y)$.")
    alpha: float = Field(default=1e-3, description="Sigma-point spread.")
    beta: float = Field(
        default=2.0, description="Prior knowledge factor (2 is optimal for Gaussians)."
    )
    kappa: float = Field(default=0.0, description="Secondary scaling parameter.")

    def model_post_init(self, context: object) -> None:
        """Ensure the observation data is a 2d array of shape $(T, n_y)$."""
        self.data = np.atleast_2d(self.data)

    # -- public API -----------------------------------------------------

    def filter(
        self,
    ) -> Annotated[
        tuple[list[MeanAndCov], float],
        Doc(
            "Tuple of ``(filtered_states, log_likelihood)``. "
            "``filtered_states[t]`` is the Gaussian approximation of "
            r"$p(x_t \mid y_{1:t})$."
        ),
    ]:
        r"""Run the UKF forward pass.

        Returns the sequence of filtered state estimates and the total
        log-likelihood of the observations.
        """
        model = self.model
        observations = self.data
        prior = model.get_px0().mean_and_cov()
        x: FloatArray = np.atleast_1d(np.asarray(prior.mean, dtype=float))
        P: FloatArray = np.atleast_2d(np.asarray(prior.cov, dtype=float))
        n_x = x.shape[0]
        Wm, Wc, lmda = self._weights(n_x)

        states: list[MeanAndCov] = []
        log_lik = 0.0
        for t in range(observations.shape[0]):
            if t > 0:
                x, P = self._predict(t, x, P, lmda, Wm, Wc)
            x, P, ll_inc = self._update(t, x, P, observations[t], lmda, Wm, Wc)
            log_lik += ll_inc
            states.append(MeanAndCov(mean=x.copy(), cov=P.copy()))

        return states, log_lik

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
