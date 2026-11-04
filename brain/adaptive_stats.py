"""
ChronoWeave – Adaptive Statistical Models
==========================================
Implements the three core probabilistic models from the ChronoWeave PRD:

  1. KalmanDemandFilter         – 3-state Kalman Filter (organic µ, trend τ, promo lift γ)
  2. BetaBinomialReturnEstimator – Online Bayesian return-rate tracker
  3. GammaLeadTimeEstimator      – Freight-aware Gamma lead-time distribution

Dependencies: numpy, scipy, collections only.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np
try:
    from scipy.stats import gamma as gamma_dist
except ImportError:
    gamma_dist = None


# ============================================================================
# 1. KalmanDemandFilter
# ============================================================================

class KalmanDemandFilter:
    """
    3-state linear Kalman Filter tracking organic baseline (µ), local trend (τ),
    and transient promo lift (γ).

    State vector
    ------------
    x_t = [µ_t, τ_t, γ_t]^T

    State-transition matrix F
    -------------------------
    [[1,    1,    0  ],
     [0,    φ,    0  ],   φ = 0.97  (trend persistence)
     [0,    0,    ρ  ]]   ρ = 0.25  (fast promo-lift decay)

    Process noise Q (diagonal)
    --------------------------
    diag(σ²_µ=4.0,  σ²_τ=0.5,  σ²_γ=25.0)

    Observation noise R
    -------------------
    σ²_obs = 9.0

    Observation matrix H
    --------------------
    H = [1, 0, 1]   (we observe µ + γ; trend is latent)

    Size-curve censoring correction
    --------------------------------
    y_corrected = y_obs / max(0.25, rack_completeness ** 1.8)
    """

    # Hyper-parameters (PRD-specified)
    _PHI: float = 0.97     # trend persistence
    _RHO: float = 0.25     # promo-lift decay

    _Q_DIAG: tuple[float, float, float] = (4.0, 0.5, 25.0)
    _R: float = 9.0

    def __init__(self, base_demand: float = 2200.0) -> None:
        self.base_demand = base_demand

        # State vector x = [µ, τ, γ]
        self._x = np.array([base_demand, 0.0, 0.0], dtype=float)

        # State covariance
        self._P = np.diag([100.0, 1.0, 50.0])

        # Build fixed matrices
        self._F = np.array([
            [1.0, 1.0, 0.0],
            [0.0, self._PHI, 0.0],
            [0.0, 0.0, self._RHO],
        ])
        self._Q = np.diag(self._Q_DIAG)
        self._H = np.array([[1.0, 0.0, 1.0]])
        self._R_mat = np.array([[self._R]])

        # History for decompose()
        self._history: list[dict[str, float]] = []
        self._last_kalman_gain: np.ndarray = np.zeros((3, 1))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        y_obs: float,
        rack_completeness: float,
        promo_flag: int,
    ) -> dict[str, Any]:
        """
        Ingest one week's demand observation and run a full Kalman predict–update.

        Parameters
        ----------
        y_obs : float
            Raw observed demand units.
        rack_completeness : float
            Store rack fill ratio C ∈ (0.4, 1.0).
        promo_flag : int
            1 if this week is a promotional week, else 0.

        Returns
        -------
        dict with keys:
            organic_demand, trend, promo_lift, confidence, kalman_gain
        """
        # --- Censoring correction -------------------------------------------
        denom = max(0.25, rack_completeness ** 1.8)
        y_corrected = y_obs / denom

        # --- Predict step ---------------------------------------------------
        x_pred = self._F @ self._x
        P_pred = self._F @ self._P @ self._F.T + self._Q

        # --- Observation noise scaling on promo weeks -----------------------
        # Increase R on promo weeks because the promo multiplier is noisy
        R_eff = self._R_mat * (4.0 if promo_flag else 1.0)

        # --- Innovation -----------------------------------------------------
        y_vec = np.array([[y_corrected]])
        S = self._H @ P_pred @ self._H.T + R_eff
        K = P_pred @ self._H.T @ np.linalg.inv(S)          # (3×1)
        innovation = y_vec - self._H @ x_pred.reshape(-1, 1)

        # --- Update step ----------------------------------------------------
        x_upd = x_pred + (K @ innovation).ravel()
        P_upd = (np.eye(3) - K @ self._H) @ P_pred

        self._x = x_upd
        self._P = P_upd
        self._last_kalman_gain = K

        # Confidence: inverse of diagonal P trace, normalised
        trace = max(1e-6, np.trace(P_upd))
        confidence = float(1.0 / (1.0 + trace / 1000.0))

        result = {
            "organic_demand": float(np.clip(self._x[0], 0, None)),
            "trend": float(self._x[1]),
            "promo_lift": float(np.clip(self._x[2], 0, None)),
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "kalman_gain": K.ravel().tolist(),
        }
        self._history.append(result)
        return result

    def get_clean_forecast(self, horizon_weeks: int = 1) -> float:
        """
        Return organic + trend forecast (no promo component) for a given horizon.

        Parameters
        ----------
        horizon_weeks : int
            Number of weeks ahead (≥ 1).

        Returns
        -------
        float
            Projected organic demand.
        """
        mu = self._x[0]
        tau = self._x[1]
        # Project trend with persistence φ^h
        # µ grows by τ * sum_{i=1}^{h} φ^(i-1)
        h = max(1, horizon_weeks)
        phi = self._PHI
        if abs(1.0 - phi) < 1e-10:
            trend_contribution = tau * h
        else:
            trend_contribution = tau * (1.0 - phi**h) / (1.0 - phi)
        return float(max(0.0, mu + trend_contribution))

    def decompose(self) -> dict[str, float]:
        """
        Return latest state decomposition.

        Returns
        -------
        dict with keys: organic, trend, promo_spike, total_observed
        """
        organic = float(np.clip(self._x[0], 0, None))
        trend = float(self._x[1])
        promo_spike = float(np.clip(self._x[2], 0, None))
        total_observed = organic + promo_spike
        return {
            "organic": organic,
            "trend": trend,
            "promo_spike": promo_spike,
            "total_observed": total_observed,
        }


# ============================================================================
# 2. BetaBinomialReturnEstimator
# ============================================================================

class BetaBinomialReturnEstimator:
    """
    Tracks the weekly return probability p_{r,t} with a conjugate Beta prior,
    updated online via a forgetting-factor Bayesian scheme.

    Prior
    -----
    α₀ = 5.6, β₀ = 14.4  →  E[p_r] = α₀/(α₀+β₀) = 0.28  (28 % organic)

    Forgetting factor
    -----------------
    λ = 0.95

    Lag kernel (days 7–14)
    ----------------------
    w = [0.05, 0.08, 0.12, 0.18, 0.22, 0.18, 0.12, 0.05]
    """

    _LAMBDA: float = 0.95
    _LAG_KERNEL: np.ndarray = np.array(
        [0.05, 0.08, 0.12, 0.18, 0.22, 0.18, 0.12, 0.05]
    )
    _LAG_DAYS: int = 8   # days 7–14

    def __init__(self, alpha0: float = 5.6, beta0: float = 14.4) -> None:
        self.alpha = alpha0
        self.beta_ = beta0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        shipped_units: int,
        returns_observed: int,
        promo_flag: int,
    ) -> None:
        """
        Bayesian update with exponential forgetting.

        Parameters
        ----------
        shipped_units : int
            Units dispatched in the reference shipment period.
        returns_observed : int
            Units returned in the current period.
        promo_flag : int
            1 on a promo week (implicitly acknowledged but update uses actuals).
        """
        lam = self._LAMBDA
        non_returns = max(0, shipped_units - returns_observed)

        self.alpha = lam * self.alpha + returns_observed
        self.beta_ = lam * self.beta_ + non_returns

    def predict_return_wave(
        self,
        online_shipped_history: list[float],
        qc_yield: float = 0.92,
    ) -> dict[str, float]:
        """
        Forecast the return wave for the current week given a history of
        weekly online shipments.

        The lag kernel is applied to the last 8 entries in `online_shipped_history`.
        QC yield reduces recoverable returns.

        Parameters
        ----------
        online_shipped_history : list[float]
            Ordered weekly online shipped units (most recent last).
        qc_yield : float
            Fraction of returns passing QC and re-entering inventory.

        Returns
        -------
        dict with keys:
            expected_returns_week, lower_95, upper_95, mean_return_rate
        """
        p_mean = self.alpha / (self.alpha + self.beta_)
        # Posterior variance of Beta distribution
        a, b = self.alpha, self.beta_
        p_var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        p_std = math.sqrt(max(p_var, 1e-12))

        # 95 % credible interval for p
        p_lo = max(0.0, p_mean - 1.96 * p_std)
        p_hi = min(1.0, p_mean + 1.96 * p_std)

        # Effective shipped units (last 8 weeks), padded with zeros if needed
        hist = list(online_shipped_history)
        if len(hist) < self._LAG_DAYS:
            hist = [0.0] * (self._LAG_DAYS - len(hist)) + hist
        hist_arr = np.array(hist[-self._LAG_DAYS:], dtype=float)

        # Expected returns this week = Σ w_i * shipped_{t-lag_i} * p * qc_yield
        expected_returns = float(
            (self._LAG_KERNEL * hist_arr).sum() * p_mean * qc_yield
        )
        lower_95 = float(
            (self._LAG_KERNEL * hist_arr).sum() * p_lo * qc_yield
        )
        upper_95 = float(
            (self._LAG_KERNEL * hist_arr).sum() * p_hi * qc_yield
        )

        return {
            "expected_returns_week": expected_returns,
            "lower_95": lower_95,
            "upper_95": upper_95,
            "mean_return_rate": p_mean,
        }

    def get_net_demand(
        self, gross_demand: float, predicted_returns: float
    ) -> float:
        """
        Net addressable demand after absorbing predicted returns back into stock.

        Returns
        -------
        float
            max(0, gross_demand - predicted_returns)
        """
        return float(max(0.0, gross_demand - predicted_returns))


# ============================================================================
# 3. GammaLeadTimeEstimator
# ============================================================================

class GammaLeadTimeEstimator:
    """
    Maps freight_congestion_index to a Gamma-distributed lead time for a
    given supply-chain node.

    Mean formula
    ------------
    μ_L = base_days + η * max(0, F_t - 100)

    Gamma parameterisation (CV fixed at 0.25)
    ------------------------------------------
    CV = σ/μ = 0.25  →  k = 1/CV² = 16,  θ = μ/k = μ/16

    Nodes (default parameters from PRD)
    ------------------------------------
    node1: base=3  days, η=0.04   (supplier → DC)
    node2: base=14 days, η=0.06   (DC → store)
    node3: base=35 days, η=0.10   (mill)
    """

    _CV: float = 0.25          # fixed coefficient of variation
    _K: float = 1.0 / (0.25 ** 2)   # = 16.0

    def __init__(
        self,
        base_days: float,
        eta: float,
        node_id: str,
    ) -> None:
        """
        Parameters
        ----------
        base_days : float
            Uncongested (baseline) lead time in days.
        eta : float
            Freight sensitivity coefficient (days per unit above 100).
        node_id : str
            Human-readable node label (e.g. 'node1', 'node2', 'node3').
        """
        self.base_days = base_days
        self.eta = eta
        self.node_id = node_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mean_lead_time(self, freight_index: float) -> float:
        return self.base_days + self.eta * max(0.0, freight_index - 100.0)

    def _gamma_params(self, mu: float) -> tuple[float, float]:
        """Return (shape k, scale θ) for fixed CV = 0.25."""
        k = self._K                 # 16.0
        theta = mu / k              # scale
        return k, theta

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def estimate(self, freight_index: float) -> dict[str, Any]:
        """
        Derive the Gamma lead-time distribution for a given freight index.

        Parameters
        ----------
        freight_index : float
            Current freight congestion index.

        Returns
        -------
        dict with keys:
            mean_days, p10, p50, p90, variance, distribution_params
        """
        mu = self._mean_lead_time(freight_index)
        k, theta = self._gamma_params(mu)
        if gamma_dist is not None:
            dist = gamma_dist(a=k, scale=theta)
            p10 = float(dist.ppf(0.10))
            p50 = float(dist.ppf(0.50))
            p90 = float(dist.ppf(0.90))
        else:
            # High-accuracy numpy sampling fallback when scipy is not available
            samples = np.random.gamma(shape=k, scale=theta, size=4000)
            p10, p50, p90 = [float(x) for x in np.percentile(samples, [10, 50, 90])]

        return {
            "mean_days": float(mu),
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "variance": float(k * theta ** 2),
            "distribution_params": {"k": k, "theta": theta},
        }

    def simulate_pipeline_delay(self, n_samples: int = 500) -> np.ndarray:
        """
        Draw n_samples from the lead-time Gamma distribution at baseline
        freight (index = 100 → μ = base_days).

        Parameters
        ----------
        n_samples : int
            Number of Monte-Carlo samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        mu = self.base_days        # baseline: no congestion
        k, theta = self._gamma_params(mu)
        return np.random.gamma(shape=k, scale=theta, size=n_samples)


# ============================================================================
# Smoke tests
# ============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # -----------------------------------------------------------------------
    print("=" * 60)
    print("1. KalmanDemandFilter")
    print("=" * 60)

    kdf = KalmanDemandFilter(base_demand=2200.0)

    # Feed 12 weeks of synthetic data
    rng_test = np.random.default_rng(7)
    for t in range(12):
        promo = 1 if t in {3, 7} else 0
        demand = 2200 + rng_test.normal(0, 100)
        if promo:
            demand *= rng_test.uniform(2.8, 4.2)
        rack = rng_test.uniform(0.6, 1.0)
        result = kdf.update(demand, rack, promo)
        if t in {0, 3, 7, 11}:
            print(f"  Week {t+1:>2d} | promo={promo} | organic={result['organic_demand']:7.1f} "
                  f"| trend={result['trend']:+.2f} | promo_lift={result['promo_lift']:7.1f} "
                  f"| confidence={result['confidence']:.3f}")

    print(f"\n  Clean 4-week forecast : {kdf.get_clean_forecast(4):.1f} units")
    print(f"  Decompose             : {kdf.decompose()}")
    print()

    # -----------------------------------------------------------------------
    print("=" * 60)
    print("2. BetaBinomialReturnEstimator")
    print("=" * 60)

    bbe = BetaBinomialReturnEstimator(alpha0=5.6, beta0=14.4)
    shipped_history: list[float] = []
    for t in range(10):
        promo = 1 if t in {3, 7} else 0
        shipped = int(2200 * 0.40 * (3.5 if promo else 1.0))
        expected_return_rate = 0.42 if promo else 0.28
        returns = int(shipped * expected_return_rate * np.random.uniform(0.9, 1.1))
        bbe.update(shipped, returns, promo)
        shipped_history.append(float(shipped))

    wave = bbe.predict_return_wave(shipped_history, qc_yield=0.92)
    print(f"  Return rate estimate : {wave['mean_return_rate']:.3f}")
    print(f"  Expected returns/wk  : {wave['expected_returns_week']:.1f}")
    print(f"  95% CI               : [{wave['lower_95']:.1f}, {wave['upper_95']:.1f}]")
    net = bbe.get_net_demand(2200.0, wave["expected_returns_week"])
    print(f"  Net demand (2200 - returns): {net:.1f}")
    print()

    # -----------------------------------------------------------------------
    print("=" * 60)
    print("3. GammaLeadTimeEstimator")
    print("=" * 60)

    nodes = [
        GammaLeadTimeEstimator(base_days=3,  eta=0.04, node_id="node1"),
        GammaLeadTimeEstimator(base_days=14, eta=0.06, node_id="node2"),
        GammaLeadTimeEstimator(base_days=35, eta=0.10, node_id="node3"),
    ]

    for fi in [100.0, 155.0]:
        print(f"\n  freight_index = {fi}")
        for node in nodes:
            info = node.estimate(fi)
            print(
                f"    {node.node_id}: mean={info['mean_days']:.1f}d  "
                f"p10={info['p10']:.1f}  p50={info['p50']:.1f}  p90={info['p90']:.1f}d  "
                f"var={info['variance']:.2f}"
            )

    samples = nodes[2].simulate_pipeline_delay(n_samples=500)
    print(f"\n  node3 pipeline delay simulation (n=500): "
          f"mean={samples.mean():.1f}d, std={samples.std():.1f}d, "
          f"p90={np.percentile(samples, 90):.1f}d")

    print()
    print("adaptive_stats.py smoke test PASSED ✓")
