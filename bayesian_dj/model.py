from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import expit as sigmoid

from .song_pool import AUDIO_FEATURES, LOUDNESS_MAX, LOUDNESS_MIN, TEMPO_MAX, TEMPO_MIN

N_FEATURES = len(AUDIO_FEATURES) + 1  # 9 audio features + bias

FEATURE_INDEX = {name: i + 1 for i, name in enumerate(AUDIO_FEATURES)}


@dataclass
class PosteriorSnapshot:
    """Records the full posterior state after a single Bayesian update."""

    step: int
    mu: np.ndarray
    sigma_diag: np.ndarray
    log_det_sigma: float
    entropy: float
    x: np.ndarray | None = None
    y: int | None = None
    pred_map: float | None = None
    pred_posterior: float | None = None

CONSTRAINT_KEY_MAP = {
    "danceability": "danceability",
    "energy": "energy",
    "loudness": "loudness",
    "speechiness": "speechiness",
    "acousticness": "acousticness",
    "instrumentalness": "instrumentalness",
    "liveness": "liveness",
    "valence": "valence",
    "tempo_bpm": "tempo",
}


@dataclass
class PosteriorSummary:
    mean: np.ndarray
    feature_weights: dict[str, float]
    top_positive: list[tuple[str, float]]
    top_negative: list[tuple[str, float]]


class BayesianLogisticRegression:
    """Online Bayesian logistic regression with Laplace-approximation updates
    and Thompson-sampling song selection."""

    def __init__(
        self,
        prior_mean: np.ndarray | None = None,
        prior_cov: np.ndarray | None = None,
    ) -> None:
        self.mu = (
            prior_mean.copy()
            if prior_mean is not None
            else np.zeros(N_FEATURES, dtype=np.float64)
        )
        self.sigma = (
            prior_cov.copy()
            if prior_cov is not None
            else np.eye(N_FEATURES, dtype=np.float64)
        )
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.n_updates = 0
        self.history: list[PosteriorSnapshot] = []

    @staticmethod
    def _normalize_constraint_target(key: str, target: float) -> float:
        if key == "tempo_bpm":
            return float(np.clip((target - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN), 0.0, 1.0))
        if key == "loudness":
            # Parser constraints for loudness may be specified in raw dB.
            if target < 0.0 or target > 1.0:
                return float(np.clip((target - LOUDNESS_MIN) / (LOUDNESS_MAX - LOUDNESS_MIN), 0.0, 1.0))
        return float(np.clip(target, 0.0, 1.0))

    @staticmethod
    def from_constraints(
        constraints: dict[str, tuple[float, float]],
        scale: float = 2.0,
        constrained_var: float = 0.5,
        unconstrained_var: float = 2.0,
    ) -> BayesianLogisticRegression:
        """Build a prior from parsed QuerySpec constraints.

        For each constrained feature the prior mean is set so that the model
        prefers songs whose feature value is near the constraint midpoint:
            mu[f] = (2 * target - 1) * scale

        Constrained features get tighter variance; unconstrained features
        stay wide to allow the feedback loop to learn them.
        """
        mu = np.zeros(N_FEATURES, dtype=np.float64)
        var = np.full(N_FEATURES, unconstrained_var, dtype=np.float64)
        var[0] = 1.0  # bias variance

        for key, (lo, hi) in constraints.items():
            feat_name = CONSTRAINT_KEY_MAP.get(key)
            if feat_name is None or feat_name not in FEATURE_INDEX:
                continue
            idx = FEATURE_INDEX[feat_name]
            target = (lo + hi) / 2.0
            target = BayesianLogisticRegression._normalize_constraint_target(key, target)
            mu[idx] = (2.0 * target - 1.0) * scale
            var[idx] = constrained_var

        cov = np.diag(var)
        return BayesianLogisticRegression(prior_mean=mu, prior_cov=cov)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(like | x) using the posterior mean."""
        return sigmoid(X @ self.mu)

    def thompson_sample_scores(self, X: np.ndarray) -> np.ndarray:
        """Draw beta ~ N(mu, Sigma), return sigmoid(X @ beta)."""
        beta_sample = np.random.multivariate_normal(self.mu, self.sigma)
        return sigmoid(X @ beta_sample)

    def predict_proba_posterior(self, X: np.ndarray) -> np.ndarray:
        """Bayesian predictive probability integrating over posterior uncertainty.

        Uses the probit approximation:  since beta ~ N(mu, Sigma),
        beta^T x ~ N(mu^T x,  x^T Sigma x).  Then
        E[sigmoid(beta^T x)] ~ sigmoid(mu^T x / sqrt(1 + pi/8 * x^T Sigma x)).

        Returns calibrated probabilities that are naturally conservative when
        uncertainty is high and converge to the MAP estimate as the posterior
        tightens.
        """
        logits = X @ self.mu
        var = np.einsum("ij,jk,ik->i", X, self.sigma, X)
        kappa = 1.0 / np.sqrt(1.0 + np.pi / 8.0 * var)
        return sigmoid(kappa * logits)

    def posterior_entropy(self) -> float:
        """Differential entropy of the multivariate Gaussian posterior.

        H = 0.5 * (k * ln(2 * pi * e) + ln|Sigma|)
        """
        sign, logdet = np.linalg.slogdet(self.sigma)
        return 0.5 * (N_FEATURES * np.log(2.0 * np.pi * np.e) + logdet)

    def snapshot(
        self, x: np.ndarray | None = None, y: int | None = None
    ) -> PosteriorSnapshot:
        """Record the current posterior state and append to history."""
        sign, logdet = np.linalg.slogdet(self.sigma)

        pred_map = None
        pred_post = None
        if x is not None:
            x2d = x.reshape(1, -1)
            pred_map = float(self.predict_proba(x2d)[0])
            pred_post = float(self.predict_proba_posterior(x2d)[0])

        snap = PosteriorSnapshot(
            step=self.n_updates,
            mu=self.mu.copy(),
            sigma_diag=np.diag(self.sigma).copy(),
            log_det_sigma=float(logdet),
            entropy=self.posterior_entropy(),
            x=x.copy() if x is not None else None,
            y=y,
            pred_map=pred_map,
            pred_posterior=pred_post,
        )
        self.history.append(snap)
        return snap

    def update(self, x: np.ndarray, y: int) -> None:
        """Single-observation Laplace-approximation update.

        x : (N_FEATURES,) feature vector (with leading bias=1)
        y : 1 for play, 0 for skip
        """
        p = float(sigmoid(self.mu @ x))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        w = p * (1.0 - p)

        self.sigma_inv = self.sigma_inv + w * np.outer(x, x)
        self.sigma = np.linalg.inv(self.sigma_inv)
        self.mu = self.mu + self.sigma @ (x * (y - p))
        self.n_updates += 1

    def get_summary(self) -> PosteriorSummary:
        weights = {}
        for name, idx in FEATURE_INDEX.items():
            weights[name] = float(self.mu[idx])

        sorted_feats = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
        top_pos = [(n, v) for n, v in sorted_feats if v > 0][:3]
        top_neg = [(n, v) for n, v in sorted_feats if v < 0][-3:]

        return PosteriorSummary(
            mean=self.mu.copy(),
            feature_weights=weights,
            top_positive=top_pos,
            top_negative=top_neg,
        )
