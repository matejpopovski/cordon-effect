"""Wild cluster restricted bootstrap for one coefficient, with test inversion.

Refitting each draw would be far too slow for 9,999 draws across dozens of specifications. As in
Roodman et al. (2019, "Fast and wild"), the bootstrap scores are linear in the cluster weights
once a G-by-G matrix has been computed, so each draw is a matrix product, and the restricted
residuals are linear in the null value, so inverting the test needs no further projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .did import Fit, absorb

WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


@dataclass(frozen=True)
class WildResult:
    p_value: float
    ci_low: float
    ci_high: float


class WildBootstrap:
    """Prepares the score matrices for one coefficient of a fitted model."""

    def __init__(self, fit: Fit, name: str, draws: int, seed: int):
        self.fit = fit
        self.index = fit.names.index(name)
        self.beta = float(fit.beta[self.index])
        others = np.delete(np.arange(len(fit.names)), self.index)

        t, n = fit.shape
        self.t, self.n = t, n
        root = fit.sqrt_w
        z = fit.x[:, others] * root[:, None]
        self.q = np.linalg.qr(z)[0] if z.size else np.zeros((t * n, 0))

        x = self._annihilate_z(fit.x[:, self.index] * root)
        y = self._annihilate_z(fit.y * root)
        self.x_tilde = x
        self.xx = float(x @ x)
        residual = y - self.beta * x

        ids = fit.cluster_ids
        self.groups = np.unique(ids)
        self.index_of = [np.flatnonzero(ids == g) for g in self.groups]
        self.A_y = self._score_matrix(y)
        self.A_x = self._score_matrix(x)
        self.c = np.array([float(x[i] @ x[i]) for i in self.index_of])

        # The observed scores use the unrestricted residuals, as the bootstrap t-statistics do.
        scores = np.array([float(x[i] @ residual[i]) for i in self.index_of])
        self.se = np.sqrt(np.sum(scores**2)) / self.xx

        rng = np.random.default_rng(seed)
        self.v = rng.choice(WEBB, size=(len(self.groups), draws))

    def _annihilate_z(self, vector: np.ndarray) -> np.ndarray:
        return vector - self.q @ (self.q.T @ vector) if self.q.size else vector

    def _annihilate(self, vector: np.ndarray) -> np.ndarray:
        """Project out fixed effects and other regressors, in the square-root-weight metric."""
        root = self.fit.sqrt_w
        field = (vector / root).reshape(self.t, self.n, 1)
        return self._annihilate_z(absorb(field, self.fit.weights).ravel() * root)

    def _score_matrix(self, residual_like: np.ndarray) -> np.ndarray:
        size = len(self.groups)
        matrix = np.zeros((size, size))
        for h, members in enumerate(self.index_of):
            piece = np.zeros_like(residual_like)
            piece[members] = residual_like[members]
            projected = self._annihilate(piece)
            for g, group_members in enumerate(self.index_of):
                matrix[g, h] = self.x_tilde[group_members] @ projected[group_members]
        return matrix

    def p_value(self, null: float) -> float:
        t_observed = (self.beta - null) / self.se
        scores = (self.A_y - null * self.A_x) @ self.v
        delta = scores.sum(axis=0) / self.xx
        adjusted = scores - np.outer(self.c, delta)
        se_star = np.sqrt((adjusted**2).sum(axis=0)) / self.xx
        with np.errstate(divide="ignore", invalid="ignore"):
            t_star = delta / se_star
        valid = np.isfinite(t_star)
        return float(np.mean(np.abs(t_star[valid]) >= abs(t_observed)))

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        """Invert the test by bisection outward from the point estimate."""
        alpha = 1 - level
        scale = max(self.se, 1e-6)

        def bound(direction: int) -> float:
            inner = self.beta
            step = scale
            outer = self.beta + direction * step
            for _ in range(40):
                if self.p_value(outer) < alpha:
                    break
                inner = outer
                step *= 1.6
                outer = self.beta + direction * step
            else:
                return float(np.sign(direction) * np.inf)
            for _ in range(32):
                middle = 0.5 * (inner + outer)
                if self.p_value(middle) < alpha:
                    outer = middle
                else:
                    inner = middle
            return 0.5 * (inner + outer)

        return bound(-1), bound(1)

    def result(self, level: float = 0.95) -> WildResult:
        low, high = self.interval(level)
        return WildResult(self.p_value(0.0), low, high)


def wild_test(fit: Fit, name: str, draws: int, seed: int) -> WildResult:
    return WildBootstrap(fit, name, draws, seed).result()
