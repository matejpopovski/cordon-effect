"""Synthetic control and synthetic difference-in-differences on one treated series.

Both solve a least-squares problem over the simplex. The shared solver is accelerated projected
gradient, which needs nothing beyond numpy, gives the same answer on every platform and is quick
enough to run once per donor for the placebo loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    cumulative = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cumulative / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - cumulative[rho] / (rho + 1), 0.0)


def simplex_least_squares(
    a: np.ndarray, b: np.ndarray, ridge: float = 0.0, iterations: int = 20_000, tol: float = 1e-10
) -> np.ndarray:
    """Minimise ||a w - b||^2 + ridge ||w||^2 over the simplex with FISTA and restarts."""
    k = a.shape[1]
    gram = a.T @ a + ridge * np.eye(k)
    target = a.T @ b
    lipschitz = np.linalg.eigvalsh(gram)[-1]
    w = np.full(k, 1 / k)
    z, momentum = w.copy(), 1.0
    for _ in range(iterations):
        step = project_simplex(z - (gram @ z - target) / lipschitz)
        if np.dot(z - step, step - w) > 0:
            z, momentum = w.copy(), 1.0
            continue
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        z = step + (momentum - 1) / next_momentum * (step - w)
        change = np.max(np.abs(step - w))
        w, momentum = step, next_momentum
        if change < tol:
            break
    return w


@dataclass(frozen=True)
class SynthResult:
    att: float
    gaps: np.ndarray
    weights: np.ndarray
    rmspe_pre: float
    rmspe_post: float


def synthetic_control(treated: np.ndarray, donors: np.ndarray, pre: np.ndarray) -> SynthResult:
    """Simplex weights with an intercept, so that only the donors' shape must match."""
    centre_y = treated[pre].mean()
    centre_x = donors[pre].mean(axis=0)
    weights = simplex_least_squares(donors[pre] - centre_x, treated[pre] - centre_y)
    synthetic = centre_y + (donors - centre_x) @ weights
    gaps = treated - synthetic
    return SynthResult(
        att=float(gaps[~pre].mean()),
        gaps=gaps,
        weights=weights,
        rmspe_pre=float(np.sqrt(np.mean(gaps[pre] ** 2))),
        rmspe_post=float(np.sqrt(np.mean(gaps[~pre] ** 2))),
    )


def synthetic_did(treated: np.ndarray, donors: np.ndarray, pre: np.ndarray) -> SynthResult:
    """Arkhangelsky et al. (2021) with one treated unit and their default regularisation."""
    t_post = int((~pre).sum())
    sigma = np.std(np.diff(donors[pre], axis=0))
    zeta = t_post**0.25 * sigma

    centre_y = treated[pre].mean()
    centre_x = donors[pre].mean(axis=0)
    unit = simplex_least_squares(
        donors[pre] - centre_x, treated[pre] - centre_y, ridge=zeta**2 * int(pre.sum())
    )

    post_mean = donors[~pre].mean(axis=0)
    a = (donors[pre] - donors[pre].mean(axis=1, keepdims=True)).T
    b = post_mean - post_mean.mean()
    time = simplex_least_squares(a, b, ridge=1e-6 * sigma**2)

    gap_series = treated - donors @ unit
    att = gap_series[~pre].mean() - gap_series[pre] @ time
    baseline = gap_series[pre].mean()
    gaps = gap_series - baseline
    return SynthResult(
        att=float(att),
        gaps=gaps,
        weights=unit,
        rmspe_pre=float(np.sqrt(np.mean(gaps[pre] ** 2))),
        rmspe_post=float(np.sqrt(np.mean(gaps[~pre] ** 2))),
    )


@dataclass(frozen=True)
class PlaceboResult:
    atts: np.ndarray
    ratios: np.ndarray
    gaps: np.ndarray
    ratio_p: float
    att_p: float
    sd: float


def in_space_placebo(
    method, treated: np.ndarray, donors: np.ndarray, pre: np.ndarray, fitted: SynthResult
) -> PlaceboResult:
    """Give each donor a turn as treated, using the rest as its donor pool."""
    n = donors.shape[1]
    atts, ratios, gaps = [], [], []
    for i in range(n):
        others = np.delete(np.arange(n), i)
        placebo = method(donors[:, i], donors[:, others], pre)
        atts.append(placebo.att)
        ratios.append(placebo.rmspe_post / placebo.rmspe_pre)
        gaps.append(placebo.gaps)
    atts_array = np.array(atts)
    ratio_array = np.array(ratios)
    observed_ratio = fitted.rmspe_post / fitted.rmspe_pre
    return PlaceboResult(
        atts=atts_array,
        ratios=ratio_array,
        gaps=np.array(gaps),
        ratio_p=float((1 + (ratio_array >= observed_ratio).sum()) / (n + 1)),
        att_p=float((1 + (np.abs(atts_array) >= abs(fitted.att)).sum()) / (n + 1)),
        sd=float(np.sqrt(np.mean(atts_array**2))),
    )
