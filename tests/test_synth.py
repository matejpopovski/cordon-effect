import numpy as np
import pytest
from scipy.optimize import minimize

from cordon import synth


def test_projection_lands_on_the_simplex():
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = synth.project_simplex(rng.normal(size=7))
        assert w.min() >= 0
        assert w.sum() == pytest.approx(1)


def test_projection_leaves_simplex_points_unchanged():
    point = np.array([0.2, 0.3, 0.5])
    assert synth.project_simplex(point) == pytest.approx(point)


def test_solver_agrees_with_a_general_constrained_optimiser():
    rng = np.random.default_rng(1)
    a = rng.normal(size=(40, 6))
    b = rng.normal(size=40)
    ours = synth.simplex_least_squares(a, b)
    reference = minimize(
        lambda w: np.sum((a @ w - b) ** 2),
        np.full(6, 1 / 6),
        bounds=[(0, 1)] * 6,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        tol=1e-12,
    )
    assert np.sum((a @ ours - b) ** 2) <= reference.fun + 1e-8


def donor_world(effect: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = 160
    pre = np.arange(t) < 110
    factors = np.column_stack([np.sin(np.arange(t) / 8), np.cumsum(rng.normal(0, 0.05, t))])
    loadings = rng.uniform(0, 1.5, (20, 2))
    donors = 7 + factors @ loadings.T + rng.normal(0, 0.01, (t, 20))
    truth = np.zeros(20)
    truth[[2, 5, 11]] = [0.5, 0.3, 0.2]
    treated = donors @ truth + rng.normal(0, 0.005, t) + 0.3
    treated[~pre] += effect
    return treated, donors, pre


def test_synthetic_control_recovers_effect_with_an_intercept_gap():
    treated, donors, pre = donor_world(-0.1)
    result = synth.synthetic_control(treated, donors, pre)
    assert result.att == pytest.approx(-0.1, abs=0.01)
    assert result.weights.sum() == pytest.approx(1)


def test_synthetic_did_recovers_effect():
    treated, donors, pre = donor_world(-0.1)
    assert synth.synthetic_did(treated, donors, pre).att == pytest.approx(-0.1, abs=0.02)


def test_placebo_rank_flags_a_real_effect_and_not_a_null():
    treated, donors, pre = donor_world(-0.3)
    fitted = synth.synthetic_control(treated, donors, pre)
    placebo = synth.in_space_placebo(synth.synthetic_control, treated, donors, pre, fitted)
    assert placebo.ratio_p <= 2 / 21

    treated, donors, pre = donor_world(0.0)
    fitted = synth.synthetic_control(treated, donors, pre)
    placebo = synth.in_space_placebo(synth.synthetic_control, treated, donors, pre, fitted)
    assert placebo.att_p > 0.1
