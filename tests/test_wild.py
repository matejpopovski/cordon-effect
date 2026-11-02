import numpy as np
import pytest

from cordon import did
from cordon.wild import WEBB, WildBootstrap, wild_test


def test_webb_weights_have_unit_variance_and_zero_mean():
    assert WEBB.mean() == pytest.approx(0)
    assert (WEBB**2).mean() == pytest.approx(1)


def test_p_value_is_one_at_the_point_estimate(panel_factory):
    p = panel_factory(effect=-0.05)
    fitted = did.fit_post(p, ["core"])
    boot = WildBootstrap(fitted, "core", 999, 1)
    assert boot.p_value(fitted.coef("core")) == pytest.approx(1.0)


def test_detects_a_large_effect(panel_factory):
    fitted = did.fit_post(panel_factory(effect=-0.2), ["core"])
    result = wild_test(fitted, "core", 1999, 1)
    assert result.p_value < 0.01
    assert result.ci_high < 0


def test_interval_inverts_the_test(panel_factory):
    fitted = did.fit_post(panel_factory(effect=-0.05), ["core"])
    boot = WildBootstrap(fitted, "core", 1999, 1)
    low, high = boot.interval()
    assert low < fitted.coef("core") < high
    assert boot.p_value(low) == pytest.approx(0.05, abs=0.01)
    assert boot.p_value(high) == pytest.approx(0.05, abs=0.01)


def test_rejection_rate_under_the_null_is_near_nominal(panel_factory):
    """With a moderate number of treated clusters the test should hold its size."""
    rejections = 0
    trials = 120
    for seed in range(trials):
        panel = panel_factory(
            effect=0.0,
            n_zones=60,
            n_treated=30,
            first_week=-26,
            last_week=25,
            seed=seed,
            noise=0.05,
            cluster_noise=0.05,
        )
        fitted = did.fit_post(panel, ["core"], seasonal=False)
        rejections += WildBootstrap(fitted, "core", 199, seed).p_value(0.0) < 0.05
    assert rejections / trials < 0.13


def test_tight_when_all_noise_is_idiosyncratic_and_wide_with_cluster_shocks(panel_factory):
    quiet = did.fit_post(panel_factory(effect=-0.05, cluster_noise=0.0), ["core"])
    noisy = did.fit_post(panel_factory(effect=-0.05, cluster_noise=0.06), ["core"])
    quiet_width = np.diff(WildBootstrap(quiet, "core", 999, 1).interval())
    noisy_width = np.diff(WildBootstrap(noisy, "core", 999, 1).interval())
    assert noisy_width > quiet_width
