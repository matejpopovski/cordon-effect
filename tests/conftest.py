from __future__ import annotations

import numpy as np
import pytest

from cordon.did import Panel


def make_panel(
    effect: float = 0.0,
    n_zones: int = 60,
    n_treated: int = 12,
    zones_per_cluster: int = 3,
    first_week: int = -104,
    last_week: int = 51,
    noise: float = 0.03,
    cluster_noise: float = 0.02,
    seasonal_gap: float = 0.0,
    seed: int = 0,
) -> Panel:
    """A panel with known truth: zone and week effects, cluster shocks, and one treated block."""
    rng = np.random.default_rng(seed)
    weeks = np.arange(first_week, last_week + 1)
    t = len(weeks)
    treated = np.arange(n_zones) < n_treated
    clusters = np.array([f"c{i // zones_per_cluster}" for i in range(n_zones)])
    groups = np.where(treated, "core", "far")

    zone_level = rng.normal(6.0, 0.5, n_zones)
    week_level = rng.normal(0, 0.1, t)
    season = np.sin(2 * np.pi * np.mod(weeks, 52) / 52)
    labels = np.unique(clusters, return_inverse=True)[1]
    shocks = rng.normal(0, cluster_noise, (t, labels.max() + 1))[:, labels]
    log_y = (
        zone_level
        + week_level[:, None]
        + shocks
        + rng.normal(0, noise, (t, n_zones))
        + np.outer(season * seasonal_gap, treated.astype(float))
        + np.outer(weeks >= 0, treated.astype(float)) * effect
    )
    counts = np.exp(log_y)
    return Panel(
        weeks=weeks,
        zones=np.arange(1, n_zones + 1),
        groups=groups,
        clusters=clusters,
        counts=counts,
        weights=counts[weeks < 0].mean(axis=0),
    )


@pytest.fixture
def panel_factory():
    return make_panel
