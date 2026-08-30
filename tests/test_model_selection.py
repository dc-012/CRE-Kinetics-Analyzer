"""
Tests for core/model_selection.py

These are the most important accuracy tests in the whole suite: they
check that, given clean (or realistically noisy) concentration-time
data generated from a KNOWN order and rate constant, the fitting
procedure actually recovers those known values within tolerance -
not just that it runs without crashing.
"""

import numpy as np
import pytest

from core.model_selection import determine_best_order


# Reasonably dense, wide time grids so the fit has enough information
# to distinguish nearby orders.
T_STANDARD = np.array([0, 5, 10, 20, 30, 45, 60, 90, 120], dtype=float)


@pytest.mark.parametrize("n_true,k_true,CA0", [
    (0.0, 0.05, 10.0),
    (1.0, 0.02, 10.0),
    (2.0, 0.01, 10.0),
])
def test_recovers_classic_integer_orders(make_synthetic_data, n_true, k_true, CA0):
    t, CA = make_synthetic_data(CA0, k_true, n_true, T_STANDARD)
    result = determine_best_order(t, CA)
    assert np.isclose(result.best_n, n_true, atol=0.05)
    assert np.isclose(result.best_k, k_true, rtol=0.02)
    assert result.r2 > 0.999


def test_recovers_higher_order(make_synthetic_data):
    # order 4 - "a higher-order case" per the project brief
    n_true, k_true, CA0 = 4.0, 0.0005, 5.0
    t, CA = make_synthetic_data(CA0, k_true, n_true, T_STANDARD)
    result = determine_best_order(t, CA)
    assert np.isclose(result.best_n, n_true, atol=0.1)
    assert np.isclose(result.best_k, k_true, rtol=0.05)
    assert result.r2 > 0.999


@pytest.mark.parametrize("n_true", [0.5, 1.5, 2.3, 2.7])
def test_recovers_non_integer_orders(make_synthetic_data, n_true):
    CA0, k_true = 8.0, 0.02
    t, CA = make_synthetic_data(CA0, k_true, n_true, T_STANDARD)
    result = determine_best_order(t, CA)
    assert np.isclose(result.best_n, n_true, atol=0.05)
    assert np.isclose(result.best_k, k_true, rtol=0.03)
    assert result.r2 > 0.999


def test_recovers_first_order_with_realistic_noise(make_synthetic_data):
    np.random.seed(0)
    CA0, k_true, n_true = 10.0, 0.02, 1.0
    t, CA_clean = make_synthetic_data(CA0, k_true, n_true, T_STANDARD)

    noise = np.random.normal(0, 0.03 * CA_clean, size=CA_clean.shape)
    CA_noisy = np.clip(CA_clean + noise, 1e-6, None)
    CA_noisy[0] = CA0  # t=0 measurement kept exact, as validation requires

    result = determine_best_order(t, CA_noisy)
    # With ~3% noise we expect the order within about 0.2 of truth.
    # k and n are correlated in a nonlinear fit (a slightly-off order
    # is compensated by a slightly-off k to still match the data), so
    # k's tolerance is intentionally looser than in the noise-free
    # tests above.
    assert np.isclose(result.best_n, n_true, atol=0.2)
    assert np.isclose(result.best_k, k_true, rtol=0.3)
    assert result.r2 > 0.95


def test_recovers_second_order_with_realistic_noise(make_synthetic_data):
    np.random.seed(1)
    CA0, k_true, n_true = 10.0, 0.015, 2.0
    t, CA_clean = make_synthetic_data(CA0, k_true, n_true, T_STANDARD)

    noise = np.random.normal(0, 0.03 * CA_clean, size=CA_clean.shape)
    CA_noisy = np.clip(CA_clean + noise, 1e-6, None)
    CA_noisy[0] = CA0

    result = determine_best_order(t, CA_noisy)
    assert np.isclose(result.best_n, n_true, atol=0.2)
    assert np.isclose(result.best_k, k_true, rtol=0.3)
    assert result.r2 > 0.95


def test_candidates_scanned_is_populated(make_synthetic_data):
    t, CA = make_synthetic_data(10.0, 0.02, 1.0, T_STANDARD)
    result = determine_best_order(t, CA)
    assert len(result.candidates_scanned) > 10
    # every scanned candidate should carry a finite n
    assert all(np.isfinite(c.n) for c in result.candidates_scanned)


def test_best_order_is_the_best_scoring_reasonable_candidate(make_synthetic_data):
    # The chosen order should not be worse (in concentration-space R^2)
    # than the best of the coarse grid candidates.
    t, CA = make_synthetic_data(10.0, 0.02, 1.0, T_STANDARD)
    result = determine_best_order(t, CA)
    best_grid_r2 = max(c.r2_concentration_space for c in result.candidates_scanned)
    assert result.r2 >= best_grid_r2 - 1e-9
