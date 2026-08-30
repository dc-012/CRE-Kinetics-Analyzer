"""
Tests for core/kinetics.py

These check the pure math (integrated law, rate, half-life,
conversion, time-for-complete-conversion) against values computed
independently by hand / from closed-form textbook relations, NOT
against the fitting engine.
"""

import numpy as np
import pytest

from core import kinetics


# ---------------------------------------------------------------------
# integrated_concentration
# ---------------------------------------------------------------------

def test_first_order_matches_exponential_decay():
    CA0, k = 10.0, 0.05
    t = np.array([0, 10, 20, 50, 100])
    expected = CA0 * np.exp(-k * t)
    actual = kinetics.integrated_concentration(t, CA0, k, n=1.0)
    assert np.allclose(actual, expected)


def test_zero_order_is_linear_decay():
    # -dCA/dt = k  =>  CA = CA0 - k*t
    CA0, k = 10.0, 0.1
    t = np.array([0, 10, 20, 30])
    expected = CA0 - k * t
    actual = kinetics.integrated_concentration(t, CA0, k, n=0.0)
    assert np.allclose(actual, expected)


def test_second_order_matches_1_over_CA_linear_form():
    # 1/CA = 1/CA0 + k*t  =>  CA = 1/(1/CA0 + k*t)
    CA0, k = 10.0, 0.02
    t = np.array([0, 10, 20, 50])
    expected = 1.0 / (1.0 / CA0 + k * t)
    actual = kinetics.integrated_concentration(t, CA0, k, n=2.0)
    assert np.allclose(actual, expected)


def test_integrated_concentration_at_t0_equals_CA0():
    for n in [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        CA0 = 7.5
        CA_t0 = kinetics.integrated_concentration(0.0, CA0, k=0.03, n=n)
        assert np.isclose(CA_t0, CA0), f"failed for n={n}"


def test_integrated_concentration_monotonically_decreasing():
    CA0, k, n = 10.0, 0.01, 1.7
    t = np.linspace(0, 50, 20)
    CA = kinetics.integrated_concentration(t, CA0, k, n)
    assert np.all(np.diff(CA) <= 1e-12)


def test_nth_order_clips_to_zero_past_complete_conversion_for_n_lt_1():
    # n < 1 reaches exact zero at a finite time; beyond that the model
    # must not go negative or complex.
    CA0, k, n = 10.0, 0.5, 0.5
    t_complete = kinetics.time_for_complete_conversion(CA0, k, n)
    CA_after = kinetics.integrated_concentration(t_complete * 2, CA0, k, n)
    assert CA_after == 0.0


def test_order_near_one_matches_true_first_order_continuously():
    # sanity check there's no discontinuity around the n=1 special case
    CA0, k = 10.0, 0.02
    t = 30.0
    CA_below = kinetics.integrated_concentration(t, CA0, k, n=1.0 - 1e-4)
    CA_at = kinetics.integrated_concentration(t, CA0, k, n=1.0)
    CA_above = kinetics.integrated_concentration(t, CA0, k, n=1.0 + 1e-4)
    assert np.isclose(CA_below, CA_at, atol=1e-3)
    assert np.isclose(CA_above, CA_at, atol=1e-3)


# ---------------------------------------------------------------------
# rate_of_reaction
# ---------------------------------------------------------------------

def test_rate_of_reaction_basic():
    k, n = 0.05, 2.0
    CA = 4.0
    assert np.isclose(kinetics.rate_of_reaction(CA, k, n), 0.05 * 4.0 ** 2)


def test_rate_of_reaction_array():
    k, n = 0.1, 1.0
    CA = np.array([1.0, 2.0, 4.0])
    expected = k * CA
    assert np.allclose(kinetics.rate_of_reaction(CA, k, n), expected)


def test_rate_zero_order_is_constant():
    k, n = 0.2, 0.0
    for CA in [1.0, 5.0, 100.0]:
        assert np.isclose(kinetics.rate_of_reaction(CA, k, n), k)


# ---------------------------------------------------------------------
# half_life
# ---------------------------------------------------------------------

def test_half_life_first_order():
    k = 0.05
    expected = np.log(2) / k
    assert np.isclose(kinetics.half_life(CA0=10.0, k=k, n=1.0), expected)


def test_half_life_zero_order():
    # For zero order: CA0 - k*t_half = CA0/2  =>  t_half = CA0/(2k)
    CA0, k = 10.0, 0.1
    expected = CA0 / (2 * k)
    assert np.isclose(kinetics.half_life(CA0, k, n=0.0), expected)


def test_half_life_second_order():
    # For second order: 1/(CA0/2) - 1/CA0 = k*t_half => t_half = 1/(k*CA0)
    CA0, k = 8.0, 0.02
    expected = 1.0 / (k * CA0)
    assert np.isclose(kinetics.half_life(CA0, k, n=2.0), expected)


def test_half_life_consistent_with_integrated_law():
    # General cross-check: plugging t_half back into the integrated
    # law should give exactly CA0/2, for a range of orders.
    CA0, k = 10.0, 0.03
    for n in [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        t_half = kinetics.half_life(CA0, k, n)
        CA_at_half = kinetics.integrated_concentration(t_half, CA0, k, n)
        assert np.isclose(CA_at_half, CA0 / 2, rtol=1e-4), f"failed for n={n}"


def test_half_life_continuous_across_n_equals_1():
    CA0, k = 10.0, 0.02
    t_below = kinetics.half_life(CA0, k, n=1.0 - 1e-4)
    t_at = kinetics.half_life(CA0, k, n=1.0)
    t_above = kinetics.half_life(CA0, k, n=1.0 + 1e-4)
    assert np.isclose(t_below, t_at, rtol=1e-3)
    assert np.isclose(t_above, t_at, rtol=1e-3)


# ---------------------------------------------------------------------
# time_for_complete_conversion
# ---------------------------------------------------------------------

def test_time_for_complete_conversion_n_less_than_1():
    # Matches "Eq 29" confirmed in the uploaded solutions manual,
    # problem 3.19: t = CA0^(1-n) / [(1-n)*k]
    CA0, k, n = 10.0, 3.0, 0.5
    expected = CA0 ** (1 - n) / ((1 - n) * k)
    assert np.isclose(kinetics.time_for_complete_conversion(CA0, k, n), expected)


def test_time_for_complete_conversion_none_for_n_gte_1():
    assert kinetics.time_for_complete_conversion(10.0, 0.1, n=1.0) is None
    assert kinetics.time_for_complete_conversion(10.0, 0.1, n=2.0) is None
    assert kinetics.time_for_complete_conversion(10.0, 0.1, n=1.5) is None


def test_time_for_complete_conversion_none_for_n_extremely_close_to_1():
    # Regression test: a nonlinear fit can return n that is extremely
    # close to but not exactly 1.0 (e.g. 0.99999997). Without using
    # the same is_first_order() tolerance as the rest of this module,
    # this used to return a huge, physically meaningless finite time
    # instead of None.
    assert kinetics.time_for_complete_conversion(10.0, 0.1, n=0.99999997) is None
    assert kinetics.time_for_complete_conversion(10.0, 0.1, n=1.00000003) is None


# ---------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------

def test_conversion_basic():
    CA0 = 10.0
    CA = np.array([10.0, 7.5, 5.0, 0.0])
    expected = np.array([0.0, 0.25, 0.5, 1.0])
    assert np.allclose(kinetics.conversion(CA0, CA), expected)


def test_conversion_scalar():
    assert np.isclose(kinetics.conversion(10.0, 5.0), 0.5)
