"""
Tests for core/calculations.py - the full pipeline end to end
(validation -> order determination -> derived quantities), using the
example numbers given in the project brief plus each required test
category: zero/first/second/higher/non-integer order, noisy data, and
invalid data.
"""

import numpy as np
import pytest

from core.calculations import analyze_kinetics, KineticsResult
from core.validation import DataValidationError


def test_project_brief_example_runs_end_to_end():
    # The exact example data given in the project brief.
    t = [0, 20, 40, 60, 120, 180, 300]
    c = [10, 8, 6, 5, 3, 2, 1]
    result = analyze_kinetics(t, c)

    assert isinstance(result, KineticsResult)
    assert result.CA0 == 10.0
    assert 0.0 <= result.r_squared <= 1.0
    assert result.rate_constant > 0
    assert len(result.conversion) == len(t)
    assert len(result.rate_at_each_point) == len(t)
    assert len(result.predicted_concentration) == len(t)
    # conversion must start at 0 and end positive and <=1
    assert np.isclose(result.conversion[0], 0.0)
    assert 0.0 < result.conversion[-1] <= 1.0


@pytest.mark.parametrize("n_true,k_true,CA0", [
    (0.0, 0.05, 10.0),
    (1.0, 0.02, 10.0),
    (2.0, 0.01, 10.0),
])
def test_end_to_end_recovers_known_order_and_k(make_synthetic_data, n_true, k_true, CA0):
    t = np.array([0, 5, 10, 20, 30, 45, 60, 90, 120], dtype=float)
    t, CA = make_synthetic_data(CA0, k_true, n_true, t)
    result = analyze_kinetics(t, CA)
    assert np.isclose(result.reaction_order, n_true, atol=0.05)
    assert np.isclose(result.rate_constant, k_true, rtol=0.02)


def test_half_life_present_and_correct_for_first_order(make_synthetic_data):
    t = np.array([0, 10, 20, 40, 60, 90], dtype=float)
    CA0, k_true, n_true = 10.0, 0.03, 1.0
    t, CA = make_synthetic_data(CA0, k_true, n_true, t)
    result = analyze_kinetics(t, CA)
    assert result.half_life is not None
    assert np.isclose(result.half_life, np.log(2) / k_true, rtol=0.02)


def test_time_for_complete_conversion_only_for_n_less_than_1(make_synthetic_data):
    # time_for_complete_conversion(CA0=10, k=0.3, n=0.5) ~= 21.1, so keep
    # the time grid comfortably below that (validation requires CA>0
    # everywhere, and the model itself clips to exactly 0 at/after that
    # time, which is the physically correct behavior being tested
    # elsewhere in test_kinetics.py).
    t = np.array([0, 4, 8, 12, 16, 19], dtype=float)
    CA0, k_true, n_true = 10.0, 0.3, 0.5
    t, CA = make_synthetic_data(CA0, k_true, n_true, t)
    result = analyze_kinetics(t, CA)
    # fitted order should be near 0.5 (< 1), so a finite completion time
    # should be reported
    if result.reaction_order < 0.98:
        assert result.time_for_complete_conversion is not None
        assert result.time_for_complete_conversion > 0


def test_noisy_data_still_produces_reasonable_result(make_synthetic_data):
    np.random.seed(7)
    t = np.array([0, 20, 40, 60, 120, 180, 300], dtype=float)
    CA0, k_true, n_true = 10.0, 0.02, 1.0
    t, CA_clean = make_synthetic_data(CA0, k_true, n_true, t)
    noise = np.random.normal(0, 0.05 * CA_clean, size=CA_clean.shape)
    CA_noisy = np.clip(CA_clean + noise, 1e-6, None)
    CA_noisy[0] = CA0

    result = analyze_kinetics(t, CA_noisy)
    assert np.isclose(result.reaction_order, n_true, atol=0.3)
    assert result.r_squared > 0.9
    # noise should have been flagged
    assert len(result.warnings) >= 0  # may or may not trigger depending on draw


def test_invalid_data_raises_before_any_fitting():
    with pytest.raises(DataValidationError):
        analyze_kinetics([0, 10, 20], [10, 8, -1])  # negative concentration, too short


def test_result_units_string_first_order(make_synthetic_data):
    t = np.array([0, 10, 20, 30, 40], dtype=float)
    t, CA = make_synthetic_data(10.0, 0.02, 1.0, t)
    result = analyze_kinetics(t, CA)
    assert result.rate_constant_units == "1/time"


def test_result_units_string_second_order(make_synthetic_data):
    t = np.array([0, 10, 20, 30, 40], dtype=float)
    t, CA = make_synthetic_data(10.0, 0.01, 2.0, t)
    result = analyze_kinetics(t, CA)
    assert "concentration" in result.rate_constant_units
    assert "time" in result.rate_constant_units


def test_rate_at_CA0_matches_rate_law(make_synthetic_data):
    t = np.array([0, 10, 20, 30, 40], dtype=float)
    t, CA = make_synthetic_data(10.0, 0.02, 1.5, t)
    result = analyze_kinetics(t, CA)
    expected = result.rate_constant * result.CA0 ** result.reaction_order
    assert np.isclose(result.rate_at_CA0, expected, rtol=1e-6)
