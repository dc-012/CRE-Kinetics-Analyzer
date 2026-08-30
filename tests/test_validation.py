"""Tests for core/validation.py"""

import numpy as np
import pytest

from core.validation import validate_concentration_time_data, DataValidationError


def test_valid_data_passes():
    t = [0, 20, 40, 60, 120, 180, 300]
    c = [10, 8, 6, 5, 3, 2, 1]
    result = validate_concentration_time_data(t, c)
    assert np.allclose(result.time, t)
    assert np.allclose(result.concentration, c)
    assert result.warnings == []


def test_mismatched_lengths_raises():
    with pytest.raises(DataValidationError, match="same length"):
        validate_concentration_time_data([0, 10, 20], [10, 8])


def test_too_few_points_raises():
    with pytest.raises(DataValidationError, match="At least"):
        validate_concentration_time_data([0, 10, 20], [10, 8, 6])


def test_nan_time_raises():
    with pytest.raises(DataValidationError, match="NaN or infinite"):
        validate_concentration_time_data([0, 10, float("nan"), 30], [10, 8, 6, 2])


def test_nan_concentration_raises():
    with pytest.raises(DataValidationError, match="NaN or infinite"):
        validate_concentration_time_data([0, 10, 20, 30], [10, 8, float("nan"), 2])


def test_zero_concentration_raises():
    with pytest.raises(DataValidationError, match="strictly positive"):
        validate_concentration_time_data([0, 10, 20, 30], [10, 8, 0, 2])


def test_negative_concentration_raises():
    with pytest.raises(DataValidationError, match="strictly positive"):
        validate_concentration_time_data([0, 10, 20, 30], [10, 8, -1, 2])


def test_non_increasing_time_raises():
    with pytest.raises(DataValidationError, match="strictly increasing"):
        validate_concentration_time_data([0, 10, 10, 30], [10, 8, 6, 2])


def test_decreasing_time_raises():
    with pytest.raises(DataValidationError, match="strictly increasing"):
        validate_concentration_time_data([0, 20, 10, 30], [10, 8, 6, 2])


def test_time_not_starting_at_zero_raises():
    with pytest.raises(DataValidationError, match="must be 0"):
        validate_concentration_time_data([5, 10, 20, 30], [10, 8, 6, 2])


def test_net_increase_in_concentration_raises():
    with pytest.raises(DataValidationError, match="net decrease"):
        validate_concentration_time_data([0, 10, 20, 30], [2, 4, 6, 8])


def test_equal_start_end_concentration_raises():
    # No net conversion at all -> nothing to analyze
    with pytest.raises(DataValidationError, match="net decrease"):
        validate_concentration_time_data([0, 10, 20, 30], [5, 4, 6, 5])


def test_none_inputs_raise():
    with pytest.raises(DataValidationError):
        validate_concentration_time_data(None, [1, 2, 3, 4])
    with pytest.raises(DataValidationError):
        validate_concentration_time_data([0, 1, 2, 3], None)


def test_non_numeric_inputs_raise():
    with pytest.raises(DataValidationError):
        validate_concentration_time_data([0, 1, 2, "x"], [10, 8, 6, 2])


def test_local_noise_produces_warning_not_error():
    # Overall decreasing trend, but one local uptick between samples
    t = [0, 10, 20, 30, 40, 50]
    c = [10, 8, 8.5, 6, 4, 2]
    result = validate_concentration_time_data(t, c)
    assert len(result.warnings) >= 1
    assert any("increases locally" in w for w in result.warnings)


def test_small_sample_size_warning():
    t = [0, 10, 20, 30]
    c = [10, 7, 5, 3]
    result = validate_concentration_time_data(t, c)
    assert any("data points supplied" in w for w in result.warnings)


def test_excessive_noise_triggers_strong_warning():
    t = [0, 10, 20, 30, 40, 50, 60]
    c = [10, 11, 9, 10, 7, 8, 5]  # >40% of intervals increase
    result = validate_concentration_time_data(t, c)
    assert any("more than 40%" in w for w in result.warnings)
