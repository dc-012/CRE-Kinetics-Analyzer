"""
Tests for app/plot_logic.py.

These only exercise the pure, framework-agnostic helper functions used
to prepare graph data - no Plotly or Streamlit import is needed to run
these, matching the separation described in app/__init__.py.
"""

import numpy as np
import pytest

from app import plot_logic
from core import kinetics


def test_dense_time_grid_spans_zero_to_tmax():
    grid = plot_logic.dense_time_grid(t_max=50.0, n_points=100)
    assert grid[0] == 0.0
    assert grid[-1] == 50.0
    assert len(grid) == 100


def test_dense_time_grid_handles_degenerate_tmax():
    # t_max <= 0 should not crash or produce a zero-width grid
    grid = plot_logic.dense_time_grid(t_max=0.0, n_points=10)
    assert grid[-1] > grid[0]


@pytest.mark.parametrize("n_true,k_true,CA0", [
    (1.0, 0.05, 10.0),
    (2.0, 0.02, 8.0),
    (0.5, 0.1, 5.0),
])
def test_fitted_concentration_curve_matches_core_kinetics(n_true, k_true, CA0):
    """The dense fitted curve must be IDENTICAL to calling
    core.kinetics.integrated_concentration directly - plot_logic must
    not reimplement or alter the model."""
    t_dense, CA_dense = plot_logic.fitted_concentration_curve(
        CA0=CA0, k=k_true, n=n_true, t_max=100.0, n_points=50
    )
    expected = kinetics.integrated_concentration(t_dense, CA0, k_true, n_true)
    assert np.allclose(CA_dense, expected)


def test_fitted_concentration_curve_is_monotonically_decreasing():
    t_dense, CA_dense = plot_logic.fitted_concentration_curve(
        CA0=10.0, k=0.05, n=1.0, t_max=100.0
    )
    assert np.all(np.diff(CA_dense) <= 1e-9)


def test_linearized_transform_first_order_is_log():
    CA = np.array([10.0, 5.0, 2.5])
    y = plot_logic.linearized_transform(CA, n=1.0)
    assert np.allclose(y, np.log(CA))


def test_linearized_transform_second_order_is_power():
    CA = np.array([10.0, 5.0, 2.5])
    y = plot_logic.linearized_transform(CA, n=2.0)
    assert np.allclose(y, CA ** (1.0 - 2.0))


def test_linearized_transform_is_linear_in_time_for_true_model_data():
    """Sanity check tying plot_logic back to the physics: if CA(t) is
    generated from the exact nth-order integrated law, the linearized
    transform of CA against t must fall on a straight line (this is
    the entire premise of the linearized plot)."""
    CA0, k, n = 10.0, 0.03, 2.0
    t = np.linspace(0, 50, 20)
    CA = kinetics.integrated_concentration(t, CA0, k, n)
    y = plot_logic.linearized_transform(CA, n)
    slope, intercept = np.polyfit(t, y, deg=1)
    y_pred = slope * t + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    assert r2 > 0.9999


def test_linearized_axis_label_first_order_has_no_unit_exponent():
    label = plot_logic.linearized_axis_label(1.0, concentration_unit="mol/L")
    assert label == "ln(C_A)"


def test_linearized_axis_label_includes_unit_for_non_first_order():
    label = plot_logic.linearized_axis_label(2.0, concentration_unit="mol/L")
    assert "C_A^-1" in label
    assert "mol/L" in label


def test_linearized_axis_label_without_unit():
    label = plot_logic.linearized_axis_label(0.0, concentration_unit=None)
    assert label == "C_A^1"


def test_linearized_plot_title_mentions_first_order():
    assert "first-order" in plot_logic.linearized_plot_title(1.0).lower()


def test_linearized_plot_title_mentions_n_for_other_orders():
    title = plot_logic.linearized_plot_title(2.3)
    assert "2.3" in title
