"""
plots.py
========

Builds the two required Plotly graphs directly from a
`core.calculations.KineticsResult`:

  1. concentration_vs_time_figure  - experimental C_A vs t, with the
     fitted nth-order model curve overlaid.
  2. linearized_kinetic_figure     - the linearized ("straight-line
     test") plot for the fitted order n, experimental points plus the
     model line.

No fitting or data cleaning happens in this file - every number comes
from the `KineticsResult` that was already produced by `core`, or from
`plot_logic.py`, which only re-evaluates the already-fitted model on a
denser time grid / applies a display transform (see that module's
docstring).
"""

import plotly.graph_objects as go

from core.calculations import KineticsResult
from core import kinetics
from . import plot_logic
from .formatting import format_r_squared


_TEMPLATE = "simple_white"
_EXPERIMENTAL_COLOR = "#5B8FC9"   # muted blue experimental data
_MODEL_COLOR = "#1F5EA8"          # formal blue model line


def _axis_title(base: str, unit: str = None) -> str:
    return f"{base} [{unit}]" if unit else base


def concentration_vs_time_figure(result: KineticsResult,
                                  time_unit: str = None,
                                  concentration_unit: str = None) -> go.Figure:
    """Graph 1: experimental concentration vs time with the fitted
    nth-order model curve."""
    t = result.time
    CA = result.concentration
    t_max = float(t.max()) if len(t) else 1.0

    t_dense, CA_dense = plot_logic.fitted_concentration_curve(
        CA0=result.CA0, k=result.rate_constant, n=result.reaction_order,
        t_max=t_max,
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=CA, mode="markers", name="Experimental data",
        marker=dict(color=_EXPERIMENTAL_COLOR, size=9, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=t_dense, y=CA_dense, mode="lines", name="Fitted nth-order model",
        line=dict(color=_MODEL_COLOR, width=2.5),
    ))

    order_label = "1st order" if kinetics.is_first_order(result.reaction_order) else f"n = {result.reaction_order:.3f}"
    fig.update_layout(
        template=_TEMPLATE,
        title=f"Concentration vs Time  (fitted order: {order_label}, R² = {format_r_squared(result.r_squared)})",
        xaxis_title=_axis_title("Time", time_unit),
        yaxis_title=_axis_title("Concentration, C_A", concentration_unit),
        legend=dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
        margin=dict(t=60, b=50, l=60, r=30),
    )
    return fig


def linearized_kinetic_figure(result: KineticsResult,
                               time_unit: str = None,
                               concentration_unit: str = None) -> go.Figure:
    """Graph 2: the linearized kinetic (integral-method) plot for the
    fitted reaction order - the transformed variable should fall on a
    straight line if the nth-order model is a good description of the
    data. Both the experimental points and the model line are shown in
    the SAME transformed space so their agreement is directly visible.
    """
    n = result.reaction_order
    t = result.time
    t_max = float(t.max()) if len(t) else 1.0

    y_experimental = plot_logic.linearized_transform(result.concentration, n)

    t_dense, CA_dense = plot_logic.fitted_concentration_curve(
        CA0=result.CA0, k=result.rate_constant, n=n, t_max=t_max,
    )
    y_model = plot_logic.linearized_transform(CA_dense, n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=y_experimental, mode="markers", name="Experimental data (transformed)",
        marker=dict(color=_EXPERIMENTAL_COLOR, size=9, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=t_dense, y=y_model, mode="lines", name="Fitted model (transformed)",
        line=dict(color=_MODEL_COLOR, width=2.5),
    ))

    fig.update_layout(
        template=_TEMPLATE,
        title=plot_logic.linearized_plot_title(n),
        xaxis_title=_axis_title("Time", time_unit),
        yaxis_title=plot_logic.linearized_axis_label(n, concentration_unit),
        legend=dict(x=0.98, y=0.02, xanchor="right", yanchor="bottom"),
        margin=dict(t=60, b=50, l=60, r=30),
    )
    return fig
