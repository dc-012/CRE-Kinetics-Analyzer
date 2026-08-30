"""
plot_logic.py
=============

Pure helper functions that prepare the NUMBERS for the two required
graphs (concentration-vs-time with fitted curve, and the linearized
kinetic plot). No Plotly import and no Streamlit import here - this
module only calls into the existing `core.kinetics` functions and
numpy, which makes it possible to unit test the plotting math without
needing the plotting/UI libraries installed at all.

Nothing here re-derives or re-fits (n, k). Both functions below take
an already-computed `core.calculations.KineticsResult` and simply:
  (a) evaluate the SAME integrated rate law (`core.kinetics.
      integrated_concentration`) that produced `predicted_concentration`,
      on a denser time grid, so the fitted curve looks smooth, and
  (b) apply the standard linearizing transform for the fitted order n
      (the same transform `core.model_selection` uses internally) to
      both the experimental and the model concentrations, purely for
      display.
"""

from typing import Tuple
import numpy as np

from core import kinetics


def dense_time_grid(t_max: float, n_points: int = 200) -> np.ndarray:
    """
    A fine time grid from 0 to t_max, used only to draw a smooth
    fitted-model curve (the experimental data stays at its own,
    typically coarser, sample times).
    """
    if t_max <= 0:
        t_max = 1.0
    return np.linspace(0.0, t_max, n_points)


def fitted_concentration_curve(CA0: float, k: float, n: float,
                                t_max: float, n_points: int = 200
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth model curve C_A(t) for the main concentration-vs-time plot,
    evaluated with `core.kinetics.integrated_concentration` - the same
    function used to produce `KineticsResult.predicted_concentration` -
    just on a denser time grid than the experimental samples so the
    plotted curve looks continuous instead of piecewise-linear.

    Returns
    -------
    (t_dense, CA_dense)
    """
    t_dense = dense_time_grid(t_max, n_points)
    CA_dense = kinetics.integrated_concentration(t_dense, CA0, k, n)
    return t_dense, np.asarray(CA_dense, dtype=float)


def linearized_transform(concentration, n: float) -> np.ndarray:
    """
    The standard integral-method linearizing transform for order n
    (see `core.model_selection`'s module docstring, step 1):

        y(n) = ln(C_A)        if n == 1  (removable singularity)
        y(n) = C_A^(1-n)      otherwise

    A correct fit makes y linear in t. Used here only to prepare data
    for the "transformed kinetic plot" - the transform itself is
    exactly the one `core.model_selection._linear_transform_fit` uses
    internally, just re-applied here for display rather than fitting.
    """
    CA = np.asarray(concentration, dtype=float)
    if kinetics.is_first_order(n):
        return np.log(CA)
    return np.power(CA, 1.0 - n)


def linearized_axis_label(n: float, concentration_unit: str = None) -> str:
    """Human-readable y-axis label for the linearized plot.

    ln(C_A) is left unitless in the label (the standard convention,
    since the argument of ln must be dimensionless); the power-law
    transform C_A^(1-n) does carry the concentration unit raised to
    the same exponent, so that is included when a unit is known.
    """
    if kinetics.is_first_order(n):
        return "ln(C_A)"
    exponent = 1.0 - n
    label = f"C_A^{exponent:.4g}"
    if concentration_unit:
        label += f"  [({concentration_unit})^{exponent:.4g}]"
    return label


def linearized_plot_title(n: float) -> str:
    if kinetics.is_first_order(n):
        return "Linearized Kinetic Plot: ln(C_A) vs t  (first-order test)"
    return f"Linearized Kinetic Plot: C_A^(1-n) vs t  (n = {n:.4g})"
