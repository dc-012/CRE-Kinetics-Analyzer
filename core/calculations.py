"""
calculations.py
================

Top-level orchestration: takes raw time/concentration data in, runs
validation -> order determination -> derived-quantity calculations,
and returns one clean, structured result.

This is the ONLY module a future UI (Streamlit, in Stage 2/3) should
need to import from `core`. It intentionally contains no I/O
(no file reading, no printing, no plotting) - just orchestration and
data assembly - so it stays reusable from a script, a test, or a web
app equally.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .validation import validate_concentration_time_data
from .model_selection import determine_best_order, OrderCandidate
from . import kinetics


@dataclass
class KineticsResult:
    """
    Clean, structured result of a full kinetics analysis. Every field
    is a plain Python type or numpy array / list, so this can be
    handed directly to a UI layer, serialized to JSON, or written into
    an Excel report in a later stage without modification.
    """
    # --- inputs (echoed back for convenience / reporting) -----------
    time: np.ndarray
    concentration: np.ndarray
    CA0: float

    # --- fitted kinetic parameters -----------------------------------
    reaction_order: float
    rate_constant: float
    rate_constant_units: str

    # --- goodness of fit ----------------------------------------------
    r_squared: float

    # --- derived quantities --------------------------------------------
    half_life: Optional[float]
    time_for_complete_conversion: Optional[float]
    conversion: np.ndarray            # X_A at every experimental time point
    rate_at_each_point: np.ndarray    # -r_A at every experimental C_A
    rate_at_CA0: float                # -r_A evaluated at the initial concentration
    predicted_concentration: np.ndarray  # model C_A(t) at experimental t, for plotting/residuals

    # --- transparency / diagnostics -------------------------------------
    candidate_orders: List[OrderCandidate]
    warnings: List[str] = field(default_factory=list)


def _rate_constant_units(n: float) -> str:
    """
    Units of k in -r_A = k*C_A^n follow directly from dimensional
    consistency: [rate] = [conc]/[time] = [k] * [conc]^n, so
    [k] = [conc]^(1-n) / [time]. We report this symbolically since the
    actual concentration/time units depend on what the user supplied.
    """
    if kinetics.is_first_order(n):
        return "1/time"
    exponent = 1.0 - n
    return f"(concentration)^{exponent:.4g} / time"


def analyze_kinetics(time, concentration, n_bounds=(-1.0, 4.0)) -> KineticsResult:
    """
    Run the complete Stage 1 analysis pipeline on raw concentration-
    time data.

    Parameters
    ----------
    time : sequence of float
        Experimental time values, first value must be 0.
    concentration : sequence of float
        Experimental concentration values aligned with `time`.
    n_bounds : (float, float), optional
        Search range for the reaction order (see model_selection.py).

    Returns
    -------
    KineticsResult

    Raises
    ------
    validation.DataValidationError
        If the input data fails basic sanity checks (see validation.py
        for the full list of checks and why each one exists).
    """
    validated = validate_concentration_time_data(time, concentration)
    t, CA = validated.time, validated.concentration
    CA0 = float(CA[0])

    selection = determine_best_order(t, CA, n_bounds=n_bounds)
    n = selection.best_n
    k = selection.best_k

    predicted_CA = kinetics.integrated_concentration(t, CA0, k, n)
    rate_at_points = kinetics.rate_of_reaction(CA, k, n)
    rate_at_CA0 = float(kinetics.rate_of_reaction(CA0, k, n))
    conv = kinetics.conversion(CA0, CA)

    try:
        t_half = kinetics.half_life(CA0, k, n)
        if not np.isfinite(t_half) or t_half < 0:
            t_half = None
    except (ZeroDivisionError, FloatingPointError, OverflowError):
        t_half = None

    t_complete = kinetics.time_for_complete_conversion(CA0, k, n)

    return KineticsResult(
        time=t,
        concentration=CA,
        CA0=CA0,
        reaction_order=n,
        rate_constant=k,
        rate_constant_units=_rate_constant_units(n),
        r_squared=selection.r2,
        half_life=t_half,
        time_for_complete_conversion=t_complete,
        conversion=conv,
        rate_at_each_point=rate_at_points,
        rate_at_CA0=rate_at_CA0,
        predicted_concentration=predicted_CA,
        candidate_orders=selection.candidates_scanned,
        warnings=list(validated.warnings),
    )
