"""
validation.py
==============

Validates raw concentration-time data before any kinetic analysis is
attempted.

This module does NOT do any kinetics math. It only checks that the data
is physically and numerically sane enough to analyze. Keeping this
separate from kinetics.py / model_selection.py makes it easy to re-use
in Stage 2 (where data will come from messier CSV/Excel uploads).

Physical background for the checks below
-----------------------------------------
The whole project assumes a single irreversible reaction studied in a
constant-volume batch reactor, with the rate law

    -r_A = -dC_A/dt = k * C_A^n          (Levenspiel, Ch. 3)

For this model to make sense with a given data set:
    * concentration must be a strictly positive, physically real
      quantity (C_A^n and ln(C_A) are only defined for C_A > 0),
    * time must start at t = 0, where C_A = C_AO is the initial
      concentration used everywhere else in the analysis,
    * time must be strictly increasing (batch reactor data is a time
      series; repeated or decreasing times make no physical sense),
    * concentration must show a genuine *net* decrease from t=0 to the
      final point, otherwise there is no conversion to analyze.

Anything less severe (e.g. a little experimental noise causing a local
uptick in concentration between two adjacent samples) is reported as a
warning rather than a hard failure, because Stage 1 is explicitly
required to handle noisy experimental data.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


class DataValidationError(ValueError):
    """Raised when the supplied data cannot be analyzed at all."""
    pass


@dataclass
class ValidatedData:
    """Clean, numpy-array version of the input data, plus any warnings
    that were raised along the way but were not serious enough to stop
    the analysis."""
    time: np.ndarray
    concentration: np.ndarray
    warnings: List[str] = field(default_factory=list)


MIN_DATA_POINTS = 4


def validate_concentration_time_data(time, concentration) -> ValidatedData:
    """
    Validate raw time/concentration input.

    Parameters
    ----------
    time : sequence of float
        Experimental time values. The first value MUST be 0, since the
        concentration at t=0 (C_AO) is the reference initial
        concentration used throughout the rest of the engine.
    concentration : sequence of float
        Experimental concentration values, same length as `time`,
        aligned index-for-index (concentration[i] measured at time[i]).

    Returns
    -------
    ValidatedData
        Cleaned numpy arrays plus a list of non-fatal warnings.

    Raises
    ------
    DataValidationError
        If the data cannot be analyzed at all (see checks below).
    """
    warnings: List[str] = []

    # --- basic structural checks -----------------------------------
    if time is None or concentration is None:
        raise DataValidationError("Both time and concentration must be provided.")

    try:
        t = np.asarray(time, dtype=float)
        c = np.asarray(concentration, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            f"Time and concentration must be numeric sequences. Got error: {exc}"
        )

    if t.ndim != 1 or c.ndim != 1:
        raise DataValidationError("Time and concentration must be 1-D sequences.")

    if len(t) != len(c):
        raise DataValidationError(
            f"Time and concentration must have the same length "
            f"(got {len(t)} time values and {len(c)} concentration values)."
        )

    if len(t) < MIN_DATA_POINTS:
        raise DataValidationError(
            f"At least {MIN_DATA_POINTS} data points are required to fit a "
            f"reaction order and rate constant reliably; got {len(t)}."
        )

    # --- NaN / infinite checks --------------------------------------
    if np.any(~np.isfinite(t)):
        raise DataValidationError("Time values contain NaN or infinite entries.")
    if np.any(~np.isfinite(c)):
        raise DataValidationError("Concentration values contain NaN or infinite entries.")

    # --- physical positivity of concentration -----------------------
    if np.any(c <= 0):
        raise DataValidationError(
            "All concentration values must be strictly positive. "
            "C_A^n and ln(C_A), used throughout the nth-order rate law "
            "-r_A = k*C_A^n, are undefined for C_A <= 0. If your data "
            "genuinely reaches zero (complete conversion), drop that "
            "point or replace it with a small positive detection-limit "
            "value."
        )

    # --- time ordering -----------------------------------------------
    if np.any(np.diff(t) <= 0):
        raise DataValidationError(
            "Time values must be strictly increasing, with no repeated "
            "timestamps. Sort your data by time and remove duplicates "
            "before analysis."
        )

    if not np.isclose(t[0], 0.0, atol=1e-9):
        raise DataValidationError(
            f"The first time value must be 0 (t[0] = {t[0]}). The "
            f"concentration at t=0 is used as C_AO, the reference "
            f"initial concentration for every calculation in this "
            f"engine (conversion, half-life, the integrated rate law, "
            f"etc). Shift your time axis so the first sample is t=0."
        )

    # --- net conversion must be a decrease ---------------------------
    if c[-1] >= c[0]:
        raise DataValidationError(
            "Concentration must show a net decrease from t=0 to the "
            "final data point (this engine models a single irreversible "
            "reactant disappearing via -r_A = k*C_A^n). "
            f"Got C_A(0) = {c[0]} and C_A(final) = {c[-1]}."
        )

    # --- soft check: local non-monotonicity (noise) -------------------
    increases = np.sum(np.diff(c) > 0)
    if increases > 0:
        frac = increases / (len(c) - 1)
        warnings.append(
            f"Concentration increases locally between {increases} of "
            f"{len(c) - 1} consecutive sample pairs. This is treated as "
            f"experimental noise since the overall trend still decreases, "
            f"but if this fraction ({frac:.0%}) is large, consider "
            f"re-checking the raw data."
        )
        if frac > 0.4:
            warnings.append(
                "WARNING: more than 40% of consecutive intervals show an "
                "increase in concentration. Reaction-order fitting may be "
                "unreliable with this much apparent noise."
            )

    # --- soft check: small sample size --------------------------------
    if len(t) < 6:
        warnings.append(
            f"Only {len(t)} data points supplied. Reaction order and rate "
            f"constant estimates are more reliable with 6 or more points."
        )

    return ValidatedData(time=t, concentration=c, warnings=warnings)
