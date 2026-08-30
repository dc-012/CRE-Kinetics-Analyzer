"""
formatting.py
=============

Small, pure formatting helpers for displaying `KineticsResult` values
in the UI. No calculation happens here - only turning already-computed
numbers (and the occasional `None`, e.g. an undefined half-life) into
readable strings. Kept separate from `streamlit_app.py` so it can be
unit tested without importing Streamlit.
"""

from typing import Optional

from core.kinetics import is_first_order


def format_number(value: Optional[float], sig_figs: int = 4, suffix: str = "") -> str:
    """
    Format a float to a sensible number of significant figures for
    engineering display, e.g. 0.0031415 -> "0.003142", 123456 ->
    "1.235e+05". Returns "N/A" for None (used for quantities that can
    be undefined for some reaction orders, e.g. half-life or time for
    complete conversion when n >= 1).
    """
    if value is None:
        return "N/A"
    try:
        if value == 0:
            return f"0{suffix}"
        formatted = f"{value:.{sig_figs}g}"
        return f"{formatted}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def format_complete_conversion_time(value: Optional[float], reaction_order: Optional[float] = None, suffix: str = "") -> str:
    """Display a finite conversion time for n < 1 and a theoretical infinity for n >= 1."""
    if value is not None:
        return format_number(value, suffix=suffix)

    if reaction_order is not None:
        if reaction_order >= 1.0 or is_first_order(reaction_order):
            return "∞ (theoretical)"

    return "N/A"


def format_reaction_order(n: float) -> str:
    """Reaction order to 3 decimal places, e.g. 1.998 -> '1.998' (kept
    at fixed decimals, not sig-figs, since 'n' is most naturally read
    as a small number close to a simple fraction)."""
    return f"{n:.3f}"


def format_percent(fraction: float, sig_figs: int = 3) -> str:
    """Fractional conversion (0-1) as a percentage string."""
    return format_number(fraction * 100.0, sig_figs=sig_figs, suffix="%")


def format_r_squared(r2: float) -> str:
    return f"{r2:.5f}"


def fit_quality_label(r2: float) -> str:
    """A short, honest, non-alarmist qualitative label for R^2, used
    next to the numeric value so a non-specialist reader has context."""
    if r2 >= 0.995:
        return "Excellent fit"
    if r2 >= 0.98:
        return "Good fit"
    if r2 >= 0.90:
        return "Fair fit - inspect the graphs"
    return "Poor fit - reaction may not follow a single nth-order law"
