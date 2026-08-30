"""
ai_interpretation.py
=====================

Optional, OpenAI-powered plain-English explanation of an
already-computed `KineticsResult`. This module performs NO kinetics
calculations of its own and cannot change any numeric result - every
number it references was already computed by
`core.calculations.analyze_kinetics`. OpenAI is only asked to explain,
in words, the reaction order, rate constant, half-life, rate of
reaction, conversion, and fit quality that Python already calculated.

Reads the API key from the `OPENAI_API_KEY` environment variable.
If that variable is not set, the `openai` package is not installed,
or the API call fails for any reason, `generate_interpretation`
returns `None` and the rest of the application continues to work
normally without this section - this feature is optional, never
required for the app to run.
"""

import os
from typing import Optional

from core.calculations import KineticsResult

DEFAULT_MODEL = "gpt-4o-mini"


def api_key_available() -> bool:
    """True if an OpenAI API key is configured in the environment."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def build_prompt(
    result: KineticsResult,
    time_unit: Optional[str] = None,
    concentration_unit: Optional[str] = None,
) -> str:
    """
    Build the prompt sent to OpenAI. Contains only numbers already
    present on `result` - no raw per-point arrays, and the prompt
    explicitly instructs the model not to recalculate anything.
    """
    time_u = time_unit or "time units"
    conc_u = concentration_unit or "concentration units"

    half_life_txt = (
        f"{result.half_life:.4g} {time_u}"
        if result.half_life is not None
        else "not defined for this reaction order"
    )
    t_complete_txt = (
        f"{result.time_for_complete_conversion:.4g} {time_u}"
        if result.time_for_complete_conversion is not None
        else "not defined for this reaction order (only meaningful for n < 1)"
    )

    return (
        "You are explaining the results of a chemical reaction kinetics "
        "analysis to a chemical engineering student. All numbers below "
        "were already calculated by a validated numerical engine - do NOT "
        "recalculate, second-guess, or change any of them. Simply explain, "
        "in clear plain English and 4-6 short sentences, what these results "
        "mean physically and how much confidence the fit quality supports.\n\n"
        f"- Reaction order (n): {result.reaction_order:.4g}\n"
        f"- Rate constant (k): {result.rate_constant:.4g} [{result.rate_constant_units}]\n"
        f"- Goodness of fit (R-squared): {result.r_squared:.5f}\n"
        f"- Half-life: {half_life_txt}\n"
        f"- Time for complete conversion: {t_complete_txt}\n"
        f"- Rate of reaction at the initial concentration: {result.rate_at_CA0:.4g} {conc_u}/{time_u}\n"
        f"- Final conversion achieved: {result.conversion[-1] * 100:.3g}%\n\n"
        "Cover: what the reaction order and rate constant say about the "
        "reaction, what the half-life / time-for-complete-conversion mean "
        "in practice, and whether the R-squared value is good enough to "
        "trust these numbers."
    )


def generate_interpretation(
    result: KineticsResult,
    time_unit: Optional[str] = None,
    concentration_unit: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Optional[str]:
    """
    Ask OpenAI to explain, in words only, the numbers already present
    on `result`. Returns the explanation text, or `None` if no API key
    is configured, the `openai` package isn't installed, or the
    request fails for any reason.

    Callers should treat `None` as "interpretation unavailable" and
    continue to display the rest of the (Python-calculated) results
    without it.
    """
    if not api_key_available():
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    prompt = build_prompt(result, time_unit, concentration_unit)

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain already-calculated chemical reaction "
                        "engineering results in plain English. You never "
                        "perform or alter any calculation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        text = response.choices[0].message.content
        return text.strip() if text else None
    except Exception:
        # Any network/auth/API error: this feature is optional, so the
        # rest of the app must keep working without it.
        return None
