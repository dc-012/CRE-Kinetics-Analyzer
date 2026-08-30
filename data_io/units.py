"""
units.py
========

Helpers for pulling a unit hint (e.g. "min", "mol/L") out of a header
cell like "Time (min)" or "Concentration, mol/L".

Important scope note
---------------------
This module only *detects and reports* units for transparency - it
never converts between them. `core.kinetics` treats k's units
symbolically and works with whatever consistent time/concentration
units the user supplied (see `core/calculations.py:_rate_constant_units`).
Stage 1's engine has no notion of "the right" unit to convert to, so
Stage 2 does not invent one either. If a file mixed genuinely
different units within the same column (e.g. some rows in seconds,
some in minutes), that is a real data problem, not a formatting one,
and is intentionally left as visibly wrong data for the user to
notice and fix - inventing a silent conversion could hide a mistake.
"""

import re
from typing import Optional, Tuple

# Recognized unit tokens, lower-cased, used only to decide whether a
# fragment of a header cell "looks like a unit" (for reporting) - this
# list is deliberately not exhaustive.
_KNOWN_TIME_UNITS = {
    "s", "sec", "secs", "second", "seconds",
    "min", "mins", "minute", "minutes",
    "hr", "hrs", "hour", "hours",
    "day", "days",
}
_KNOWN_CONC_UNITS = {
    "m", "mm", "um", "µm", "nm",
    "mol/l", "mmol/l", "mol/ml", "g/l", "mg/l", "mg/ml", "g/ml",
    "ppm", "ppb", "%", "wt%",
}

_PAREN_RE = re.compile(r"[\(\[]([^\)\]]+)[\)\]]")


def split_name_and_unit(header_cell: str) -> Tuple[str, Optional[str]]:
    """
    Split a raw header cell into (clean_name, unit_or_None).

    Handles the common patterns:
        "Time (min)"        -> ("Time", "min")
        "Concentration [M]"  -> ("Concentration", "M")
        "Time, s"            -> ("Time", "s")
        "Concentration"      -> ("Concentration", None)

    The unit string is returned verbatim (for the cleaning report) -
    it is never parsed into a conversion factor.
    """
    text = str(header_cell).strip()

    match = _PAREN_RE.search(text)
    if match:
        unit = match.group(1).strip()
        name = _PAREN_RE.sub("", text).strip()
        return name, (unit if unit else None)

    if "," in text:
        name_part, _, unit_part = text.partition(",")
        candidate_unit = unit_part.strip()
        if _looks_like_unit(candidate_unit):
            return name_part.strip(), candidate_unit

    return text, None


def _looks_like_unit(token: str) -> bool:
    normalized = token.strip().lower()
    return normalized in _KNOWN_TIME_UNITS or normalized in _KNOWN_CONC_UNITS


def looks_like_units_row(cells) -> bool:
    """
    True if a whole spreadsheet row looks like a standalone "units row"
    sitting between the header and the data, e.g. a row of bare
    "(min)", "(mol/L)" cells with no numeric content. Used to detect
    and skip the common two-row-header pattern:

        Time | Concentration
        (min) | (mol/L)
        0     | 10.0
        ...
    """
    non_empty = [str(c).strip() for c in cells if str(c).strip() != "" and str(c).lower() != "nan"]
    if not non_empty:
        return False
    matches = 0
    for cell in non_empty:
        stripped = cell.strip("()[]").strip()
        if _looks_like_unit(stripped):
            matches += 1
    return matches == len(non_empty)
