"""
column_detection.py
====================

Finds which row is the real header row (real spreadsheets often have
a title row, a blank row, or a second "units" row before the actual
column headers) and which columns are time vs concentration,
regardless of their order or exact wording.

Detection is name-based (against a list of common aliases), not
position-based, which is what lets this handle "differently-named/
ordered columns" as required by the Stage 2 brief.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .units import split_name_and_unit

# Normalized (lowercase, alphanumeric-only) aliases for each column.
_TIME_ALIASES = {
    "time", "t", "reactiontime", "elapsedtime", "timeelapsed",
    "reactiontimet", "timet",
}
_CONCENTRATION_ALIASES = {
    "concentration", "conc", "ca", "concentrationofa", "concentrationa",
    "concofa", "cadata", "reactantconcentration",
}
# Bracket-notation chemistry shorthand for concentration, e.g. "[A]".
_BRACKET_CONCENTRATION_RE = re.compile(r"^\s*\[\s*A\s*\]\s*$", re.IGNORECASE)

# How many rows from the top to search for the header row.
_MAX_HEADER_SEARCH_ROWS = 10


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


@dataclass
class ColumnMap:
    header_row_index: int
    time_col: int
    concentration_col: int
    time_col_label: str
    concentration_col_label: str
    time_unit: Optional[str]
    concentration_unit: Optional[str]


def _classify_cell(raw_cell: str) -> Optional[str]:
    """Return 'time', 'concentration', or None for one header cell."""
    if _BRACKET_CONCENTRATION_RE.match(str(raw_cell)):
        return "concentration"
    name, _ = split_name_and_unit(raw_cell)
    normalized = _normalize(name)
    if normalized in _TIME_ALIASES:
        return "time"
    if normalized in _CONCENTRATION_ALIASES:
        return "concentration"
    return None


def find_header_and_columns(rows: List[List[str]]) -> ColumnMap:
    """
    Scan the first few rows of a raw (untyped) table for a row that
    contains both a recognizable time column and a recognizable
    concentration column.

    Parameters
    ----------
    rows : list of list of str
        Raw cell values, row-major, exactly as read from the file
        (before any cleaning).

    Returns
    -------
    ColumnMap

    Raises
    ------
    ValueError
        If no row in the search window has both columns identifiable.
        (Callers should translate this into an IngestionError with
        file context - see pipeline.py.)
    """
    search_limit = min(len(rows), _MAX_HEADER_SEARCH_ROWS)
    for row_idx in range(search_limit):
        row = rows[row_idx]
        time_col = None
        conc_col = None
        for col_idx, cell in enumerate(row):
            role = _classify_cell(cell)
            if role == "time" and time_col is None:
                time_col = col_idx
            elif role == "concentration" and conc_col is None:
                conc_col = col_idx

        if time_col is not None and conc_col is not None:
            time_name, time_unit = split_name_and_unit(row[time_col])
            conc_name, conc_unit = split_name_and_unit(row[conc_col])
            return ColumnMap(
                header_row_index=row_idx,
                time_col=time_col,
                concentration_col=conc_col,
                time_col_label=str(row[time_col]).strip(),
                concentration_col_label=str(row[conc_col]).strip(),
                time_unit=time_unit,
                concentration_unit=conc_unit,
            )

    raise ValueError(
        "Could not find a header row with recognizable time and "
        "concentration columns in the first "
        f"{search_limit} row(s) of the file. Recognized time column "
        "names include 'time' or 't' (optionally with a unit, e.g. "
        "'Time (min)'); recognized concentration column names include "
        "'concentration', 'conc', 'CA', 'C_A', or '[A]' (optionally "
        "with a unit, e.g. 'Concentration (mol/L)')."
    )
