"""
table_loader.py
================

Reads a CSV or Excel file into raw, untyped string tables - one table
per sheet for Excel, a single table for CSV.

Deliberately reads everything as strings (`dtype=str` for CSV,
`dtype=object` then str-cast for Excel) rather than letting pandas
guess numeric types. Letting pandas infer types would silently mangle
exactly the messiness we need to see and clean explicitly, e.g. a
decimal comma "3,14" would otherwise be read as the string "3" and
"14" split into two columns, or a genuinely blank cell would look the
same as a cell that already failed to parse.
"""

import csv
import os
from typing import List, Tuple

import pandas as pd

from .errors import IngestionError

_CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]


def load_raw_tables(path: str) -> List[Tuple[str, List[List[str]], object]]:
    """
    Read a file into one or more raw tables.

    Returns
    -------
    list of (sheet_name, rows, delimiter)
        `sheet_name` is "" for CSV files (which have no sheets) or the
        Excel sheet name. `rows` is a list of rows, each a list of
        cell strings (empty string for blank cells). `delimiter` is
        the detected CSV field delimiter (e.g. ',' or ';'), or None
        for Excel files - passed through so numeric parsing knows
        whether a ',' inside a value could possibly be a field
        separator (see `numeric_parsing.parse_numeric_cell`).

    Raises
    ------
    IngestionError
        If the file doesn't exist, has an unsupported extension, or
        can't be parsed at all (corrupt file, wrong format, etc.).
    """
    if not os.path.exists(path):
        raise IngestionError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        rows, delimiter = _read_csv(path)
        return [("", rows, delimiter)]
    elif ext in (".xlsx", ".xls"):
        return [(name, rows, None) for name, rows in _read_excel(path)]
    else:
        raise IngestionError(
            f"Unsupported file type '{ext}'. Only .csv, .xlsx, and "
            f".xls files are supported."
        )


def _read_csv(path: str) -> Tuple[List[List[str]], str]:
    """
    Read a CSV file into raw rows, tolerating ragged/messy files
    (varying column counts, stray title/footer lines with no
    delimiter at all) that would make pandas' strict CSV parser raise.

    The delimiter is detected by simple majority vote across
    candidates (comma, semicolon, tab, pipe) rather than assumed,
    since files using ';' as the field separator are common wherever
    ',' is used as the decimal separator.
    """
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as fh:
            raw_text = fh.read()
    except OSError as exc:
        raise IngestionError(f"Could not open '{path}': {exc}") from exc

    lines = raw_text.splitlines()
    delimiter = _detect_delimiter(lines)

    try:
        reader = csv.reader(lines, delimiter=delimiter)
        rows = [row for row in reader]
    except csv.Error as exc:
        raise IngestionError(f"Could not parse '{path}' as CSV: {exc}") from exc

    return rows, delimiter


def _detect_delimiter(lines: List[str]) -> str:
    best_delim, best_score = ",", -1
    for delim in _CANDIDATE_DELIMITERS:
        counts = [line.count(delim) for line in lines if line.strip()]
        if not counts:
            continue
        # Score = how many lines share the most common (nonzero) count
        # of this delimiter - a real field separator shows up a
        # consistent number of times per data row.
        nonzero_counts = [c for c in counts if c > 0]
        if not nonzero_counts:
            continue
        mode_count = max(set(nonzero_counts), key=nonzero_counts.count)
        score = nonzero_counts.count(mode_count)
        if score > best_score:
            best_delim, best_score = delim, score
    return best_delim


def _read_excel(path: str) -> List[Tuple[str, List[List[str]]]]:
    try:
        sheets = pd.read_excel(path, header=None, dtype=str, sheet_name=None)
    except Exception as exc:
        raise IngestionError(f"Could not parse '{path}' as Excel: {exc}") from exc

    tables = []
    for sheet_name, df in sheets.items():
        df = df.fillna("")
        tables.append((sheet_name, _dataframe_to_rows(df)))
    if not tables:
        raise IngestionError(f"'{path}' contains no sheets.")
    return tables


def _dataframe_to_rows(df) -> List[List[str]]:
    df = df.fillna("")
    return [[("" if cell is None else str(cell)) for cell in row] for row in df.values.tolist()]
