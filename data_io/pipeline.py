"""
pipeline.py
===========

The two public entry points for Stage 2:

    load_kinetics_data(path)       -> (time, concentration, CleaningReport)
    analyze_kinetics_file(path)    -> (KineticsResult, CleaningReport)

Neither function duplicates any of `core.validation`'s data-quality
checks. `load_kinetics_data` only handles file-format messiness; once
it returns, the `(time, concentration)` arrays are passed to
`core.calculations.analyze_kinetics` completely unchanged, and any
`DataValidationError` it raises is left to propagate (with a short
prefix noting which file it came from) rather than being re-implemented
here.
"""

from typing import List, Tuple

from core.calculations import analyze_kinetics, KineticsResult
from core.validation import DataValidationError

from .cleaning import clean_table, CleaningReport
from .errors import IngestionError
from .table_loader import load_raw_tables


def load_kinetics_data(path: str) -> Tuple[List[float], List[float], CleaningReport]:
    """
    Read and clean a CSV or Excel file into (time, concentration).

    For Excel files with multiple sheets, the first sheet on which a
    header row with recognizable time/concentration columns can be
    found is used; if more than one sheet qualifies, this is noted in
    the report so the user knows other sheets were ignored.

    Raises
    ------
    IngestionError
        If the file can't be opened, no recognizable time/concentration
        columns are found in any sheet, or cleaning leaves no usable
        data rows.
    """
    tables = load_raw_tables(path)

    qualifying_sheets = []
    last_error = None
    for sheet_name, rows, delimiter in tables:
        try:
            time, concentration, report = clean_table(
                rows, source_file=path, sheet_name=sheet_name, delimiter=delimiter
            )
            qualifying_sheets.append((sheet_name, time, concentration, report))
        except ValueError as exc:
            last_error = exc
            continue

    if not qualifying_sheets:
        detail = f" Details: {last_error}" if last_error else ""
        raise IngestionError(
            f"Could not extract usable (time, concentration) data from "
            f"'{path}'.{detail}"
        )

    sheet_name, time, concentration, report = qualifying_sheets[0]
    if len(qualifying_sheets) > 1:
        other_names = [s for s, *_ in qualifying_sheets[1:]]
        report.notes.append(
            f"Multiple sheets contained usable data; used the first "
            f"one ('{sheet_name}' if named) and ignored: {other_names}."
        )

    return time, concentration, report


def analyze_kinetics_file(path: str, n_bounds=(-1.0, 4.0)) -> Tuple[KineticsResult, CleaningReport]:
    """
    Clean a file and run it straight through Stage 1's
    `analyze_kinetics`, unchanged.

    Raises
    ------
    IngestionError
        If the file itself can't be cleaned into usable data (see
        `load_kinetics_data`).
    core.validation.DataValidationError
        If the cleaned data fails Stage 1's own data-quality checks
        (e.g. negative concentrations, non-increasing time). The
        original message from `core.validation` is preserved and only
        prefixed with the file name, per the brief's instruction to
        build on top of those messages rather than duplicate them.
    """
    time, concentration, report = load_kinetics_data(path)

    try:
        result = analyze_kinetics(time, concentration, n_bounds=n_bounds)
    except DataValidationError as exc:
        raise DataValidationError(
            f"'{path}' was read and cleaned successfully "
            f"({report.rows_used} usable rows), but the resulting data "
            f"failed validation: {exc}"
        ) from exc

    return result, report
