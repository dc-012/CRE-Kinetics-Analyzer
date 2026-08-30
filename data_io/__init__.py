"""
data_io
=======

Stage 2: turns messy real-world CSV/Excel file uploads into the clean
`(time, concentration)` arrays that `core.calculations.analyze_kinetics`
expects.

This package contains NO kinetics math and NO UI code. It only reads
files, figures out which columns are time/concentration, cleans up
common spreadsheet messiness (extra header rows, stray text, blank
rows, inconsistent decimal separators, etc.), and reports exactly what
it did. Genuine data-quality problems (negative concentrations,
non-increasing time, no net conversion, ...) are NOT re-checked here -
that is `core.validation`'s job, and this package deliberately calls
into it rather than duplicating it.

Public entry points
--------------------
load_kinetics_data(path)
    Reads and cleans a CSV/Excel file. Returns (time, concentration,
    CleaningReport). Raises IngestionError if the file cannot be
    cleaned into usable (time, concentration) pairs at all.

analyze_kinetics_file(path, n_bounds=(-1.0, 4.0))
    Convenience wrapper: cleans the file, then calls
    `core.calculations.analyze_kinetics` on the result. Returns
    (KineticsResult, CleaningReport). Raises IngestionError (cleaning
    problems) or `core.validation.DataValidationError` (data-quality
    problems caught by Stage 1's own checks) - see each error's
    message for which one occurred.
"""

from .errors import IngestionError
from .cleaning import CleaningReport
from .pipeline import load_kinetics_data, analyze_kinetics_file

__all__ = [
    "IngestionError",
    "CleaningReport",
    "load_kinetics_data",
    "analyze_kinetics_file",
]
