"""
errors.py
=========

`IngestionError` covers problems with the *file itself*: it couldn't be
opened, no time/concentration columns could be identified, or cleaning
left too little usable data to analyze.

This is intentionally a different exception from
`core.validation.DataValidationError`, which covers problems with the
*data values* (negative concentration, non-increasing time, etc.) once
a clean (time, concentration) pair has already been extracted. Keeping
them separate means a user-facing UI (Stage 3) can tell "your file was
malformed" apart from "your data doesn't behave like this rate law"
without any string-matching on error messages.
"""


class IngestionError(ValueError):
    """Raised when a file cannot be read or cleaned into usable
    (time, concentration) data at all."""
    pass
