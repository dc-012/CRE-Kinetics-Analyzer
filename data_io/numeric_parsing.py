"""
numeric_parsing.py
===================

Turns one raw spreadsheet cell (already a Python string, since files
are read with dtype=str so nothing is silently mangled by pandas'
automatic type inference) into a float, or reports that it could not
be parsed.

Handles the messiness explicitly called out in the Stage 2 brief:
    * inconsistent decimal separators ("3,14" vs "3.14")
    * thousands separators ("1,234.5")
    * a stray unit suffix stuck to a single value ("10.5 mM")
    * genuinely non-numeric stray text ("N/A", "pending", "-")
    * blank cells
"""

import re
from dataclasses import dataclass
from typing import Optional

_BLANK_TOKENS = {"", "nan", "none", "null", "n/a", "na", "-", "--", "?"}

# A number, optionally followed by a unit-like suffix, e.g. "10.5 mM",
# "3.2mol/L". The numeric part is captured in group 1.
_TRAILING_UNIT_RE = re.compile(r"^\s*(-?[0-9.,]+)\s*[a-zA-Z%/µ]*\s*$")


@dataclass
class ParsedCell:
    value: Optional[float]      # None if unparseable or blank
    is_blank: bool               # True if the cell was empty/NaN-like
    used_comma_decimal: bool     # True if a decimal-comma fix was applied
    stripped_suffix: Optional[str]  # unit text stripped off, if any


def parse_numeric_cell(raw, prefer_comma_decimal: bool = False) -> ParsedCell:
    """Parse one raw cell value. Never raises - always returns a
    ParsedCell describing what happened.

    Parameters
    ----------
    prefer_comma_decimal : bool
        Pass True when the surrounding file's field delimiter is
        already known to be something other than ',' (e.g. ';' or a
        tab) - in that case a comma inside a value cannot possibly be
        a field separator, so it is unambiguously a decimal separator
        regardless of how many digits follow it. When the delimiter is
        ',' itself (the ambiguous case), a shorter, more conservative
        heuristic is used instead (see `_parse_numeric_token`).
    """
    if raw is None:
        return ParsedCell(None, True, False, None)

    text = str(raw).strip()
    if text.lower() in _BLANK_TOKENS:
        return ParsedCell(None, True, False, None)

    # Fast path: plain float (covers the overwhelming majority of
    # already-clean files without any of the special-case logic below).
    try:
        return ParsedCell(float(text), False, False, None)
    except ValueError:
        pass

    match = _TRAILING_UNIT_RE.match(text)
    if not match:
        return ParsedCell(None, False, False, None)

    numeric_part = match.group(1)
    suffix = text[match.end(1):].strip() or None

    value, used_comma = _parse_numeric_token(numeric_part, prefer_comma_decimal)
    if value is None:
        return ParsedCell(None, False, False, None)

    return ParsedCell(value, False, used_comma, suffix)


def _parse_numeric_token(token: str, prefer_comma_decimal: bool = False):
    """
    Parse a numeric token that may use ',' as either a thousands
    separator or a decimal separator.

    Rules (deliberately simple and conservative - ambiguous or unusual
    formatting is left unparsed rather than guessed at):
        "3.14"       -> 3.14, False           (plain)
        "1,234"      -> 1234.0, False          (comma = thousands: >=3 digits after it, no dot present, comma IS the field delimiter elsewhere)
        "3,14"       -> 3.14, True             (comma = decimal: short tail, or file's field delimiter is not ',')
        "6,703"      -> 6.703, True            (comma = decimal, when file's delimiter is ';' or tab - see prefer_comma_decimal)
        "1,234.5"    -> 1234.5, False          (comma = thousands, dot = decimal)
        "1.234,5"    -> 1234.5, True           (dot = thousands, comma = decimal)
    """
    try:
        return float(token), False
    except ValueError:
        pass

    has_dot = "." in token
    has_comma = "," in token

    if has_comma and not has_dot:
        head, _, tail = token.rpartition(",")
        head_is_plain_int = head.lstrip("-").isdigit()
        short_tail_decimal = 1 <= len(tail) <= 2 and head_is_plain_int
        if prefer_comma_decimal or short_tail_decimal:
            # A comma that cannot be a field separator (because the
            # file uses a different delimiter), or a short tail that
            # looks like cents/decimals rather than a thousands group.
            try:
                return float(f"{head}.{tail}"), True
            except ValueError:
                return None, False
        # e.g. "1,234" -> thousands separator
        candidate = token.replace(",", "")
        try:
            return float(candidate), False
        except ValueError:
            return None, False

    if has_comma and has_dot:
        # Whichever separator appears last is the decimal separator.
        if token.rfind(",") > token.rfind("."):
            candidate = token.replace(".", "").replace(",", ".")
            try:
                return float(candidate), True
            except ValueError:
                return None, False
        else:
            candidate = token.replace(",", "")
            try:
                return float(candidate), False
            except ValueError:
                return None, False

    return None, False
