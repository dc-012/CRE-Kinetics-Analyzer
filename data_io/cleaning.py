"""
cleaning.py
===========

Takes one raw (untyped) table - as produced by `table_loader` - and
extracts clean `(time, concentration)` float lists, handling the
common real-world spreadsheet messiness described in the Stage 2
brief:

    * extra header/title rows before the real header
    * a second "units" row between the header and the data
    * differently-named/ordered columns (handled by `column_detection`)
    * units embedded in header cells (handled by `units.py`)
    * blank rows
    * trailing notes/footer text after the data block
    * non-numeric stray values mixed into otherwise-numeric columns
    * inconsistent decimal separators (handled by `numeric_parsing.py`)

What this module deliberately does NOT do: decide whether the
resulting data is physically sensible (positive concentrations,
increasing time, net conversion, etc.). Those checks belong to
`core.validation` alone - duplicating them here would risk the two
disagreeing. This module's job ends at "here is a clean list of
numbers, paired up correctly."
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .column_detection import find_header_and_columns
from .numeric_parsing import parse_numeric_cell
from .units import looks_like_units_row


@dataclass
class CleaningReport:
    """Everything Stage 2 did to a file, for a user-facing summary."""
    source_file: str
    sheet_name: str = ""
    header_row_index: int = 0
    time_column_label: str = ""
    concentration_column_label: str = ""
    time_unit: Optional[str] = None
    concentration_unit: Optional[str] = None
    rows_used: int = 0
    rows_dropped_blank: int = 0
    rows_dropped_non_numeric: int = 0
    rows_dropped_footer: int = 0
    decimal_comma_fixes: int = 0
    units_row_skipped: bool = False
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Short human-readable summary, suitable for display to the
        end user (a UI in a later stage, or a printed report now)."""
        lines = [
            f"Source: {self.source_file}"
            + (f" (sheet '{self.sheet_name}')" if self.sheet_name else ""),
            f"Detected columns: time = '{self.time_column_label}'"
            + (f" [{self.time_unit}]" if self.time_unit else "")
            + f", concentration = '{self.concentration_column_label}'"
            + (f" [{self.concentration_unit}]" if self.concentration_unit else ""),
            f"Rows used: {self.rows_used}",
        ]
        if self.units_row_skipped:
            lines.append("Skipped a units row directly under the header.")
        if self.rows_dropped_blank:
            lines.append(f"Dropped {self.rows_dropped_blank} blank row(s).")
        if self.rows_dropped_non_numeric:
            lines.append(
                f"Dropped {self.rows_dropped_non_numeric} row(s) with "
                f"non-numeric stray values."
            )
        if self.rows_dropped_footer:
            lines.append(
                f"Dropped {self.rows_dropped_footer} trailing footer/notes "
                f"row(s)."
            )
        if self.decimal_comma_fixes:
            lines.append(
                f"Interpreted ',' as a decimal separator in "
                f"{self.decimal_comma_fixes} cell(s)."
            )
        for note in self.notes:
            lines.append(note)
        return "\n".join(lines)


def clean_table(rows: List[List[str]], source_file: str, sheet_name: str = "",
                 delimiter: str = None) -> ("tuple[list, list, CleaningReport]"):
    """
    Clean one raw table into (time_list, concentration_list, report).

    Parameters
    ----------
    delimiter : str or None
        The file's field delimiter, if it came from a CSV (see
        `table_loader.load_raw_tables`). Used only to disambiguate a
        ',' found inside a numeric value: if the file's delimiter is
        something else (e.g. ';'), a ',' inside a value cannot be a
        field separator and is unambiguously a decimal separator.
        None (e.g. for Excel) falls back to the more conservative
        digit-count heuristic in `numeric_parsing`.

    Raises
    ------
    ValueError
        If no header row with recognizable columns can be found
        (propagated from `column_detection.find_header_and_columns`),
        or if cleaning leaves fewer than 1 usable data row.
    """
    prefer_comma_decimal = delimiter is not None and delimiter != ","
    col_map = find_header_and_columns(rows)

    report = CleaningReport(
        source_file=source_file,
        sheet_name=sheet_name,
        header_row_index=col_map.header_row_index,
        time_column_label=col_map.time_col_label,
        concentration_column_label=col_map.concentration_col_label,
        time_unit=col_map.time_unit,
        concentration_unit=col_map.concentration_unit,
    )

    data_rows = rows[col_map.header_row_index + 1:]

    # A second "units" row (e.g. "(min)", "(mol/L)") directly under the
    # header is common enough to special-case: skip it explicitly
    # rather than letting it fall through to the stray-value logic
    # below, which would also work but couldn't label it as clearly.
    if data_rows and looks_like_units_row(data_rows[0]):
        data_rows = data_rows[1:]
        report.units_row_skipped = True

    times: List[float] = []
    concentrations: List[float] = []

    base_index = col_map.header_row_index + 1 + (1 if report.units_row_skipped else 0)
    for offset, row in enumerate(data_rows):
        row_number = base_index + offset + 1  # 1-based, for human-readable messages
        time_raw = row[col_map.time_col] if col_map.time_col < len(row) else ""
        conc_raw = row[col_map.concentration_col] if col_map.concentration_col < len(row) else ""

        time_parsed = parse_numeric_cell(time_raw, prefer_comma_decimal)
        conc_parsed = parse_numeric_cell(conc_raw, prefer_comma_decimal)

        if time_parsed.is_blank and conc_parsed.is_blank:
            report.rows_dropped_blank += 1
            continue

        if time_parsed.value is None or conc_parsed.value is None:
            if _later_rows_have_numeric_data(data_rows, offset + 1, col_map, prefer_comma_decimal):
                # A stray bad value in the middle of otherwise-usable
                # data - drop just this row.
                report.rows_dropped_non_numeric += 1
                report.notes.append(
                    f"Row {row_number}: could not parse as numeric "
                    f"(time='{time_raw}', concentration='{conc_raw}') - dropped."
                )
                continue
            else:
                # Nothing parseable remains after this point - treat
                # this row and everything after it as trailing
                # notes/footer text, not data.
                report.rows_dropped_footer += len(data_rows) - offset
                report.notes.append(
                    f"Rows {row_number}-{row_number + (len(data_rows) - offset) - 1}: "
                    f"treated as trailing notes/footer (no further numeric "
                    f"data found) and dropped."
                )
                break

        times.append(time_parsed.value)
        concentrations.append(conc_parsed.value)
        if time_parsed.used_comma_decimal:
            report.decimal_comma_fixes += 1
        if conc_parsed.used_comma_decimal:
            report.decimal_comma_fixes += 1

    report.rows_used = len(times)

    if report.rows_used == 0:
        raise ValueError(
            f"After cleaning, no usable numeric (time, concentration) rows "
            f"remained in '{source_file}'"
            + (f" (sheet '{sheet_name}')" if sheet_name else "") + "."
        )

    return times, concentrations, report


def _later_rows_have_numeric_data(data_rows, start_offset, col_map, prefer_comma_decimal) -> bool:
    """Look ahead from `start_offset` for at least one more row where
    both the time and concentration cells parse as numeric. Used to
    tell a single stray bad value apart from the start of a trailing
    footer block."""
    for row in data_rows[start_offset:]:
        time_raw = row[col_map.time_col] if col_map.time_col < len(row) else ""
        conc_raw = row[col_map.concentration_col] if col_map.concentration_col < len(row) else ""
        if (parse_numeric_cell(time_raw, prefer_comma_decimal).value is not None
                and parse_numeric_cell(conc_raw, prefer_comma_decimal).value is not None):
            return True
    return False
