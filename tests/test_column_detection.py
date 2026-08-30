"""Tests for data_io/column_detection.py and data_io/units.py"""

import pytest

from data_io.column_detection import find_header_and_columns
from data_io.units import split_name_and_unit, looks_like_units_row


def test_simple_header_found_at_row_zero():
    rows = [
        ["time", "concentration"],
        ["0", "10.0"],
        ["10", "8.0"],
    ]
    col_map = find_header_and_columns(rows)
    assert col_map.header_row_index == 0
    assert col_map.time_col == 0
    assert col_map.concentration_col == 1


def test_header_with_units_in_parentheses():
    rows = [
        ["Time (min)", "Concentration (mol/L)"],
        ["0", "10.0"],
    ]
    col_map = find_header_and_columns(rows)
    assert col_map.time_unit == "min"
    assert col_map.concentration_unit == "mol/L"
    assert col_map.time_col_label == "Time (min)"


def test_reordered_columns_detected_by_name_not_position():
    rows = [
        ["Concentration", "Time"],
        ["10.0", "0"],
    ]
    col_map = find_header_and_columns(rows)
    assert col_map.concentration_col == 0
    assert col_map.time_col == 1


def test_bracket_notation_recognized_as_concentration():
    rows = [
        ["t", "[A]"],
        ["0", "10.0"],
    ]
    col_map = find_header_and_columns(rows)
    assert col_map.concentration_col == 1


def test_header_row_found_after_title_rows():
    rows = [
        ["Experiment log, run 3"],
        [""],
        ["Time", "Concentration"],
        ["0", "10.0"],
    ]
    col_map = find_header_and_columns(rows)
    assert col_map.header_row_index == 2


def test_no_recognizable_columns_raises():
    rows = [
        ["Run ID", "Temperature", "Pressure"],
        ["1", "25", "1.0"],
    ]
    with pytest.raises(ValueError, match="Could not find a header row"):
        find_header_and_columns(rows)


def test_ca_alias_recognized_for_concentration():
    rows = [["t", "CA"], ["0", "10"]]
    col_map = find_header_and_columns(rows)
    assert col_map.concentration_col == 1


def test_split_name_and_unit_parentheses():
    name, unit = split_name_and_unit("Time (min)")
    assert name == "Time"
    assert unit == "min"


def test_split_name_and_unit_brackets():
    name, unit = split_name_and_unit("Concentration [mol/L]")
    assert name == "Concentration"
    assert unit == "mol/L"


def test_split_name_and_unit_comma_separated():
    name, unit = split_name_and_unit("Time, s")
    assert name == "Time"
    assert unit == "s"


def test_split_name_and_unit_no_unit():
    name, unit = split_name_and_unit("Concentration")
    assert name == "Concentration"
    assert unit is None


def test_looks_like_units_row_true():
    assert looks_like_units_row(["(min)", "(mol/L)"])


def test_looks_like_units_row_false_for_data():
    assert not looks_like_units_row(["0", "10.0"])
