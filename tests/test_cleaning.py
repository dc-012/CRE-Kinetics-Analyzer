"""Tests for data_io/cleaning.py, using small hand-built raw tables so
each cleaning behavior is isolated and its exact numeric/report output
can be checked."""

import pytest

from data_io.cleaning import clean_table


def test_clean_data_no_messiness():
    rows = [
        ["time", "concentration"],
        ["0", "10.0"],
        ["10", "8.0"],
        ["20", "6.4"],
        ["30", "5.1"],
    ]
    time, conc, report = clean_table(rows, source_file="test.csv")
    assert time == [0.0, 10.0, 20.0, 30.0]
    assert conc == [10.0, 8.0, 6.4, 5.1]
    assert report.rows_used == 4
    assert report.rows_dropped_blank == 0
    assert report.rows_dropped_non_numeric == 0
    assert report.rows_dropped_footer == 0


def test_blank_rows_dropped_and_counted():
    rows = [
        ["time", "concentration"],
        ["0", "10.0"],
        ["", ""],
        ["10", "8.0"],
    ]
    time, conc, report = clean_table(rows, source_file="test.csv")
    assert time == [0.0, 10.0]
    assert report.rows_dropped_blank == 1


def test_units_row_skipped_and_labeled():
    rows = [
        ["Time", "Concentration"],
        ["(min)", "(mol/L)"],
        ["0", "10.0"],
        ["10", "8.0"],
    ]
    time, conc, report = clean_table(rows, source_file="test.csv")
    assert time == [0.0, 10.0]
    assert report.units_row_skipped is True


def test_stray_non_numeric_row_dropped_when_more_data_follows():
    rows = [
        ["time", "concentration"],
        ["0", "10.0"],
        ["10", "pending"],
        ["20", "6.4"],
        ["30", "5.1"],
    ]
    time, conc, report = clean_table(rows, source_file="test.csv")
    # the "pending" row is dropped, but the two valid rows after it are kept
    assert time == [0.0, 20.0, 30.0]
    assert conc == [10.0, 6.4, 5.1]
    assert report.rows_dropped_non_numeric == 1
    assert report.rows_dropped_footer == 0


def test_trailing_footer_dropped_when_no_more_data_follows():
    rows = [
        ["time", "concentration"],
        ["0", "10.0"],
        ["10", "8.0"],
        ["Notes: sample lost after this point"],
        ["Average k = 0.05"],
    ]
    time, conc, report = clean_table(rows, source_file="test.csv")
    assert time == [0.0, 10.0]
    assert report.rows_dropped_footer == 2
    assert report.rows_dropped_non_numeric == 0


def test_decimal_comma_fix_counted():
    rows = [
        ["Time", "Concentration"],
        ["0", "10,0"],
        ["10", "8,0"],
    ]
    time, conc, report = clean_table(rows, source_file="test.csv", delimiter=";")
    assert conc == [10.0, 8.0]
    assert report.decimal_comma_fixes == 2


def test_no_usable_rows_raises():
    rows = [
        ["time", "concentration"],
        ["not a number", "also not a number"],
    ]
    with pytest.raises(ValueError, match="no usable numeric"):
        clean_table(rows, source_file="test.csv")


def test_no_header_found_raises():
    rows = [
        ["Run ID", "Temperature"],
        ["1", "25"],
    ]
    with pytest.raises(ValueError, match="Could not find a header row"):
        clean_table(rows, source_file="test.csv")


def test_report_summary_is_readable_text():
    rows = [
        ["time", "concentration"],
        ["0", "10.0"],
        ["10", "8.0"],
    ]
    _, _, report = clean_table(rows, source_file="test.csv")
    text = report.summary()
    assert "test.csv" in text
    assert "Rows used: 2" in text
