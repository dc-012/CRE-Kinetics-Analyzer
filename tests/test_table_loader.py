"""Tests for data_io/table_loader.py"""

import pytest

from data_io.table_loader import load_raw_tables
from data_io.errors import IngestionError


def test_missing_file_raises_ingestion_error():
    with pytest.raises(IngestionError, match="File not found"):
        load_raw_tables("sample_data_messy/does_not_exist.csv")


def test_unsupported_extension_raises(tmp_path):
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("time,concentration\n0,10\n")
    with pytest.raises(IngestionError, match="Unsupported file type"):
        load_raw_tables(str(bad_file))


def test_comma_csv_loads_with_comma_delimiter():
    tables = load_raw_tables("sample_data/first_order.csv")
    assert len(tables) == 1
    sheet_name, rows, delimiter = tables[0]
    assert sheet_name == ""
    assert delimiter == ","
    assert rows[0] == ["time", "concentration"]


def test_semicolon_csv_detected():
    tables = load_raw_tables("sample_data_messy/decimal_comma_semicolon.csv")
    _, rows, delimiter = tables[0]
    assert delimiter == ";"
    assert rows[0] == ["Time", "Concentration"]


def test_ragged_rows_with_no_delimiter_do_not_crash():
    # extra_header_footer.csv has preamble/footer lines with no commas
    # at all - a strict CSV parser would raise on the ragged row
    # counts; this loader must tolerate it.
    tables = load_raw_tables("sample_data_messy/extra_header_footer.csv")
    _, rows, _ = tables[0]
    assert len(rows) > 0


def test_excel_multi_sheet_returns_all_sheets():
    tables = load_raw_tables("sample_data_messy/multi_sheet_reordered.xlsx")
    sheet_names = [name for name, _, _ in tables]
    assert "Notes" in sheet_names
    assert "Run1 Data" in sheet_names
    for _, _, delimiter in tables:
        assert delimiter is None
