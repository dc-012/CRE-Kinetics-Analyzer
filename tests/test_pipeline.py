"""
Integration tests for data_io/pipeline.py, run against the realistic
messy sample files in sample_data_messy/. These check actual cleaned
numeric values and cleaning-report counts (not just "it runs"),
following the same testing standard used in Stage 1, and also confirm
that the full analyze_kinetics_file() -> core.calculations hand-off
works end to end.
"""

import numpy as np
import pytest

from data_io.pipeline import load_kinetics_data, analyze_kinetics_file
from data_io.errors import IngestionError
from core.validation import DataValidationError
from core.calculations import KineticsResult


def test_extra_header_and_footer_rows_cleaned():
    time, conc, report = load_kinetics_data("sample_data_messy/extra_header_footer.csv")
    assert time == [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]
    assert conc == [10.0, 7.5, 5.6, 4.2, 3.1, 2.4]
    assert report.rows_dropped_blank == 1
    assert report.rows_dropped_footer == 2
    assert report.time_unit == "min"
    assert report.concentration_unit == "mol/L"


def test_units_row_and_reordered_columns_cleaned():
    time, conc, report = load_kinetics_data("sample_data_messy/units_row_reordered.csv")
    assert time == [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    assert conc == [10.0, 7.2, 5.3, 3.9, 2.9, 2.1]
    assert report.units_row_skipped is True
    # concentration was the first column in the file, time the second
    assert report.concentration_column_label == "Concentration"
    assert report.time_column_label == "Time"


def test_decimal_comma_and_semicolon_delimiter_cleaned():
    time, conc, report = load_kinetics_data("sample_data_messy/decimal_comma_semicolon.csv")
    assert time == [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
    assert np.allclose(conc, [10.0, 6.703, 4.493, 3.012, 2.02, 1.353])
    assert report.decimal_comma_fixes == 6


def test_stray_values_and_blank_rows_cleaned():
    time, conc, report = load_kinetics_data("sample_data_messy/stray_values_and_blanks.csv")
    assert time == [0.0, 20.0, 60.0, 80.0, 100.0]
    assert conc == [10.0, 7.5, 4.2, 3.1, 2.4]
    assert report.rows_dropped_blank == 2
    assert report.rows_dropped_non_numeric == 1


def test_mixed_case_and_alias_headers_cleaned():
    time, conc, report = load_kinetics_data("sample_data_messy/mixed_case_and_alias_headers.csv")
    assert time == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert conc == [10.0, 8.2, 6.7, 5.5, 4.4]


def test_excel_multi_sheet_uses_the_data_sheet_not_the_notes_sheet():
    time, conc, report = load_kinetics_data("sample_data_messy/multi_sheet_reordered.xlsx")
    assert time == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert conc == [10.0, 7.4, 5.6, 4.2, 3.2, 2.4]
    assert report.sheet_name == "Run1 Data"
    assert report.rows_dropped_footer == 1


def test_excel_units_row_skipped():
    time, conc, report = load_kinetics_data("sample_data_messy/clean_simple.xlsx")
    assert time == [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
    assert conc == [50.0, 41.2, 34.6, 28.9, 24.1, 20.5]
    assert report.units_row_skipped is True


def test_missing_time_or_concentration_column_raises_ingestion_error():
    with pytest.raises(IngestionError, match="Could not extract usable"):
        load_kinetics_data("sample_data_messy/uncleanable_missing_column.csv")


def test_all_original_stage1_sample_csvs_still_load_unchanged():
    # Stage 2 must not alter or misread Stage 1's already-clean files.
    import glob
    for path in glob.glob("sample_data/*.csv"):
        if "invalid" in path:
            continue
        time, conc, report = load_kinetics_data(path)
        assert len(time) == len(conc) >= 4
        assert time[0] == 0.0


def test_analyze_kinetics_file_end_to_end_on_messy_file():
    result, report = analyze_kinetics_file("sample_data_messy/extra_header_footer.csv")
    assert isinstance(result, KineticsResult)
    # first-order-ish decay: reaction order should come back near 1
    assert 0.8 < result.reaction_order < 1.3
    assert result.r_squared > 0.99
    assert report.rows_used == 6


def test_analyze_kinetics_file_surfaces_core_validation_error_with_file_context():
    # This file cleans down to only 1 usable row - not enough for
    # core.validation's own "at least 4 points" check. Stage 2 must
    # NOT re-implement that check; it should just let it propagate
    # with file context prepended.
    with pytest.raises(DataValidationError, match="At least 4 data points"):
        analyze_kinetics_file("sample_data_messy/uncleanable_too_few_rows.csv")


def test_analyze_kinetics_file_result_matches_direct_clean_csv_result():
    # Sanity check that going through the messy-file pipeline gives the
    # exact same KineticsResult as calling analyze_kinetics directly on
    # the same underlying (clean) numbers would.
    from core.calculations import analyze_kinetics
    time, conc, _ = load_kinetics_data("sample_data_messy/mixed_case_and_alias_headers.csv")
    direct_result = analyze_kinetics(time, conc)
    piped_result, _ = analyze_kinetics_file("sample_data_messy/mixed_case_and_alias_headers.csv")
    assert np.isclose(direct_result.reaction_order, piped_result.reaction_order)
    assert np.isclose(direct_result.rate_constant, piped_result.rate_constant)
