"""Tests for data_io/numeric_parsing.py"""

from data_io.numeric_parsing import parse_numeric_cell


def test_plain_float_parses():
    result = parse_numeric_cell("3.14")
    assert result.value == 3.14
    assert not result.is_blank
    assert not result.used_comma_decimal


def test_plain_integer_parses():
    result = parse_numeric_cell("10")
    assert result.value == 10.0


def test_negative_number_parses():
    result = parse_numeric_cell("-2.5")
    assert result.value == -2.5


def test_blank_string_is_blank():
    result = parse_numeric_cell("")
    assert result.is_blank
    assert result.value is None


def test_common_na_tokens_are_blank():
    for token in ["N/A", "na", "-", "--", "NULL", "None"]:
        result = parse_numeric_cell(token)
        assert result.is_blank, f"expected {token!r} to be treated as blank"


def test_non_numeric_text_is_not_blank_and_unparseable():
    result = parse_numeric_cell("pending")
    assert not result.is_blank
    assert result.value is None


def test_decimal_comma_short_tail_without_delimiter_hint():
    # No delimiter hint given (defaults to conservative heuristic):
    # short tail after a single comma is treated as a decimal.
    result = parse_numeric_cell("3,14")
    assert result.value == 3.14
    assert result.used_comma_decimal


def test_thousands_comma_without_delimiter_hint():
    # Long tail after a single comma, no delimiter hint -> thousands separator.
    result = parse_numeric_cell("1,234")
    assert result.value == 1234.0
    assert not result.used_comma_decimal


def test_comma_decimal_forced_by_delimiter_hint():
    # When the file's field delimiter is known to be ';' (not ','), a
    # comma inside a value can only be a decimal separator, regardless
    # of how many digits follow - this is the bug case found while
    # testing against a real semicolon-delimited European file.
    result = parse_numeric_cell("6,703", prefer_comma_decimal=True)
    assert result.value == 6.703
    assert result.used_comma_decimal


def test_dot_thousands_comma_decimal():
    result = parse_numeric_cell("1.234,5")
    assert result.value == 1234.5
    assert result.used_comma_decimal


def test_comma_thousands_dot_decimal():
    result = parse_numeric_cell("1,234.5")
    assert result.value == 1234.5
    assert not result.used_comma_decimal


def test_stray_unit_suffix_is_stripped():
    result = parse_numeric_cell("10.5 mM")
    assert result.value == 10.5
    assert result.stripped_suffix == "mM"


def test_stray_unit_suffix_no_space():
    result = parse_numeric_cell("3.2mol/L")
    assert result.value == 3.2
    assert result.stripped_suffix == "mol/L"
