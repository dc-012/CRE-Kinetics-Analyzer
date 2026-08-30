"""Tests for app/formatting.py - pure text/number formatting helpers."""

from app import formatting


def test_format_number_none_is_na():
    assert formatting.format_number(None) == "N/A"


def test_format_number_zero():
    assert formatting.format_number(0.0) == "0"


def test_format_number_typical_value():
    # 4 sig figs by default
    assert formatting.format_number(0.0031415926) == "0.003142"


def test_format_number_with_suffix():
    assert formatting.format_number(12.3456, suffix=" min").startswith("12.35")
    assert formatting.format_number(12.3456, suffix=" min").endswith(" min")


def test_format_reaction_order():
    assert formatting.format_reaction_order(1.9998) == "2.000"
    assert formatting.format_reaction_order(0.5) == "0.500"


def test_format_percent():
    assert formatting.format_percent(0.5) == "50%"
    # default is 3 significant figures, not 2 decimal places
    assert formatting.format_percent(0.8765) == "87.6%"
    assert formatting.format_percent(0.8765, sig_figs=4) == "87.65%"


def test_format_r_squared():
    assert formatting.format_r_squared(0.999812345) == "0.99981"


def test_fit_quality_label_thresholds():
    assert formatting.fit_quality_label(0.999) == "Excellent fit"
    assert formatting.fit_quality_label(0.985) == "Good fit"
    assert formatting.fit_quality_label(0.92) == "Fair fit - inspect the graphs"
    assert formatting.fit_quality_label(0.5) == "Poor fit - reaction may not follow a single nth-order law"
