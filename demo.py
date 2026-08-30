"""
demo.py
=======

A minimal, readable example of using the Stage 1 calculation engine
directly from Python - no UI, no file I/O beyond reading one sample
CSV. Run with:

    python demo.py

This is a good starting point to study how the pieces
(validation -> model_selection -> calculations) fit together.
"""

import csv
from core.calculations import analyze_kinetics


def load_csv(path):
    """Minimal CSV reader: expects a header row, then time,concentration."""
    time, concentration = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            time.append(float(row[0]))
            concentration.append(float(row[1]))
    return time, concentration


def print_result(result):
    print(f"  Reaction order (n)      : {result.reaction_order:.4f}")
    print(f"  Rate constant (k)       : {result.rate_constant:.6g} [{result.rate_constant_units}]")
    print(f"  R^2 (goodness of fit)   : {result.r_squared:.5f}")
    if result.half_life is not None:
        print(f"  Half-life               : {result.half_life:.4g}")
    else:
        print(f"  Half-life               : undefined for this order/data")
    if result.time_for_complete_conversion is not None:
        print(f"  Time to complete conv.  : {result.time_for_complete_conversion:.4g}")
    print(f"  Rate at C_AO            : {result.rate_at_CA0:.6g}")
    print(f"  Final conversion X_A    : {result.conversion[-1]:.4f}")
    if result.warnings:
        print("  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")


if __name__ == "__main__":
    # Example straight from the project brief
    print("=== Example from the project brief ===")
    time = [0, 20, 40, 60, 120, 180, 300]
    concentration = [10, 8, 6, 5, 3, 2, 1]
    result = analyze_kinetics(time, concentration)
    print_result(result)

    # Example loading from a sample CSV file
    print("\n=== sample_data/second_order.csv ===")
    time, concentration = load_csv("sample_data/second_order.csv")
    result = analyze_kinetics(time, concentration)
    print_result(result)
