"""
demo_stage2.py
==============

A minimal, readable example of using the Stage 2 file-ingestion
pipeline on messy real-world files, ending with the same
`core.calculations.analyze_kinetics` result Stage 1's demo.py
produces - the point being that Stage 2 is just a cleaner front door
onto the unchanged Stage 1 engine. Run with:

    python demo_stage2.py
"""

from data_io.pipeline import analyze_kinetics_file
from demo import print_result  # reuse Stage 1's result-printing helper


if __name__ == "__main__":
    messy_files = [
        "sample_data_messy/extra_header_footer.csv",
        "sample_data_messy/units_row_reordered.csv",
        "sample_data_messy/decimal_comma_semicolon.csv",
        "sample_data_messy/stray_values_and_blanks.csv",
        "sample_data_messy/multi_sheet_reordered.xlsx",
    ]

    for path in messy_files:
        print(f"\n=== {path} ===")
        result, report = analyze_kinetics_file(path)
        print("Cleaning report:")
        for line in report.summary().splitlines():
            print(f"  {line}")
        print("Analysis result:")
        print_result(result)
