"""
app
===

Final-stage UI layer (Streamlit) for the Reaction Kinetics Analyzer.

This package contains ONLY presentation code: reading user input,
calling the existing `core` (Stage 1) and `data_io` (Stage 2) engines,
and rendering the results. It performs no kinetics math and no data
cleaning of its own - every number shown in the UI is produced by
`core.calculations.analyze_kinetics` (or `data_io.pipeline`, which
itself just calls that same function after cleaning a file).

Modules
-------
plot_logic.py
    Pure, framework-agnostic helper functions that prepare the data
    for the graphs (dense model curves, linearized-plot transforms).
    No Streamlit or Plotly import here, so this module can be unit
    tested on its own.
formatting.py
    Small, pure text/number formatting helpers used by the UI (e.g.
    turning `None` half-lives into a readable "N/A", picking a sane
    number of significant figures for display).
plots.py
    Builds the two Plotly figures from a `KineticsResult`, using
    `plot_logic.py` for the underlying numbers.
streamlit_app.py
    The actual Streamlit page: file upload / manual entry, data
    preview, "Run Analysis" button, results display, graphs.
"""
