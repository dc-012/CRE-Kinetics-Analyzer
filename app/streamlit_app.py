"""
streamlit_app.py
=================

The Streamlit UI for the Reaction Kinetics Analyzer (final stage,
Part 1: interface + graphs).

Workflow implemented here, matching the project brief exactly:

    Excel/CSV or manual data
            v
    Stage 2 validation/cleaning   (data_io.pipeline)
            v
    Stage 1 kinetic calculation   (core.calculations.analyze_kinetics)
            v
    Reaction order, k, rate, half-life, conversion, R^2
            v
    Graphs                        (app.plots)
            v
    Final presentation

This file contains NO kinetics math and NO data-cleaning logic of its
own. It only collects input, calls the existing `core` / `data_io`
functions, and displays what they return. See app/__init__.py for how
this module fits with the rest of `app/`.

Run with:
    streamlit run app/streamlit_app.py
(from the project root, so that `core` and `data_io` are importable).
"""

import os
import sys
import tempfile

import pandas as pd
import streamlit as st

# Make sure the project root (parent of this file's directory) is on
# sys.path, so `core` and `data_io` import correctly regardless of the
# working directory Streamlit was launched from.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.calculations import analyze_kinetics, KineticsResult
from core.validation import DataValidationError
from data_io.pipeline import load_kinetics_data
from data_io.cleaning import CleaningReport
from data_io.errors import IngestionError

from app import plots, report_export
from app.formatting import (
    format_number, format_reaction_order, format_percent,
    format_r_squared, fit_quality_label, format_complete_conversion_time,
)

SAMPLE_DATA_DIR = os.path.join(_PROJECT_ROOT, "sample_data")


# --------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------
st.set_page_config(
    page_title="CRE Reaction Kinetics Analyzer",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --bg-base: #050d18;
            --bg-panel: #0d1d2d;
            --ink: #f4f8fd;
            --ink-soft: #d5e2f2;
            --ink-muted: #a7bfdc;
            --accent: #4a86c5;
            --line: rgba(154, 191, 240, 0.18);
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
        .stApp, .stMain {
            background: var(--bg-base) !important;
            color: var(--ink) !important;
        }

        /* Hide Streamlit's own chrome so the custom application header
           starts directly at the top of the page without the extra
           toolbar/header strip. */
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none !important;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        .hero-shell {
            background: #0b1726;
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 1.5rem 1.3rem 1.1rem;
            margin-bottom: 0.9rem;
            min-height: 150px;
            text-align: center;
        }

        .hero-title {
            font-size: clamp(1.2rem, 1.8vw, 1.75rem);
            font-weight: 700;
            letter-spacing: 0.18em;
            line-height: 1.3;
            color: var(--ink);
            margin: 0.2rem 0 0.1rem;
            text-transform: uppercase;
        }

        .hero-main-title {
            font-size: clamp(2rem, 3vw, 3rem);
            font-weight: 800;
            letter-spacing: 0.09em;
            line-height: 1.15;
            color: var(--ink);
            margin: 0.6rem 0 0.2rem;
            text-transform: uppercase;
        }

        .hero-subtitle {
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--ink-muted);
            margin-top: 0.3rem;
        }

        .identity-block {
            display: flex;
            justify-content: flex-end;
            align-items: flex-end;
            margin-top: 0.35rem;
            margin-bottom: 0.75rem;
            width: 100%;
        }

        .identity-card {
            display: inline-block;
            text-align: right;
            padding: 0.2rem 0;
            border: none;
            background: transparent;
            box-shadow: none;
        }

        .signature-name {
            color: var(--ink);
            font-size: 1.08rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            line-height: 1.4;
        }

        .signature-roll {
            color: var(--ink-muted);
            font-size: 0.82rem;
            letter-spacing: 0.1em;
            margin-top: 0.1rem;
            text-transform: uppercase;
        }

        .section-heading {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin: 1.4rem 0 0.9rem;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid var(--line);
            color: var(--ink);
            font-weight: 700;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            font-size: 0.74rem;
        }

        .section-number {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.8rem;
            height: 1.8rem;
            border-radius: 50%;
            background: rgba(74, 134, 197, 0.12);
            border: 1px solid var(--line);
            color: var(--accent);
            font-weight: 700;
            font-size: 0.68rem;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .hero-shell {
                padding: 1.2rem 0.9rem 1rem;
                min-height: 170px;
            }

            .identity-block {
                justify-content: center;
                margin-top: 0.6rem;
            }

            .identity-card {
                text-align: center;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-title">CHEMICAL REACTION ENGINEERING</div>
        <div class="hero-main-title">REACTION KINETICS ANALYZER</div>
        <div class="hero-subtitle">General nth-Order Kinetic Analysis</div>
    </div>
    <div class="identity-block">
        <div class="identity-card">
            <div class="signature-name">DIVYANSH CHOUDHARY</div>
            <div class="signature-roll">2024UCH0015</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------
st.session_state.setdefault("time", None)
st.session_state.setdefault("concentration", None)
st.session_state.setdefault("cleaning_report", None)
st.session_state.setdefault("time_unit", None)
st.session_state.setdefault("concentration_unit", None)
st.session_state.setdefault("result", None)
st.session_state.setdefault("source_label", None)
st.session_state.setdefault("ai_interpretation", None)


def _clear_downstream_state():
    """Whenever the input data changes, any previously computed result
    is stale and must not be shown alongside new/different data."""
    st.session_state["result"] = None
    st.session_state["ai_interpretation"] = None


# --------------------------------------------------------------------
# 1. Data input
# --------------------------------------------------------------------
st.markdown('<div class="section-heading"><span class="section-number">1</span> Data Input</div>', unsafe_allow_html=True)

with st.container():
    input_mode = st.radio(
        "Data source",
        ["Upload Excel/CSV file", "Enter data manually", "Use a bundled sample dataset"],
        horizontal=True,
    )

if input_mode == "Upload Excel/CSV file":
    uploaded_file = st.file_uploader(
        "Upload a concentration-vs-time file (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded_file is not None:
        # data_io works on a file path, so the uploaded bytes are
        # written to a temp file first; no parsing happens here.
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            time, concentration, report = load_kinetics_data(tmp_path)
            st.session_state["time"] = time
            st.session_state["concentration"] = concentration
            st.session_state["cleaning_report"] = report
            st.session_state["time_unit"] = report.time_unit
            st.session_state["concentration_unit"] = report.concentration_unit
            st.session_state["source_label"] = uploaded_file.name
            _clear_downstream_state()
        except IngestionError as exc:
            st.error(f"Could not read this file: {exc}")
        finally:
            os.unlink(tmp_path)

elif input_mode == "Enter data manually":
    st.write("Edit the table below (add/remove rows as needed). "
             "The first time value must be 0.")
    col_units1, col_units2 = st.columns(2)
    with col_units1:
        manual_time_unit = st.text_input("Time unit (optional, for axis labels only)", value="min")
    with col_units2:
        manual_conc_unit = st.text_input("Concentration unit (optional, for axis labels only)", value="mol/L")

    default_table = pd.DataFrame({
        "time": [0.0, 20.0, 40.0, 60.0, 80.0, 100.0],
        "concentration": [10.0, 6.70, 4.49, 3.01, 2.02, 1.35],
    })
    edited = st.data_editor(
        default_table,
        num_rows="dynamic",
        use_container_width=True,
        key="manual_data_editor",
        column_config={
            "time": st.column_config.NumberColumn("time", required=True),
            "concentration": st.column_config.NumberColumn("concentration", required=True),
        },
    )
    # Drop fully-empty rows the data_editor may leave behind, but do
    # NOT do any other cleaning here - that stays Stage 2's job. Raw
    # manual entry has no file-format messiness (headers, units rows,
    # stray text) for Stage 2 to clean, so it is validated directly by
    # core.validation via analyze_kinetics.
    edited_clean = edited.dropna(how="all")
    time_vals = edited_clean["time"].tolist()
    conc_vals = edited_clean["concentration"].tolist()

    if (time_vals, conc_vals) != (
        st.session_state.get("_manual_time_cache"), st.session_state.get("_manual_conc_cache")
    ):
        _clear_downstream_state()
    st.session_state["_manual_time_cache"] = time_vals
    st.session_state["_manual_conc_cache"] = conc_vals

    st.session_state["time"] = time_vals
    st.session_state["concentration"] = conc_vals
    st.session_state["cleaning_report"] = None
    st.session_state["time_unit"] = manual_time_unit.strip() or None
    st.session_state["concentration_unit"] = manual_conc_unit.strip() or None
    st.session_state["source_label"] = "Manual entry"

else:  # bundled sample dataset
    if os.path.isdir(SAMPLE_DATA_DIR):
        sample_files = sorted(f for f in os.listdir(SAMPLE_DATA_DIR) if f.endswith(".csv"))
    else:
        sample_files = []

    if not sample_files:
        st.warning("No bundled sample datasets were found.")
    else:
        chosen = st.selectbox("Sample dataset", sample_files)
        if st.button("Load sample dataset"):
            sample_path = os.path.join(SAMPLE_DATA_DIR, chosen)
            try:
                time, concentration, report = load_kinetics_data(sample_path)
                st.session_state["time"] = time
                st.session_state["concentration"] = concentration
                st.session_state["cleaning_report"] = report
                st.session_state["time_unit"] = report.time_unit
                st.session_state["concentration_unit"] = report.concentration_unit
                st.session_state["source_label"] = chosen
                _clear_downstream_state()
            except IngestionError as exc:
                st.error(f"Could not read this sample file: {exc}")

# --------------------------------------------------------------------
# 2. Data preview + cleaning report
# --------------------------------------------------------------------
have_data = (
    st.session_state["time"] is not None
    and st.session_state["concentration"] is not None
    and len(st.session_state["time"]) > 0
)

if have_data:
    st.markdown('<div class="section-heading"><span class="section-number">2</span> Data Preview</div>', unsafe_allow_html=True)
    st.caption(f"Source: {st.session_state['source_label']}")

    preview_df = pd.DataFrame({
        "time": st.session_state["time"],
        "concentration": st.session_state["concentration"],
    })
    st.dataframe(preview_df, use_container_width=True, height=min(300, 40 + 35 * len(preview_df)))

    report: CleaningReport = st.session_state["cleaning_report"]
    if report is not None:
        with st.expander("Data Processing Details ▸", expanded=False):
            st.text(report.summary())
            if report.notes:
                for note in report.notes:
                    st.caption(f"Note: {note}")

# --------------------------------------------------------------------
# 3. Run analysis
# --------------------------------------------------------------------
st.markdown('<div class="section-heading"><span class="section-number">3</span> Run Analysis</div>', unsafe_allow_html=True)

run_clicked = st.button("Run Kinetic Analysis", type="primary", disabled=not have_data)

if run_clicked:
    try:
        result = analyze_kinetics(
            st.session_state["time"], st.session_state["concentration"]
        )
        st.session_state["result"] = result
        st.session_state["ai_interpretation"] = None
    except DataValidationError as exc:
        st.session_state["result"] = None
        st.error(f"Data validation failed: {exc}")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        st.session_state["result"] = None
        st.error(f"Unexpected error while running the analysis: {exc}")

if not have_data:
    st.info("Provide data above (upload a file, enter it manually, or load a sample) to enable analysis.")

# --------------------------------------------------------------------
# 4. Results
# --------------------------------------------------------------------
result: KineticsResult = st.session_state["result"]

if result is not None:
    st.markdown('<div class="section-heading"><span class="section-number">4</span> Results</div>', unsafe_allow_html=True)

    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    time_unit = st.session_state["time_unit"]
    conc_unit = st.session_state["concentration_unit"]

    row1 = st.columns(4)
    for idx, (label, value, extra_class) in enumerate([
        ("Reaction Order (n)", format_reaction_order(result.reaction_order), "metric-value--accent"),
        ("Rate Constant (k)", format_number(result.rate_constant), "metric-value--warm"),
        ("R² (goodness of fit)", format_r_squared(result.r_squared), ""),
        ("Fit Quality", fit_quality_label(result.r_squared), ""),
    ]):
        with row1[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {extra_class}">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.caption(f"Rate constant units: {result.rate_constant_units}")

    row2 = st.columns(4)
    for idx, (label, value, extra_class) in enumerate([
        ("Initial Concentration (C_A0)", format_number(result.CA0, suffix=f" {conc_unit}" if conc_unit else ""), ""),
        ("Final Concentration (C_A, last point)", format_number(float(result.concentration[-1]), suffix=f" {conc_unit}" if conc_unit else ""), ""),
        ("Conversion (final data point)", format_percent(float(result.conversion[-1])), "metric-value--warm"),
        ("Rate of Reaction (at C_A0)", format_number(result.rate_at_CA0), ""),
    ]):
        with row2[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {extra_class}">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    row3 = st.columns(2)
    for idx, (label, value, extra_class) in enumerate([
        ("Half-life (t_1/2)", format_number(result.half_life, suffix=f" {time_unit}" if (result.half_life is not None and time_unit) else ""), "metric-value--accent"),
        (
            "Time for Complete Conversion",
            format_complete_conversion_time(
                result.time_for_complete_conversion,
                reaction_order=result.reaction_order,
                suffix=f" {time_unit}" if (result.time_for_complete_conversion is not None and time_unit) else "",
            ),
            "metric-value--warm",
        ),
    ]):
        with row3[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value {extra_class}">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ----------------------------------------------------------------
    # 5. Graphs
    # ----------------------------------------------------------------
    st.markdown('<div class="section-heading"><span class="section-number">5</span> Graphical Analysis</div>', unsafe_allow_html=True)

    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.plotly_chart(
            plots.concentration_vs_time_figure(result, time_unit=time_unit, concentration_unit=conc_unit),
            use_container_width=True,
        )
    with fig_col2:
        st.plotly_chart(
            plots.linearized_kinetic_figure(result, time_unit=time_unit, concentration_unit=conc_unit),
            use_container_width=True,
        )

    st.caption(
        "Left: experimental concentration data with the fitted nth-order model curve. "
        "Right: the same data and model in the linearized ('straight-line test') space "
        "for the fitted order - a good fit should look close to a straight line here."
    )

    # ----------------------------------------------------------------
    # 6. Downloadable Excel report
    # ----------------------------------------------------------------
    st.markdown('<div class="section-heading"><span class="section-number">6</span> Download / Report</div>', unsafe_allow_html=True)
    st.caption(
        "An Excel workbook with the input data, processed per-point data, "
        "kinetic results, and final results - all read directly from the "
        "analysis above."
    )

    excel_bytes = report_export.build_excel_report(
        result,
        source_label=st.session_state["source_label"],
        time_unit=time_unit,
        concentration_unit=conc_unit,
        ai_interpretation=st.session_state["ai_interpretation"],
    )
    st.download_button(
        "Download Excel Report (.xlsx)",
        data=excel_bytes,
        file_name="reaction_kinetics_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.markdown(
        '<div class="footer-note">CRE Reaction Kinetics Analyzer<br>Built for Chemical Reaction Engineering Analysis</div>',
        unsafe_allow_html=True,
    )
