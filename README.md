# CRE Reaction Kinetics Analyzer

A Streamlit application for analyzing chemical reaction kinetics data
from a constant-volume batch reactor. Given experimental
concentration-vs-time data, it determines the reaction order `n`, the
rate constant `k`, and derived quantities (rate of reaction, half-life,
conversion, R²) for the general nth-order rate law:

```
-r_A = -dC_A/dt = k * C_A^n
```

(Levenspiel, *Chemical Reaction Engineering*, Ch. 3.)

## What it does

- **Data input** — upload a CSV/Excel file, enter data manually, or
  load a bundled sample dataset.
- **Data cleaning** — automatically handles messy real-world files:
  extra header rows, a units row, differently-named/ordered columns,
  blank rows, footer notes, stray non-numeric values, and mixed
  decimal separators.
- **Kinetic analysis** — fits the general nth-order model (continuous
  order search, not limited to integer orders), reporting `n`, `k`,
  R², half-life, time for complete conversion, rate of reaction, and
  conversion at every data point.
- **Graphs** — concentration-vs-time with the fitted model curve, and
  the linearized ("straight-line test") plot for the fitted order.
- **AI interpretation (optional)** — if an OpenAI API key is
  configured, a plain-English explanation of the *already-calculated*
  results (order, k, half-life, rate, conversion, fit quality). All
  numbers are computed in Python; OpenAI only explains them and can
  never change a result. The app works normally without a key.
- **Excel report** — a downloadable `.xlsx` workbook with four sheets:
  Input Data, Processed Data, Kinetic Results, and Final Results.

## Project structure

```
core/           Calculation engine (validation, kinetics equations,
                order/rate-constant fitting, orchestration)
data_io/        File reading and cleaning (CSV/Excel -> clean data)
app/            Streamlit UI, plots, formatting, Excel report export,
                AI interpretation
tests/          Full test suite (unit tests for every module above)
sample_data/    Bundled example datasets (zero/first/second/higher
                order, noisy data)
sample_data_messy/  Example messy files used to test data_io cleaning
```

## Installation

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app/streamlit_app.py
```

Run this from the project root so `core` and `data_io` import
correctly. The app opens in your browser (default
`http://localhost:8501`).

## OpenAI API key setup (optional)

The AI interpretation section is optional; the rest of the app works
fully without it.

```bash
export OPENAI_API_KEY="sk-..."          # macOS/Linux
setx OPENAI_API_KEY "sk-..."            # Windows (new terminal after)
```

If the key isn't set, that section simply shows a short notice and
everything else in the app (data input, cleaning, kinetic analysis,
graphs, Excel report) continues to work normally.

## Running tests

```bash
pytest
```

All 147 tests pass at the time of this final packaging step.

## How the calculation engine works (brief)

1. **`core/validation.py`** — sanity-checks raw `(time, concentration)`
   data (matching lengths, no NaN/negative values, strictly increasing
   time, net decrease in concentration, etc.).
2. **`core/kinetics.py`** — the pure nth-order equations (integrated
   concentration law, rate of reaction, half-life, time for complete
   conversion, conversion), derived directly from
   `-r_A = k*C_A^n` by calculus. No fitting or I/O.
3. **`core/model_selection.py`** — scans candidate reaction orders and
   picks the one (with its fitted `k`) that best matches the data.
4. **`core/calculations.py`** — `analyze_kinetics(time, concentration)`
   orchestrates the above into one `KineticsResult` object; this is
   the single entry point the UI, the Excel report, and the AI
   interpretation section all read from — no calculation is ever
   duplicated outside this module.
5. **`data_io/pipeline.py`** — `load_kinetics_data(path)` cleans a
   messy CSV/Excel file into the same `(time, concentration)` shape,
   then hands off to `analyze_kinetics` unchanged.

## Known issue fixed in this final packaging step

`core.kinetics.time_for_complete_conversion` previously checked
`n >= 1.0` literally to decide whether "time for complete conversion"
is defined. For a fitted order extremely close to but not exactly 1.0
(e.g. `n = 0.99999997`, which happens with the bundled
`first_order.csv` sample), this returned a huge, physically
meaningless number instead of `None`/"N/A". It now uses the same
`is_first_order()` tolerance as the rest of `kinetics.py`. A
regression test (`test_time_for_complete_conversion_none_for_n_extremely_close_to_1`)
was added alongside the fix.

## Notes on this final packaging step

- No existing calculation logic in `core/` or `data_io/` was changed
  other than the one-line fix above.
- The Excel report (`app/report_export.py`) and AI interpretation
  (`app/ai_interpretation.py`) modules read only values already
  produced by `core.calculations.analyze_kinetics` — neither performs
  or duplicates any kinetics math.
- This sandbox environment had no network access, so the real
  `streamlit`/`plotly`/`pytest`/`openai` packages could not be
  installed here to run a live server. The app was instead verified
  by executing the real `app/streamlit_app.py` end-to-end against
  small hand-written stand-ins for those APIs (exercising real
  `core`/`data_io`/`app` code and a real generated Excel byte stream),
  and the full test suite (147 tests) was run via a pytest-compatible
  shim implementing the fixtures/parametrize/raises features this
  suite uses. Please do run `streamlit run app/streamlit_app.py` for
  real once you have network access, as a final check.
