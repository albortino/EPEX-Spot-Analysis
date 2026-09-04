# Contributing to EPEX Spot Analysis

Thank you for your interest in contributing! This document provides instructions for setting up your local environment, understanding the project architecture, adding new smart meter formats, and submitting contributions.

---

## 1. Development Setup

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/albortino/EPEX-Spot-Analysis.git
cd EPEX-Spot-Analysis

# Create and activate the conda environment
conda env create -f environment.yml
conda activate epex-analysis

# Run the application locally
streamlit run app.py
```

### Using venv / pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 2. Architecture Overview

The codebase is designed around a modular structure:

- **Entry Point (`app.py`)**: Initializes application state, handles navigation tabs, and triggers data processing workflows.
- **Presentation (`methods/ui_components.py`, `methods/charts.py`)**: Renders Streamlit widgets, layout tabs, tables, and Plotly interactive visualizations.
- **Business Logic (`methods/analysis.py`, `methods/tariffs.py`)**:
  - `analysis.py`: Load profile decomposition (Base, Regular, Peak via FFT and derivative analysis), peak load shifting simulation, and Prophet forecasting.
  - `tariffs.py`: Tariff definitions (`TariffType.STATIC` / `TariffType.FLEXIBLE`) and cost calculation models.
- **Data & I/O (`methods/file_parser.py`, `methods/data_loader.py`)**:
  - `data_loader.py`: Fetches and caches EPEX spot market price data from the aWATTar API.
  - `file_parser.py`: Auto-detects and parses various European smart meter CSV exports into standardized UTC time series.

---

## 3. Adding Support for a New Smart Meter Format

If your regional network operator (Netzbetreiber) exports a CSV format that is not yet automatically parsed:

1. Add a test CSV sample (anonymized) to `uploads/` or `tests/`.
2. Add a new configuration entry to `cache/additional_provider_formats.json`:
   ```json
   {
       "name": "My Provider Export",
       "usage_col": "!Verbrauch [kWh]",
       "timestamp_col": "Messzeitpunkt",
       "time_sub_col": null,
       "date_format": "%d.%m.%Y %H:%M",
       "other_cols": [],
       "fixup_timestamp": false,
       "separator": ";",
       "decimal": ",",
       "skiprows": 0,
       "encoding": "utf-8-sig",
       "feedin": false,
       "end_timestamp_col": null
   }
   ```
   *Note: Prefix `usage_col` with `!` to enable substring/fuzzy column matching.*
3. Add a test case in `tests/test_file_parser.py`.
4. Run `pytest` to confirm your format parses cleanly.

---

## 4. Testing & Code Quality

Before opening a pull request, verify that all tests pass and code passes lint checks:

```bash
# Run unit and integration tests
PYTHONPATH=. pytest tests/

# Run flake8 syntax and static analysis checks
flake8 methods/ tests/ app.py --count --select=E9,F63,F7,F82 --show-source
```

---

## 5. Pull Request Guidelines

1. Create a feature branch (`git checkout -b feature/my-feature`).
2. Keep changes focused and avoid unnecessary refactoring of unaffected components.
3. Write self-explanatory code with clear docstrings explaining the *why* rather than just the *what*.
4. Ensure CI tests pass on your pull request.
