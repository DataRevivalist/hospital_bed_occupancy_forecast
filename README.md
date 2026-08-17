# Predictive Bed Demand Forecasting - Albion Care Network

AI-powered hospital bed demand forecasting system covering daily, weekly, and hourly horizons, with scenario simulation, explainability, and time-based filtering.

**[Live Demo](https://hospital-bed-occupancy-forecast.streamlit.app/)** · Built with Python, LightGBM, SHAP, and Streamlit

---

## Overview

Albion Care Network's bed capacity was being managed reactively, using historical averages, with admissions, occupancy, staffing, and surgery data held in separate systems. This project delivers a forecasting pipeline and interactive application that predicts ward-level bed demand ahead of time, explains *why* each forecast looks the way it does, and lets planners stress-test the network against operational scenarios like flu outbreaks or staffing shortages.

**What it does:**
- Forecasts daily and weekly bed occupancy by hospital, ward, and bed type
- Flags wards approaching capacity using 85%/90% bottleneck alert thresholds
- Explains individual forecasts with SHAP, showing which features are driving each prediction
- Simulates four operational stress scenarios with an adjustable intensity slider
- Monitors for model drift and supports scheduled retraining

---

## App Features

| Feature | Description |
|---|---|
| **Hospital / ward / bed-type selection** | Populated dynamically from the underlying data |
| **Daily & weekly occupancy forecasts** | With an alert banner triggered by the 85%/90% bottleneck thresholds identified during EDA |
| **Scenario simulator** | Four validated scenarios (flu outbreak, emergency admission spike, delayed discharges, staffing shortage) with an adjustable intensity slider, applied globally across KPIs, alerts, trend charts, and hourly views |
| **72-hour window & time-of-day filters** | Morning / Afternoon / Evening filtering on a dedicated KPI row (avg arrivals/hour, total arrivals, busiest period), driven by hourly A&E arrivals data are kept separate from the daily occupancy KPIs since the daily data has no hourly resolution |
| **SHAP explainability panel** | Shows which features are driving the forecast for the currently selected ward |
| **Inverse-color alerts** | Rising forecasts that push a ward toward its bottleneck threshold are shown in red, not the default green-for-positive |

---

## Methodology

| Notebook | Focus | Key Findings |
|---|---|---|
| 01 - Data Cleaning | Resolved inconsistent categorical text, duplicate records, and impossible timestamps across six raw datasets | |
| 02 - EDA | Explored seasonality and bottlenecks | Strong shared winter seasonality across occupancy, admissions, cancellations, and staffing; Orthopaedics identified as the true bottleneck ward by *time spent above threshold*, not average occupancy |
| 03 - Feature Engineering | Built a leakage-checked daily feature panel plus a supplementary hourly A&E arrivals panel | |
| 04 - Model Development | Trained and validated eight forecasting approaches | |
| 05 - Model Evaluation | Compared models on RMSE, MAE, MAPE, SMAPE, and R² | LightGBM selected for production over a marginally stronger XGBoost, for faster training, native categorical handling, and a smaller file size |
| 06 - Scenario Simulation | Simulated flu outbreak, delayed discharge, emergency spike, and staffing shortage scenarios | Staffing shortage alone does not drive the model's forecast which is an important usage caveat, documented in the model card |
| 07 - Deployment | Shipped the Streamlit app, alerts, hourly filtering, and a continuous learning pipeline | |

---

## Project Structure

```
hospital_bed_occupancy_forecast/
├── app/                          # Streamlit application
├── data/                         # Processed data (auto-located relative to app/)
├── models/                       # Trained model artefacts
├── monitoring/                   # Drift detection & retraining pipeline
├── notebooks/                    # Notebooks 01-07 (cleaning through deployment)
├── model_card.md                 # Model performance & full limitations list
├── Project_Documentation_Report.docx  # Full technical write-up and design rationale
└── README.md
```

---

## Running Locally

The app locates `data/processed/` and `models/` automatically, relative to its own file location — it must stay inside an `app/` folder that sits directly under the same repository root as `data/` and `models/`. If a required file is missing, the app shows exactly which file and path it expected, rather than crashing.

```bash
streamlit run app/app.py
```

### Deploying on Streamlit Community Cloud

1. Push the full repository to GitHub (`data/` and `models/` included, all files are well under GitHub's size limits)
2. On [share.streamlit.io](https://share.streamlit.io), create a new app, select this repository and branch, and set the main file path to `app/app.py`
3. Streamlit Cloud installs from `app/requirements.txt` automatically

### Running the Monitoring Pipeline

See the docstrings in `monitoring/bed_demand_monitoring.py` for `monitor_performance`, `check_for_drift`, and `retrain_daily_model` designed to be called from a scheduler (e.g. a monthly cron job, or triggered on drift detection).

---

## Known Limitations

This model should **not** be used alone to forecast the bed-capacity impact of a staffing shortage (see Notebook 06, Section 8). See `model_card.md` for the full list of limitations.

---

## Further Documentation

- [`model_card.md`](./model_card.md) - model performance and full limitations list
- [`Project_Documentation_Report.docx`](./Project_Documentation_Report.docx) — complete technical account of every notebook and the deployed application, including justification for every major design, technology, and methodological decision

---

## Tech Stack

Python · LightGBM · XGBoost · SHAP · Streamlit · Pandas
