# Albion Care Network: Predictive Bed Demand Forecasting

AI-powered bed demand forecasting for Albion Care Network, covering daily, weekly, and hourly
horizons, with scenario simulation, explainability, and time-based filtering.

## Project Structure

## Running the App
The app locates `data/processed/` and `models/` automatically, relative to its own file
location, so it must stay inside an `app/` folder that sits directly under the same
repository root as `data/` and `models/`. If a required file is missing, the app will show
exactly which file and which path it expected, rather than crashing (verified by
deliberately testing it against a folder with every data and model file removed).

## App Features

    - **Hospital / ward / bed-type selection**, populated dynamically from the data.
    - **Daily and weekly occupancy forecasts**, with an operational alert banner using the
    85%/90% bottleneck thresholds identified in Notebook 02.
    - **Scenario simulator** for the four scenarios validated in Notebook 06 (flu outbreak,
    emergency admission spike, delayed discharges, staffing shortage), with an adjustable
    intensity slider. The scenario filter applies globally, across the KPI rows, the alert
    banner, the trend chart, and the hourly views, not just one isolated section.
    - **72-hour window and Time of Day filters** (Morning / Afternoon / Evening), applied to a
    second, dedicated KPI row (Avg arrivals/hour, Total arrivals, Busiest period) driven by the
    hourly ED-arrivals data. This is a separate row from the daily occupancy KPIs deliberately:
    the daily bed data has no hourly resolution at all, so only the hourly-derived metrics can
    honestly respond to these two filters.
    - **SHAP-based explanation panel**, showing which features are driving the forecast for the
    currently selected ward.
    - Forecast metrics use Streamlit's `delta_color="inverse"` when a rising forecast is pushing
    a ward toward or past the bottleneck threshold, so an increase that is operationally bad is
    shown in red, not the default green-for-positive.

## Running the Monitoring Pipeline
See `monitoring/bed_demand_monitoring.py` docstrings for `monitor_performance`,
`check_for_drift`, and `retrain_daily_model`, which are designed to be called from a
scheduler (e.g. a monthly cron job, or triggered whenever drift is detected).

## Deploying on Streamlit Community Cloud

1. Push this entire repository to GitHub (data/ and models/ included; every file here is
   well under GitHub's size limits).
2. On share.streamlit.io, create a new app, select this repository and branch, and set the
   main file path to `app/app.py`.
3. Streamlit Cloud installs from `app/requirements.txt` automatically, since it looks in the
   same folder as the main file first.

## Key Findings by Notebook

    - **01 (Data Cleaning):** resolved inconsistent categorical text, duplicate records, and
    impossible timestamps across six raw datasets.
    - **02 (EDA):** found strong shared winter seasonality across occupancy, admissions,
    cancellations, and staffing; identified Orthopaedics wards as the true bottleneck by
    time-above-threshold, not average occupancy.
    - **03 (Feature Engineering):** built a leakage-checked daily feature panel and a
    supplementary hourly ED-arrivals panel.
    - **04 (Model Development):** trained and lightly validated eight forecasting approaches.
    - **05 (Model Evaluation):** selected LightGBM based on held-out test-set performance across
    five metrics (RMSE, MAE, MAPE, SMAPE, R2), despite XGBoost scoring marginally better, for
    practical deployment reasons (faster training, native categorical handling, smaller file).
    - **06 (Scenario Simulation):** simulated flu outbreak, delayed discharge, emergency spike, and
    staffing shortage scenarios; found staffing shortage alone does not drive the model's
    forecast, an important usage caveat.
    - **07 (Deployment):** shipped the Streamlit app, operational alerts, hourly/time-of-day
    filtering, and a continuous learning pipeline, reading directly from the project's existing
    data/ and models/ folders.

## Further Documentation

- `model_card.md`: model performance and full limitations list.
- `Project_Documentation_Report.docx`: a complete technical account of every notebook and the
  deployed application, with justifications for every major design, technology, and
  methodological decision made throughout the project.

## Known Limitations

See `model_card.md` for the full list. The most important: this model should not be used alone
to forecast the bed-capacity impact of a staffing shortage (see Notebook 06 Section 8).
