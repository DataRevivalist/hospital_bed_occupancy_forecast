# Albion Care Network: Predictive Bed Demand Forecasting

AI-powered bed demand forecasting for Albion Care Network, covering daily, weekly, and hourly
horizons, with scenario simulation and explainability.

## Project Structure

```
notebooks/    01-07: the full data science lifecycle, from raw data to deployment
data/         raw and processed datasets (see notebooks/01 for the data dictionary)
models/       trained model artefacts (LightGBM daily/weekly/hourly, plus SARIMAX/Prophet/
              LSTM/Random Forest comparison models from Notebook 04)
monitoring/   the continuous learning pipeline (bed_demand_monitoring.py)
app/          the deployed Streamlit application (self-contained)
```

## Running the App

```
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Running the Monitoring Pipeline

```
python monitoring/bed_demand_monitoring.py
```

See `monitoring/bed_demand_monitoring.py` docstrings for `monitor_performance`, `check_for_drift`, and
`retrain_daily_model`, which are designed to be called from a scheduler (e.g. a monthly cron
job, or triggered whenever drift is detected).

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
  five metrics (RMSE, MAE, MAPE, SMAPE, R2).
- **06 (Scenario Simulation):** simulated flu outbreak, delayed discharge, emergency spike, and
  staffing shortage scenarios; found staffing shortage alone does not drive the model's
  forecast, an important usage caveat.
- **07 (Deployment):** shipped the Streamlit app, operational alerts, and a continuous
  learning pipeline.

## Known Limitations

See `model_card.md` for the full list. The most important: this model should not be used alone
to forecast the bed-capacity impact of a staffing shortage (see Notebook 06 Section 8).
