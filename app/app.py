"""
Albion Care Network - Predictive Bed Demand Forecasting
Streamlit operational dashboard.

Run with: streamlit run app.py

Folder layout this file expects (repo root is the parent of this app/ folder):

    hospital_bed_occupancy_forecast/     <- repo root
      app/
        app.py                          <- this file
        requirements.txt
      data/
        processed/
          daily_feature_panel.parquet
          hospital_reference_clean.parquet
          hourly_ed_arrivals_panel.parquet
      models/
        lightgbm_daily_tuned.txt
        lightgbm_weekly.txt
        lightgbm_hourly_ed.txt
        shap_explainer_background.pkl

There is deliberately no separate copy of data/models inside app/ itself.
This file reads directly from the single, real data/ and models/ folders
that Notebooks 01-06 already produce at the repo root, so there is only
ever one copy of each file to keep in sync.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Albion Care Network - Bed Demand Forecasting",
    page_icon=None,
    layout="wide",
)

# This file lives at <repo_root>/app/app.py, so its own parent is app/ and
# the parent of that is the repo root. Resolving to an absolute path first
# means this works correctly regardless of the working directory the app
# happens to be launched from.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models"

REQUIRED_DATA_FILES = [
    "daily_feature_panel.parquet",
    "hospital_reference_clean.parquet",
    "hourly_ed_arrivals_panel.parquet",
]
REQUIRED_MODEL_FILES = [
    "lightgbm_daily_tuned.txt",
    "lightgbm_weekly.txt",
    "lightgbm_hourly_ed.txt",
    "shap_explainer_background.pkl",
]


def check_required_files():
    """Check every file this app needs before trying to load any of them, and show
    a clear, specific error naming exactly what is missing and where it was expected,
    rather than letting a bare FileNotFoundError and a raw Python traceback surface
    from deep inside a caching decorator. This turns a deployment misconfiguration
    (wrong folder structure, a file that never made it into the repo) into something
    fixable in seconds instead of a confusing stack trace."""
    missing = []
    for fname in REQUIRED_DATA_FILES:
        path = DATA_DIR / fname
        if not path.is_file():
            missing.append(str(path))
    for fname in REQUIRED_MODEL_FILES:
        path = MODEL_DIR / fname
        if not path.is_file():
            missing.append(str(path))

    if missing:
        st.error(
            "This app cannot start because required data/model files are missing.\n\n"
            "Expected repo layout:\n"
            "```\n"
            f"{REPO_ROOT.name}/\n"
            "  app/app.py          <- this file\n"
            "  data/processed/     <- daily_feature_panel.parquet and 2 others\n"
            "  models/             <- lightgbm_daily_tuned.txt and 3 others\n"
            "```\n\n"
            "Missing file(s):\n" + "\n".join(f"- `{p}`" for p in missing)
        )
        st.stop()


check_required_files()

EXCLUDE_COLS = ['hospital_id', 'ward', 'bed_type', 'date', 'target_next_day_occupied',
                'target_next_week_occupied', 'split', 'median_los_hours']
CAT_COLS = ['hospital_id', 'ward', 'bed_type']

# Bottleneck thresholds established in Notebook 02 (Section 7.1) and used consistently
# throughout the project for identifying wards under capacity pressure.
WARNING_THRESHOLD = 0.85
CRITICAL_THRESHOLD = 0.90

# Winter surge ratios measured directly from the data in Notebook 02/06 (emergency
# admissions and ED arrivals run at roughly this multiple of their summer level in winter).
FLU_OUTBREAK_MULTIPLIER = 1.6
EMERGENCY_SPIKE_MULTIPLIER = 2.2
DELAYED_DISCHARGE_LOS_MULTIPLIER = 1.3
STAFFING_SHORTAGE_RATE = 0.70


# -----------------------------------------------------------------------------
# Cached data and model loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_daily_panel():
    df = pd.read_parquet(DATA_DIR / "daily_feature_panel.parquet")
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    for c in feature_cols:
        if 'pyarrow' in str(df[c].dtype):
            df[c] = df[c].astype('float64')
    return df, feature_cols


@st.cache_data
def load_hospital_reference():
    return pd.read_parquet(DATA_DIR / "hospital_reference_clean.parquet")


@st.cache_resource
def load_daily_model():
    return lgb.Booster(model_file=str(MODEL_DIR / "lightgbm_daily_tuned.txt"))


@st.cache_resource
def load_weekly_model():
    return lgb.Booster(model_file=str(MODEL_DIR / "lightgbm_weekly.txt"))


@st.cache_resource
def load_shap_explainer(_booster):
    # TreeExplainer needs no background sample for tree models; it reads the model
    # structure directly, which is why gradient boosting models can be explained quickly.
    return shap.TreeExplainer(_booster)


def predict(booster, df, feature_cols):
    X = df[feature_cols + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype('category')
    return booster.predict(X)


# -----------------------------------------------------------------------------
# Scenario simulation (same framework validated in Notebook 06)
# -----------------------------------------------------------------------------
def apply_scenario(baseline_row, scenario, intensity):
    """Return a perturbed copy of a single baseline row for the given scenario.
    `intensity` is a 0-2 multiplier on the default effect size, so the app user can dial
    a scenario up or down from the Notebook 06 reference magnitude (intensity = 1.0)."""
    row = baseline_row.copy()

    if scenario == "Flu outbreak (winter emergency surge)":
        mult = 1 + (FLU_OUTBREAK_MULTIPLIER - 1) * intensity
        row['n_emergency_admissions'] = baseline_row['n_emergency_admissions'] * mult
        row['n_admissions'] = (baseline_row['n_admissions'] +
                                (row['n_emergency_admissions'] - baseline_row['n_emergency_admissions']))
        row['ed_arrivals'] = baseline_row['ed_arrivals'] * mult
        row['ed_high_acuity_arrivals'] = baseline_row['ed_high_acuity_arrivals'] * mult
        partial_mult = 1 + 0.3 * intensity
        row['n_emergency_admissions_roll7'] = baseline_row['n_emergency_admissions_roll7'] * partial_mult
        row['ed_arrivals_roll7'] = baseline_row['ed_arrivals_roll7'] * partial_mult

    elif scenario == "Delayed discharges":
        los_mult = 1 + (DELAYED_DISCHARGE_LOS_MULTIPLIER - 1) * intensity
        occ_bump = 1 + 0.08 * intensity
        row['median_los_roll30'] = baseline_row['median_los_roll30'] * los_mult
        row['occupied_lag_1'] = baseline_row['occupied_lag_1'] * occ_bump
        row['occupied_roll_mean_7'] = baseline_row['occupied_roll_mean_7'] * occ_bump
        row['occupied_beds'] = baseline_row['occupied_beds'] * occ_bump
        row['occupancy_rate'] = row['occupied_beds'] / row['staffed_beds']
        row['available_bed_ratio'] = 1 - row['occupancy_rate']

    elif scenario == "Emergency admission spike":
        mult = 1 + (EMERGENCY_SPIKE_MULTIPLIER - 1) * intensity
        row['n_emergency_admissions'] = baseline_row['n_emergency_admissions'] * mult
        row['n_admissions'] = (baseline_row['n_admissions'] +
                                (row['n_emergency_admissions'] - baseline_row['n_emergency_admissions']))
        row['ed_arrivals'] = baseline_row['ed_arrivals'] * mult
        row['ed_high_acuity_arrivals'] = baseline_row['ed_high_acuity_arrivals'] * mult

    elif scenario == "Staffing shortage":
        target_rate = 1.0 - (1.0 - STAFFING_SHORTAGE_RATE) * intensity
        row['safe_staffing_rate_lag1'] = target_rate
        row['safe_staffing_rate_roll7_lag1'] = target_rate

    return row


# -----------------------------------------------------------------------------
# Load everything
# -----------------------------------------------------------------------------
daily, FEATURE_COLS = load_daily_panel()
hospital_ref = load_hospital_reference()
daily_model = load_daily_model()
weekly_model = load_weekly_model()
explainer = load_shap_explainer(daily_model)

LATEST_DATE = daily['date'].max()

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.title("Albion Care Network")
st.sidebar.caption("Predictive Bed Demand Forecasting")

hospital_options = (
    daily[['hospital_id']].drop_duplicates()
    .merge(hospital_ref[['hospital_id', 'hospital_name']], on='hospital_id', how='left')
)
hospital_label_map = dict(zip(hospital_options['hospital_name'], hospital_options['hospital_id']))
selected_hospital_name = st.sidebar.selectbox("Hospital", sorted(hospital_label_map.keys()))
selected_hospital_id = hospital_label_map[selected_hospital_name]

ward_options = sorted(daily.loc[daily['hospital_id'] == selected_hospital_id, 'ward'].unique())
selected_ward = st.sidebar.selectbox("Ward", ward_options)

bed_type_options = sorted(daily.loc[
    (daily['hospital_id'] == selected_hospital_id) & (daily['ward'] == selected_ward), 'bed_type'
].unique())
selected_bed_type = st.sidebar.selectbox("Bed type", bed_type_options)

st.sidebar.markdown("---")
st.sidebar.caption(f"Data current to {LATEST_DATE.date()}")
st.sidebar.caption("Model: LightGBM (selected in Notebook 05 based on test-set performance)")

# -----------------------------------------------------------------------------
# Current series and forecast
# -----------------------------------------------------------------------------
series = (daily[(daily['hospital_id'] == selected_hospital_id) &
                (daily['ward'] == selected_ward) &
                (daily['bed_type'] == selected_bed_type)]
          .sort_values('date').reset_index(drop=True))
latest_row = series[series['date'] == LATEST_DATE].iloc[0]

daily_forecast = predict(daily_model, series[series['date'] == LATEST_DATE], FEATURE_COLS)[0]
weekly_forecast = predict(weekly_model, series[series['date'] == LATEST_DATE], FEATURE_COLS)[0]
forecast_occ_rate = daily_forecast / latest_row['staffed_beds']

st.title(f"{selected_hospital_name}: {selected_ward} ({selected_bed_type})")
st.caption("AI-powered bed demand forecast, scenario simulation, and explainability")

# -----------------------------------------------------------------------------
# KPI row
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current occupied beds", f"{latest_row['occupied_beds']:.1f}",
            help=f"Staffed beds: {latest_row['staffed_beds']:.0f}")
col2.metric("Current occupancy", f"{latest_row['occupancy_rate']:.0%}")
col3.metric("Forecast tomorrow", f"{daily_forecast:.1f} beds",
            f"{daily_forecast - latest_row['occupied_beds']:+.1f}")
col4.metric("Forecast next week", f"{weekly_forecast:.1f} beds",
            f"{weekly_forecast - latest_row['occupied_beds']:+.1f}")

# -----------------------------------------------------------------------------
# Operational alert banner
# -----------------------------------------------------------------------------
if forecast_occ_rate >= CRITICAL_THRESHOLD:
    st.error(
        f"CRITICAL: tomorrow's forecast occupancy is {forecast_occ_rate:.0%}, at or above the "
        f"{CRITICAL_THRESHOLD:.0%} bottleneck threshold identified in Notebook 02. This ward is "
        f"forecast to be critically full."
    )
elif forecast_occ_rate >= WARNING_THRESHOLD:
    st.warning(
        f"WARNING: tomorrow's forecast occupancy is {forecast_occ_rate:.0%}, at or above the "
        f"{WARNING_THRESHOLD:.0%} watch threshold. Capacity pressure is building."
    )
else:
    st.success(f"Tomorrow's forecast occupancy ({forecast_occ_rate:.0%}) is within normal range.")

# -----------------------------------------------------------------------------
# Historical trend chart
# -----------------------------------------------------------------------------
st.subheader("Recent Occupancy Trend")
recent = series[series['date'] >= LATEST_DATE - pd.Timedelta(days=60)]
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(recent['date'], recent['occupied_beds'], color='#2c6e91', linewidth=1.5, label='Actual occupied beds')
ax.axhline(latest_row['staffed_beds'] * WARNING_THRESHOLD, color='#e59866', linestyle='--',
           linewidth=1, label=f'{WARNING_THRESHOLD:.0%} threshold')
ax.axhline(latest_row['staffed_beds'] * CRITICAL_THRESHOLD, color='#c0392b', linestyle='--',
           linewidth=1, label=f'{CRITICAL_THRESHOLD:.0%} threshold')
ax.scatter([LATEST_DATE + pd.Timedelta(days=1)], [daily_forecast], color='#c0392b', zorder=5,
           s=60, label='Tomorrow (forecast)')
ax.legend(loc='upper left', fontsize=8)
ax.set_ylabel('Occupied beds')
st.pyplot(fig)
plt.close(fig)

# -----------------------------------------------------------------------------
# Scenario simulation
# -----------------------------------------------------------------------------
st.subheader("Scenario Simulation")
st.caption(
    "Simulates the four operational scenarios named in the project brief, using the same "
    "evidence-grounded framework validated in Notebook 06. Intensity 1.0 reproduces the "
    "reference magnitude used in that notebook; values above or below scale it up or down."
)

scenario = st.selectbox(
    "Scenario",
    ["Flu outbreak (winter emergency surge)", "Delayed discharges",
     "Emergency admission spike", "Staffing shortage"],
)
intensity = st.slider("Scenario intensity", 0.0, 2.0, 1.0, 0.1)

baseline_row = series[series['date'] == LATEST_DATE]
scenario_row = apply_scenario(baseline_row.iloc[0], scenario, intensity)
scenario_df = pd.DataFrame([scenario_row])
scenario_forecast = predict(daily_model, scenario_df, FEATURE_COLS)[0]
delta = scenario_forecast - daily_forecast

scol1, scol2 = st.columns(2)
scol1.metric("Baseline forecast (tomorrow)", f"{daily_forecast:.1f} beds")
scol2.metric(f"{scenario} forecast", f"{scenario_forecast:.1f} beds", f"{delta:+.1f}")

if scenario == "Staffing shortage":
    st.info(
        "Note (from Notebook 06): this model does not treat staffing shortages as a strong "
        "driver of next-day occupancy, most likely because the true relationship runs the "
        "other way (high occupancy strains staffing, not the reverse). A small forecast "
        "change here should not be read as reassurance that staffing shortages are low-risk; "
        "it means this particular model is not the right tool for forecasting a staffing "
        "crisis's bed-capacity impact."
    )

# -----------------------------------------------------------------------------
# Explainability
# -----------------------------------------------------------------------------
st.subheader("Why This Forecast: Feature Contributions")
st.caption("SHAP values showing which factors pushed tomorrow's forecast up or down for this ward.")

X_explain = baseline_row[FEATURE_COLS + CAT_COLS].copy()
for c in CAT_COLS:
    X_explain[c] = X_explain[c].astype('category')
shap_values_row = explainer.shap_values(X_explain)[0]

shap_df = pd.DataFrame({
    'feature': FEATURE_COLS + CAT_COLS,
    'shap_value': shap_values_row,
}).sort_values('shap_value', key=abs, ascending=False).head(10)

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
colors = ['#c0392b' if v > 0 else '#2c6e91' for v in shap_df['shap_value'][::-1]]
ax2.barh(shap_df['feature'][::-1], shap_df['shap_value'][::-1], color=colors)
ax2.set_xlabel('SHAP value (impact on forecast, beds)')
ax2.axvline(0, color='black', linewidth=0.8)
st.pyplot(fig2)
plt.close(fig2)

# -----------------------------------------------------------------------------
# Footer: model info and known limitations
# -----------------------------------------------------------------------------
st.markdown("---")
with st.expander("Model information and known limitations"):
    st.markdown("""
    **Model:** LightGBM, selected in Notebook 05 based on held-out test-set performance
    (RMSE 1.41 beds daily, 2.68 beds weekly), trained on all 40 hospital-ward-bed_type series.

    **Known limitations, documented in Notebooks 05-06:**
    - The test split used for evaluation covers October-December 2025 only; a full
      multi-year seasonal backtest has not yet been performed, though the available
      December 2025 data showed no accuracy degradation in winter.
    - This model does not reliably forecast the bed-capacity impact of a staffing shortage
      in isolation (see the Staffing Shortage scenario note above).
    - Forecasts assume operational patterns similar to the 2024-2025 training period; a
      structural change (e.g. a new ward, major policy change) would need retraining.
    """)
