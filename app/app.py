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
import matplotlib.dates as mdates
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

SCENARIO_OPTIONS = [
    "No scenario (baseline)",
    "Flu outbreak (winter emergency surge)",
    "Delayed discharges",
    "Emergency admission spike",
    "Staffing shortage",
]
# Scenarios that plausibly change ED arrival volume (used by the hourly arrivals filter
# below). Delayed discharges and staffing shortage act on occupancy/staffing directly,
# not on how many patients arrive at the door, per the Notebook 06 scenario design, so
# they deliberately leave the arrivals chart unchanged.
ARRIVAL_AFFECTING_SCENARIOS = {
    "Flu outbreak (winter emergency surge)": FLU_OUTBREAK_MULTIPLIER,
    "Emergency admission spike": EMERGENCY_SPIKE_MULTIPLIER,
}

# Time-of-day buckets used by Filters 2 and 4. Three categories were requested (not four),
# so "Evening" is defined to also cover overnight hours (18:00-05:59), keeping every hour
# of the day assigned to exactly one bucket rather than silently dropping the night hours.
TIME_OF_DAY_OPTIONS = ["Morning", "Afternoon", "Evening"]


def time_of_day_bucket(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    else:
        return "Evening"


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


@st.cache_data
def load_hourly_panel():
    df = pd.read_parquet(DATA_DIR / "hourly_ed_arrivals_panel.parquet")
    df['time_of_day'] = df['hour'].apply(time_of_day_bucket)
    return df


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
    a scenario up or down from the reference magnitude (intensity = 1.0).
    Returns the baseline row unchanged if no scenario is selected."""
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
hourly = load_hourly_panel()
daily_model = load_daily_model()
weekly_model = load_weekly_model()
explainer = load_shap_explainer(daily_model)

LATEST_DATE = daily['date'].max()
LATEST_HOUR = hourly['datetime'].max()

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
st.sidebar.subheader("Time-Based Filters")

hourly_min_date = hourly['datetime'].min().normalize().date()
hourly_max_start_date = (LATEST_HOUR - pd.Timedelta(hours=71)).normalize().date()
default_start_date = (LATEST_HOUR - pd.Timedelta(hours=71)).normalize().date()

window_start_date = st.sidebar.date_input(
    "72-hour window start date",
    value=default_start_date,
    min_value=hourly_min_date,
    max_value=hourly_max_start_date,
    help="Picks a 3-day (72-hour) window for the hourly arrivals views below. "
         "The window always runs from 00:00 on this date for exactly 72 hours.",
)
window_start = pd.Timestamp(window_start_date)
window_end = window_start + pd.Timedelta(hours=72)

selected_times_of_day = st.sidebar.multiselect(
    "Time of day",
    TIME_OF_DAY_OPTIONS,
    default=TIME_OF_DAY_OPTIONS,
    help="Morning = 06:00-11:59, Afternoon = 12:00-17:59, Evening = 18:00-05:59 "
         "(evening bucket also covers overnight hours).",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Scenario Filter")
st.sidebar.caption(
    "Applies to every forecast, alert, chart, and explanation on this page, using the "
    "same evidence-grounded framework validated in Notebook 06."
)
selected_scenario = st.sidebar.selectbox("Scenario", SCENARIO_OPTIONS)
if selected_scenario != "No scenario (baseline)":
    scenario_intensity = st.sidebar.slider(
        "Scenario intensity", 0.0, 2.0, 1.0, 0.1,
        help="1.0 reproduces the Notebook 06 reference magnitude; higher or lower scales it.",
    )
else:
    scenario_intensity = 0.0

st.sidebar.markdown("---")
st.sidebar.caption(f"Data current to {LATEST_DATE.date()}")
st.sidebar.caption("Model: LightGBM (selected in Notebook 05 based on test-set performance)")

# -----------------------------------------------------------------------------
# Current series, baseline row, and scenario-adjusted row
# -----------------------------------------------------------------------------
series = (daily[(daily['hospital_id'] == selected_hospital_id) &
                (daily['ward'] == selected_ward) &
                (daily['bed_type'] == selected_bed_type)]
          .sort_values('date').reset_index(drop=True))
baseline_row = series[series['date'] == LATEST_DATE].iloc[0]
active_row = apply_scenario(baseline_row, selected_scenario, scenario_intensity)
active_df = pd.DataFrame([active_row])

baseline_forecast = predict(daily_model, series[series['date'] == LATEST_DATE], FEATURE_COLS)[0]
active_daily_forecast = predict(daily_model, active_df, FEATURE_COLS)[0]
active_weekly_forecast = predict(weekly_model, active_df, FEATURE_COLS)[0]
forecast_occ_rate = active_daily_forecast / baseline_row['staffed_beds']

scenario_active = selected_scenario != "No scenario (baseline)"

# -----------------------------------------------------------------------------
# Hourly arrivals for the selected 72-hour window, time-of-day filter, and
# scenario -- computed here (rather than lower down) specifically so the KPI
# row below can be genuinely wired to these two filters, not just the charts
# further down the page.
# -----------------------------------------------------------------------------
hospital_hourly = hourly[
    (hourly['hospital_id'] == selected_hospital_id) &
    (hourly['datetime'] >= window_start) &
    (hourly['datetime'] < window_end)
].sort_values('datetime').copy()

# Apply the same scenario multiplier used for the daily model, where the scenario
# plausibly affects arrival volume (see ARRIVAL_AFFECTING_SCENARIOS above).
if selected_scenario in ARRIVAL_AFFECTING_SCENARIOS:
    base_mult = ARRIVAL_AFFECTING_SCENARIOS[selected_scenario]
    mult = 1 + (base_mult - 1) * scenario_intensity
    hospital_hourly['ed_arrivals_display'] = hospital_hourly['ed_arrivals'] * mult
else:
    hospital_hourly['ed_arrivals_display'] = hospital_hourly['ed_arrivals']

hospital_hourly_filtered = hospital_hourly[hospital_hourly['time_of_day'].isin(selected_times_of_day)]

st.title(f"{selected_hospital_name}: {selected_ward} ({selected_bed_type})")
if scenario_active:
    st.caption(
        f"AI-powered bed demand forecast, scenario simulation, and explainability -- "
        f"showing **{selected_scenario}** at intensity {scenario_intensity:.1f}x"
    )
else:
    st.caption("AI-powered bed demand forecast, scenario simulation, and explainability")

# -----------------------------------------------------------------------------
# KPI row 1: daily bed forecast (reflects Hospital/Ward/Bed type and Scenario;
# NOT affected by the 72-hour window or Time of Day filters, since the daily
# bed-occupancy data this is built from has no hourly resolution at all -- that
# is a genuine limitation of the source data, not something these filters can
# meaningfully change, so it deliberately isn't faked here).
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current occupied beds", f"{baseline_row['occupied_beds']:.1f}",
            help=f"Staffed beds: {baseline_row['staffed_beds']:.0f}. Reflects today's actual "
                 f"occupancy, so it is not affected by the Scenario, 72-hour window, or Time "
                 f"of Day filters.")
col2.metric("Current occupancy", f"{baseline_row['occupancy_rate']:.0%}")
col3.metric("Forecast tomorrow", f"{active_daily_forecast:.1f} beds",
            f"{active_daily_forecast - baseline_row['occupied_beds']:+.1f}",
            help="Under the selected scenario filter" if scenario_active else None)
col4.metric("Forecast next week", f"{active_weekly_forecast:.1f} beds",
            f"{active_weekly_forecast - baseline_row['occupied_beds']:+.1f}",
            help="Under the selected scenario filter" if scenario_active else None)

# -----------------------------------------------------------------------------
# KPI row 2: hourly ED arrivals snapshot -- this is the one genuinely wired to
# the 72-hour window, Time of Day, and Scenario filters, since ED arrivals are
# the one dataset in this app that actually has hourly resolution.
# -----------------------------------------------------------------------------
st.markdown("###### Hourly Arrivals Snapshot")
st.caption(
    f"Reflects the 72-hour window ({window_start.strftime('%d %b')}-"
    f"{(window_end - pd.Timedelta(hours=1)).strftime('%d %b %Y')}), the Time of Day filter, "
    f"and the Scenario filter, all selected in the sidebar. Hospital-wide, not ward-specific "
    f"(ED arrivals are not recorded per ward in the source data)."
)

if hospital_hourly_filtered.empty:
    st.info(
        "No hours match the current Time of Day filter for this 72-hour window. "
        "Adjust the filters in the sidebar to see this snapshot."
    )
else:
    hcol1, hcol2, hcol3 = st.columns(3)

    avg_arrivals = hospital_hourly_filtered['ed_arrivals_display'].mean()
    baseline_avg_arrivals = hospital_hourly_filtered['ed_arrivals'].mean()
    arrivals_delta = avg_arrivals - baseline_avg_arrivals
    hcol1.metric(
        "Avg arrivals/hour (filtered)", f"{avg_arrivals:.1f}",
        f"{arrivals_delta:+.1f}" if scenario_active and selected_scenario in ARRIVAL_AFFECTING_SCENARIOS else None,
        help="Average hourly ED arrivals across the selected 72-hour window and Time of Day filter.",
    )

    total_arrivals = hospital_hourly_filtered['ed_arrivals_display'].sum()
    hcol2.metric(
        "Total arrivals (filtered)", f"{total_arrivals:.0f}",
        help="Summed across every hour matching the current window and Time of Day filter.",
    )

    tod_avg_now = hospital_hourly_filtered.groupby('time_of_day')['ed_arrivals_display'].mean()
    busiest_period = tod_avg_now.idxmax()
    hcol3.metric(
        "Busiest period (filtered)", busiest_period,
        f"{tod_avg_now.max():.1f}/hr",
        help="Whichever Time of Day period selected in the sidebar has the highest average "
             "arrivals within this window.",
    )

    if scenario_active and selected_scenario not in ARRIVAL_AFFECTING_SCENARIOS:
        st.caption(
            f"Note: \u201c{selected_scenario}\u201d does not change these arrival figures -- "
            "in the Notebook 06 scenario design, it acts on occupancy or staffing directly, "
            "not on how many patients arrive at the door."
        )

# -----------------------------------------------------------------------------
# Operational alert banner (reflects the active scenario)
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

if scenario_active:
    st.caption(
        f"Alert reflects the **{selected_scenario}** scenario at intensity "
        f"{scenario_intensity:.1f}x. Switch back to \u201cNo scenario (baseline)\u201d in the "
        f"sidebar to see the unadjusted alert."
    )

# -----------------------------------------------------------------------------
# Historical trend chart
# -----------------------------------------------------------------------------
st.subheader("Recent Occupancy Trend")
recent = series[series['date'] >= LATEST_DATE - pd.Timedelta(days=60)]
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(recent['date'], recent['occupied_beds'], color='#2c6e91', linewidth=1.5, label='Actual occupied beds')
ax.axhline(baseline_row['staffed_beds'] * WARNING_THRESHOLD, color='#e59866', linestyle='--',
           linewidth=1, label=f'{WARNING_THRESHOLD:.0%} threshold')
ax.axhline(baseline_row['staffed_beds'] * CRITICAL_THRESHOLD, color='#c0392b', linestyle='--',
           linewidth=1, label=f'{CRITICAL_THRESHOLD:.0%} threshold')
ax.scatter([LATEST_DATE + pd.Timedelta(days=1)], [baseline_forecast], color='#999999', zorder=4,
           s=50, label='Tomorrow (baseline forecast)')
if scenario_active:
    ax.scatter([LATEST_DATE + pd.Timedelta(days=1)], [active_daily_forecast], color='#c0392b', zorder=5,
               s=60, marker='D', label=f'Tomorrow (under {selected_scenario})')
ax.legend(loc='upper left', fontsize=8)
ax.set_ylabel('Occupied beds')
st.pyplot(fig)
plt.close(fig)

# -----------------------------------------------------------------------------
# Hourly Arrivals - 72-Hour Window (Filters 1 and 2: hour range + time of day)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Hourly Arrivals: 72-Hour Window")
st.caption(
    f"{window_start.strftime('%d %b %Y')} 00:00 to {(window_end - pd.Timedelta(hours=1)).strftime('%d %b %Y')} "
    f"23:00, for {selected_hospital_name}. ED arrivals are recorded at the hospital level in the "
    f"source data, not per ward, so this view is hospital-wide rather than ward-specific."
)

# hospital_hourly and hospital_hourly_filtered were already computed earlier, right before
# the KPI row, so the two are guaranteed to stay in sync rather than being calculated twice.

if hospital_hourly_filtered.empty:
    st.info("No hours match the current Time of Day filter for this window. Adjust the filter in the sidebar.")
else:
    tod_colors = {'Morning': '#F2994A', 'Afternoon': '#2C6E91', 'Evening': '#5C6B73'}
    fig3, ax3 = plt.subplots(figsize=(11, 3.8))
    ax3.plot(hospital_hourly['datetime'], hospital_hourly['ed_arrivals_display'],
             color='#CBD5D9', linewidth=1, zorder=1, label='All hours in window')
    for tod in TIME_OF_DAY_OPTIONS:
        if tod not in selected_times_of_day:
            continue
        subset = hospital_hourly_filtered[hospital_hourly_filtered['time_of_day'] == tod]
        ax3.scatter(subset['datetime'], subset['ed_arrivals_display'], color=tod_colors[tod],
                    s=18, zorder=3, label=tod)
    ax3.set_ylabel('ED arrivals per hour')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%d %b\n%H:%M'))
    ax3.legend(loc='upper left', fontsize=8, ncol=4)
    if selected_scenario in ARRIVAL_AFFECTING_SCENARIOS:
        ax3.set_title(f"Shown under {selected_scenario} (intensity {scenario_intensity:.1f}x)", fontsize=10)
    st.pyplot(fig3)
    plt.close(fig3)

# -----------------------------------------------------------------------------
# Arrivals by Time of Day (Filter 4)
# -----------------------------------------------------------------------------
st.subheader("Arrivals by Time of Day")
st.caption(
    "Average hourly arrivals within each time-of-day period, for the 72-hour window and "
    "scenario filters selected above. Updates automatically as those filters change."
)

if hospital_hourly_filtered.empty:
    st.info("No data to summarise for the current Time of Day filter.")
else:
    tod_summary = (hospital_hourly_filtered.groupby('time_of_day')['ed_arrivals_display']
                   .mean().reindex([t for t in TIME_OF_DAY_OPTIONS if t in selected_times_of_day]))
    fig4, ax4 = plt.subplots(figsize=(7, 3.8))
    bar_colors = [tod_colors[t] for t in tod_summary.index]
    ax4.bar(tod_summary.index, tod_summary.values, color=bar_colors)
    ax4.set_ylabel('Average arrivals per hour')
    for i, v in enumerate(tod_summary.values):
        ax4.text(i, v + max(tod_summary.values) * 0.02, f'{v:.1f}', ha='center', fontsize=10)
    st.pyplot(fig4)
    plt.close(fig4)

    fastest = tod_summary.idxmax()
    st.caption(
        f"Busiest period in this window: **{fastest}** "
        f"({tod_summary.max():.1f} arrivals/hour on average)."
    )

# -----------------------------------------------------------------------------
# Scenario impact summary (Filter 3: global scenario, all analyses)
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Scenario Impact Summary")

if not scenario_active:
    st.info(
        "No scenario is currently selected. Choose one from the **Scenario Filter** in the "
        "sidebar to see its effect on tomorrow's forecast, the alert banner, the trend chart, "
        "and the arrivals views above."
    )
else:
    delta = active_daily_forecast - baseline_forecast
    scol1, scol2 = st.columns(2)
    scol1.metric("Baseline forecast (tomorrow)", f"{baseline_forecast:.1f} beds")
    scol2.metric(f"Under {selected_scenario}", f"{active_daily_forecast:.1f} beds", f"{delta:+.1f}")

    if selected_scenario == "Staffing shortage":
        st.info(
            "Note (from Notebook 06): this model does not treat staffing shortages as a strong "
            "driver of next-day occupancy, most likely because the true relationship runs the "
            "other way (high occupancy strains staffing, not the reverse). A small forecast "
            "change here should not be read as reassurance that staffing shortages are low-risk; "
            "it means this particular model is not the right tool for forecasting a staffing "
            "crisis's bed-capacity impact."
        )

# -----------------------------------------------------------------------------
# Explainability (reflects the active scenario)
# -----------------------------------------------------------------------------
st.subheader("Why This Forecast: Feature Contributions")
st.caption(
    "SHAP values showing which factors pushed tomorrow's forecast up or down for this ward"
    + (f", under the **{selected_scenario}** scenario." if scenario_active else ".")
)

X_explain = active_df[FEATURE_COLS + CAT_COLS].copy()
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
    - ED arrivals are recorded at the hospital level, not per ward, so the hourly arrivals
      views above are hospital-wide even when a specific ward is selected in the sidebar.
    """)
