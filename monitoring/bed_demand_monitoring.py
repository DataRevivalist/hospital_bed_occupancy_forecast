"""
Continuous learning / monitoring pipeline for the Albion Care Network bed demand
forecasting model.

This script implements the brief's "continuous learning pipeline that improves forecasting
performance over time" objective as three composable steps:

1. monitor_performance(): compute rolling forecast error as new ground truth arrives.
2. check_for_drift(): compare rolling error against the Notebook 05 test-set baseline and
   flag when it degrades beyond a defined tolerance.
3. retrain_daily_model(): rerun the exact training procedure from Notebook 04 on an updated
   feature panel, producing a refreshed model artefact.

In production this would be scheduled (e.g. a nightly job); here it is provided as a set of
functions that Notebook 07 calls directly and tests against real historical data.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb

# Baseline established in Notebook 05's final test-set evaluation. A future run's rolling
# error is compared against this fixed reference, not a moving target, so genuine
# degradation over time is detectable.
BASELINE_TEST_MAE = 1.06
DEFAULT_DRIFT_TOLERANCE = 1.25  # flag if rolling MAE exceeds 25% above the baseline
DEFAULT_ROLLING_WINDOW = 14

EXCLUDE_COLS = ['hospital_id', 'ward', 'bed_type', 'date', 'target_next_day_occupied',
                'target_next_week_occupied', 'split', 'median_los_hours']
CAT_COLS = ['hospital_id', 'ward', 'bed_type']


def monitor_performance(predictions_df, date_col='date', error_col='abs_error',
                         window=DEFAULT_ROLLING_WINDOW):
    """Compute a rolling mean absolute error over calendar time from a dataframe of daily
    predictions with a known ground-truth error already attached. Returns a date-indexed
    Series of rolling MAE."""
    daily_mae = predictions_df.groupby(date_col)[error_col].mean().sort_index()
    return daily_mae.rolling(window).mean()


def check_for_drift(rolling_mae, baseline_mae=BASELINE_TEST_MAE,
                     tolerance=DEFAULT_DRIFT_TOLERANCE):
    """Flag dates where rolling MAE exceeds the allowed multiple of the established
    baseline. Returns the flagged dates and a single boolean for whether retraining should
    be triggered (any breach in the most recent window)."""
    threshold = baseline_mae * tolerance
    breaches = rolling_mae[rolling_mae > threshold]
    retrain_recommended = bool(len(rolling_mae) and rolling_mae.iloc[-1] > threshold)
    return {
        'threshold': threshold,
        'breach_dates': breaches,
        'latest_rolling_mae': rolling_mae.iloc[-1] if len(rolling_mae) else None,
        'retrain_recommended': retrain_recommended,
    }


def retrain_daily_model(feature_panel_path, output_model_path, up_to_date=None,
                         lgb_params=None):
    """Retrain the daily LightGBM model on all data up to (and including) `up_to_date`
    (defaults to the latest date in the panel), using the same feature set and parameters
    established in Notebooks 03-04. This is the function a scheduled job would call once
    `check_for_drift` recommends retraining, or on a fixed cadence (e.g. monthly)
    regardless of drift, to keep the model current as new seasons of data accumulate."""
    daily = pd.read_parquet(feature_panel_path)
    feature_cols = [c for c in daily.columns if c not in EXCLUDE_COLS]
    for c in feature_cols:
        if 'pyarrow' in str(daily[c].dtype):
            daily[c] = daily[c].astype('float64')

    if up_to_date is not None:
        daily = daily[daily['date'] <= pd.Timestamp(up_to_date)]

    train_data = daily.dropna(subset=['target_next_day_occupied']).copy()
    X = train_data[feature_cols + CAT_COLS].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype('category')
    y = train_data['target_next_day_occupied']

    params = lgb_params or {
        'num_leaves': 15, 'max_depth': 8, 'learning_rate': 0.05,
        'n_estimators': 200, 'min_child_samples': 40, 'random_state': 42, 'verbosity': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X, y, categorical_feature=CAT_COLS)
    model.booster_.save_model(str(output_model_path))

    return {
        'output_model_path': str(output_model_path),
        'training_rows': len(train_data),
        'training_end_date': str(train_data['date'].max().date()),
    }


if __name__ == '__main__':
    # Example manual invocation: monitor the existing test split, check for drift, and
    # retrain only if recommended. In production this would be triggered by a scheduler.
    import sys
    panel_path = Path(__file__).parent.parent / 'data' / 'processed' / 'daily_feature_panel.parquet'
    model_path = Path(__file__).parent.parent / 'models' / 'lightgbm_daily_tuned.txt'

    daily = pd.read_parquet(panel_path)
    print(f'Loaded panel: {len(daily):,} rows, latest date {daily["date"].max().date()}')
    print('Run this module\'s functions from Notebook 07 or a scheduler; see '
          'bed_demand_monitoring.py docstrings for monitor_performance, check_for_drift, and '
          'retrain_daily_model.')
