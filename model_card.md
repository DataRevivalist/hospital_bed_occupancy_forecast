# Model Card: Albion Care Network Bed Demand Forecasting

## Model Details
- **Type:** Gradient boosted trees (LightGBM)
- **Horizons:** Daily (t+1), weekly (t+7), and hourly (ED arrivals, t+1h)
- **Selected in:** Notebook 05, based on held-out test-set performance
- **Training data:** 2024-01-11 to 2025-12-30 (after excluding an 10-day simulation
  burn-in artefact identified in Notebook 02), across 40 hospital-ward-bed_type series

## Intended Use
Forecasting next-day and next-week inpatient bed occupancy, and next-hour ED arrivals, to
support proactive capacity planning at Albion Care Network. Includes scenario simulation for
flu outbreaks, delayed discharges, emergency admission spikes, and staffing shortages.

## Performance (Held-Out Test Set, Notebook 05)
| Horizon | Model | RMSE | MAE | R2 |
|---|---|---|---|---|
| Daily (t+1) | LightGBM (tuned) | 1.41 beds | 0.93 beds | 0.975 |
| Weekly (t+7) | LightGBM | 2.68 beds | 1.79 beds | 0.910 |
| Hourly (ED arrivals) | LightGBM | 2.31 arrivals | 1.73 arrivals | 0.604 |

XGBoost had a marginally lower daily-horizon test RMSE (1.396 vs 1.407); LightGBM was selected
for practical deployment reasons (training speed, native categorical handling, smaller model
file), a documented trade-off rather than the single most accurate option available.

## Known Limitations
1. **Staffing shortage scenario:** this model does not treat staffing shortfalls as a
   meaningful driver of next-day occupancy (Notebook 06 Section 8). It should not be used
   alone to forecast the bed-capacity impact of a staffing crisis.
2. **Seasonal test coverage:** the test split covers Q4 2025 only. December 2025 (a genuine
   winter month) showed the lowest error of the three test months, which is reassuring, but a
   full multi-year seasonal backtest has not been performed.
3. **Day Case Unit:** occupancy rate is a misleading capacity metric for this ward (Notebook
   02); throughput would be a better metric for it in future iterations.
4. **Deep learning:** the LSTM tested in Notebook 04 did not outperform gradient boosting or
   naive persistence with the data volume available (~20,000 rows across 40 series); this may
   change with substantially more historical data.
5. **Elective surgery mapping:** surgery-derived features use a specialty-to-ward mapping
   documented in Notebook 03 that is a simplification for Orthopaedics (split evenly across
   Ward A/B).

## Deployment Notes
The Streamlit app (`app/app.py`) reads data and model files directly from the repository's
`data/processed/` and `models/` folders; it does not keep a separate copy. Keeping the app
folder free of duplicated data avoids the two copies silently going out of sync.

## Retraining
See `monitoring/bed_demand_monitoring.py`. Retraining uses the identical feature engineering
and model configuration validated in Notebooks 03-05, applied to all data up to the
retraining date.
