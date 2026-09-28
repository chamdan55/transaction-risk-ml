# Tests

Test suite memverifikasi data pipeline, feature engineering, model training, dan evaluation flow.

## Structure

- `unit/` — menguji satu modul atau satu contract secara terisolasi.
- `contract/` — menguji schema/version contract publik dan internal.
- `integration/` — menguji alur Spark/data pipeline dan training pipeline dengan dataset kecil.
- `load/` — skenario baseline beban API menggunakan k6.
- `conftest.py` — shared Spark fixture dan test configuration.

## Running Tests

```powershell
pytest
pytest tests/unit
pytest tests/integration
```

## Sprint 2 Coverage

Sprint 2 tests mencakup:

- Training dataset contract dan split isolation.
- Preprocessing dan unseen categorical values.
- Class imbalance handling.
- Logistic Regression, Random Forest, dan XGBoost.
- Classification metrics.
- Threshold/business-cost analysis.
- Production candidate selection.
- Training artifact dan reproducibility flow.

Sprint 6 adds contract and integration coverage for privacy-safe prediction events, delayed-label
idempotency, inference failure isolation, and scheduled drift/performance report generation. Run
`pytest tests/contract/test_prediction_events.py` and
`pytest tests/integration/test_feedback_pipeline.py tests/integration/test_drift_job.py` for that
coverage; report-job runtime validation additionally requires the optional `monitoring` dependencies
and local model/reference artifacts.
