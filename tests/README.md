# Tests

Test suite memverifikasi data pipeline, feature engineering, model training, dan evaluation flow.

## Structure

- `unit/` — menguji satu modul atau satu contract secara terisolasi.
- `integration/` — menguji alur Spark/data pipeline dan training pipeline dengan dataset kecil.
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
