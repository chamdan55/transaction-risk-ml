# Model Training

Folder ini berisi kontrak dataset dan komponen training Sprint 2.

## Components

- `dataset.py` — membaca dan memvalidasi train/validation/test Parquet.
- `config.py` — typed loader dan validator `configs/model.yaml`.
- `preprocessing.py` — numeric imputation/scaling dan categorical encoding.
- `imbalance.py` — target distribution, balanced class weight, dan XGBoost `scale_pos_weight`.
- `baseline.py` — Logistic Regression baseline serta artifact loader/saver.
- `models.py` — Random Forest dan XGBoost comparison models.

## Training Rules

- Fit preprocessing hanya pada training split.
- Validation digunakan untuk perbandingan model dan threshold selection.
- Test hanya digunakan untuk final evaluation.
- Seed dan model parameters berasal dari `configs/model.yaml`.
- Model artifacts disimpan di `artifacts/models/` dan tidak di-commit.
