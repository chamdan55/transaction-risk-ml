# Machine Learning Modules

Folder `ml/` berisi reusable logic untuk data engineering, feature engineering, model training, dan evaluation. Modul di sini tidak seharusnya menjalankan pipeline secara langsung; entry point executable berada di `pipelines/`.

## Subpackages

- `data/` — ingestion, schema, validation, transformation, profiling, dan chronological split.
- `features/` — transaction-level dan behavioral feature engineering.
- `training/` — training dataset contract, preprocessing, imbalance handling, dan model training.
- `evaluation/` — metrics, threshold analysis, dan production-candidate selection.
- `monitoring/` — reserved untuk model/system monitoring pada sprint lanjutan.

## Design Rules

- Logic harus dapat diuji tanpa bergantung pada entry point pipeline.
- Kontrak schema dan target harus eksplisit.
- Identifier, target, dan proxy label tidak boleh masuk ke model features.
- Preprocessing harus di-fit pada training data saja.
