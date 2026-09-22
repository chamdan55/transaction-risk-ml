# Experiment Tracking

Folder ini berisi integration boundary untuk MLflow pada Sprint 3.

## Components

- `config.py` — typed loader untuk `configs/tracking.yaml`.
- `client.py` — wrapper untuk tracking URI, experiment creation/reuse, dan run lifecycle.
- `logging.py` — flattening dan logging training parameters ke active MLflow run.
- `metadata.py` — dataset summary, config hash, git commit, feature-contract version, dan training
  timestamp.
- `lineage.py` — stable dataset/schema/content/config/code fingerprints, split row counts, dan time
  ranges.
- `registry.py` — logging, registration reference, dan loading model artifact.

## Contract

- Experiment name: `transaction-risk-classification`.
- Registered model name: `transaction-risk-model`.
- Tracking URI SQLite dan artifact location berasal dari `configs/tracking.yaml`.
- Tracking storage lokal tidak di-commit ke Git.

Model artifact disimpan sebagai satu bundle yang mencakup preprocessing dan estimator. Model registry metadata tetap berasal dari MLflow run.

## Workflow

Jalankan training dan tracking bersama-sama:

```pwsh
python -m pipelines.train_models --config configs/model.yaml --tracking-config configs/tracking.yaml
```

Perintah tersebut membuat parent run, nested validation run untuk setiap model, lalu final
candidate run yang menyimpan evaluation report, calibration data, dataset manifest, model
signature/input example, dan mendaftarkan model pemenang. Alias
`candidate` hanya bergerak setelah signature dan feature-only input example tervalidasi terhadap
model yang dimuat kembali. Artifact cloudpickle hanya boleh dimuat dari registry yang terpercaya dan
access-controlled; jangan memuat artifact model dari sumber eksternal. Alias `staging` dan
`production` selalu membutuhkan persetujuan eksplisit:

```pwsh
python -m pipelines.promote_model --version 3 --stage staging --approved-by "reviewer" --reason "PR-AUC dan recall memenuhi quality gate."
```

Model version yang dibuat sebelum signature/input validation berhasil harus diperlakukan sebagai
non-promotable audit history. Version tersebut tidak dihapus otomatis; hanya candidate baru yang
lolos semua validation gate yang boleh menerima alias `candidate`.
