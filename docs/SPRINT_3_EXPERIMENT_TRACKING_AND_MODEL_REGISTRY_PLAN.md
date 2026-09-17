# Sprint 3 — Experiment Tracking & Model Registry

## 1. Sprint Goal

Mengintegrasikan MLflow untuk mencatat experiment training, menyimpan parameter dan metrics, mengelola model artifact, serta mendaftarkan production candidate ke model registry secara reproducible.

Sprint 3 menggunakan output Sprint 2:

```text
data/processed/features/train/
data/processed/features/validation/
data/processed/features/test/
artifacts/models/
artifacts/evaluation_report.json
configs/model.yaml
```

## 2. Scope

### In scope

- MLflow experiment tracking.
- Tracking URI yang dapat dijalankan secara lokal.
- Logging parameter training dan preprocessing.
- Logging validation dan final test metrics.
- Logging threshold analysis dan selected threshold.
- Logging model artifacts.
- Registering production candidate ke MLflow Model Registry.
- Model version metadata dan dataset metadata.
- Candidate/staging/production promotion workflow.
- Reproducibility test untuk MLflow run.

### Out of scope

- FastAPI inference endpoint.
- Docker image dan Kubernetes deployment.
- Prometheus/Grafana monitoring.
- Evidently drift monitoring.
- Automated retraining.
- Cloud-hosted MLflow server.
- Automatic model promotion tanpa quality gate.

## 3. MLflow Tracking Strategy

Experiment utama:

```text
transaction-risk-classification
```

Satu training pipeline menghasilkan satu parent run atau satu run per model, dengan struktur yang konsisten untuk:

```text
Training Run
├── Parameters
├── Metrics
├── Tags
├── Dataset metadata
├── Evaluation report
└── Model artifact
```

Local development menggunakan MLflow tracking URI berbasis local file atau SQLite. Tracking storage dan model artifacts tidak boleh masuk Git.

## 4. Tracking Contract

### Required parameters

- `model_name`.
- `random_seed`.
- `target_column`.
- `feature_count`.
- `train_row_count`.
- `validation_row_count`.
- `test_row_count`.
- Model hyperparameters.
- Class imbalance strategy.
- Preprocessing strategy.

### Required metrics

- Validation precision.
- Validation recall.
- Validation F1.
- Validation ROC-AUC.
- Validation PR-AUC.
- Selected threshold.
- Final test precision.
- Final test recall.
- Final test F1.
- Final test ROC-AUC.
- Final test PR-AUC.

### Required tags

- `project`: `transaction-risk-ml`.
- `stage`: `validation` atau `final`.
- `dataset_name`: `paysim`.
- `candidate_status`: `candidate`, `staging`, atau `production`.
- `git_commit` jika tersedia.

## 5. Target Repository Artifacts

Struktur implementasi yang direncanakan:

```text
configs/
└── tracking.yaml

ml/
└── tracking/
    ├── __init__.py
    ├── client.py
    ├── logging.py
    ├── registry.py
    └── metadata.py

pipelines/
└── track_training_run.py

tests/
├── unit/
│   ├── test_tracking_metadata.py
│   ├── test_tracking_logging.py
│   └── test_model_registry.py
└── integration/
    └── test_mlflow_training_run.py
```

Nama modul dapat disesuaikan saat implementasi selama tracking dan registry responsibilities tetap terpisah.

## 6. Step-by-Step Implementation Plan

### Step 1 — Define Tracking Configuration

Tujuan: membuat MLflow tracking dapat dikonfigurasi tanpa hard-coded URI.

Implementation:

- Menambahkan `configs/tracking.yaml`.
- Mendefinisikan experiment name.
- Mendefinisikan tracking URI.
- Mendefinisikan artifact location.
- Mendefinisikan registered model name.
- Mendefinisikan apakah local run boleh membuat experiment otomatis.

Contoh konfigurasi:

```yaml
tracking:
  uri: mlruns
  experiment_name: transaction-risk-classification
  registered_model_name: transaction-risk-model
  artifact_location: mlartifacts
```

### Step 2 — Add MLflow Dependency and Client

Tujuan: menyediakan client MLflow yang konsisten untuk semua pipeline.

Implementation:

- Menambahkan dependency MLflow.
- Membuat tracking client wrapper.
- Menetapkan tracking URI.
- Mendapatkan atau membuat experiment.
- Menangani missing experiment secara deterministik.
- Memastikan client dapat digunakan pada local filesystem.

### Step 3 — Define Dataset and Run Metadata

Tujuan: memastikan setiap run dapat ditelusuri kembali ke input dan konfigurasi.

Implementation:

- Membuat dataset metadata dari training dataset contract.
- Menyimpan split row counts.
- Menyimpan feature count dan target column.
- Menyimpan config path dan config hash.
- Menyimpan git commit jika tersedia.
- Menyimpan timestamp training dalam run metadata.

### Step 4 — Log Training Parameters

Tujuan: membuat setiap experiment dapat dibandingkan.

Implementation:

- Log model type.
- Log random seed.
- Log preprocessing parameters.
- Log imbalance strategy.
- Log model hyperparameters.
- Log threshold candidates.
- Menjaga parameter names flat dan MLflow-compatible.

### Step 5 — Log Validation Metrics and Threshold Analysis

Tujuan: menyimpan seluruh informasi yang digunakan untuk model selection.

Implementation:

- Log precision, recall, F1, ROC-AUC, dan PR-AUC validation.
- Log confusion matrix sebagai artifact atau JSON.
- Log threshold analysis sebagai JSON/CSV artifact.
- Log selected threshold.
- Log expected business cost jika tersedia.
- Menandai validation run sebagai `candidate`.

### Step 6 — Log and Register Model Artifacts

Tujuan: membuat model terpilih dapat dimuat kembali melalui registry.

Implementation:

- Log preprocessing dan estimator sebagai satu model artifact.
- Gunakan flavor yang sesuai untuk Logistic Regression dan Random Forest.
- Gunakan flavor yang sesuai untuk XGBoost.
- Register hanya production candidate, bukan semua model secara otomatis.
- Simpan model signature dan input example jika memungkinkan.
- Verifikasi model yang terdaftar dapat dimuat kembali.

### Step 7 — Implement Model Promotion Workflow

Tujuan: memisahkan candidate selection dari promotion.

Promotion flow:

```text
Validation Candidate
        ↓
Registered Model Version
        ↓
Staging
        ↓
Quality Gate
        ↓
Production
```

Implementation:

- Membuat registered model dengan nama stabil.
- Menambahkan version metadata.
- Menambahkan aliases atau stage sesuai kemampuan MLflow version yang digunakan.
- Menyimpan validation dan final test metrics pada version metadata.
- Tidak melakukan automatic production promotion tanpa quality gate.

### Step 8 — Integrate with Training Pipeline

Tujuan: menjalankan training dan tracking dari satu command.

Implementation:

- Mengintegrasikan MLflow ke `pipelines/train_models.py` atau membuat wrapper pipeline.
- Memulai run untuk setiap model atau satu parent run dengan nested model runs.
- Menjalankan existing candidate selection dari Sprint 2.
- Menjalankan final test evaluation setelah candidate dipilih.
- Menutup run secara aman ketika terjadi exception.
- Menyimpan run ID dan registered model version pada summary.

### Step 9 — Add Tests and Quality Gates

Minimum test coverage:

- Tracking configuration dapat dibaca.
- Metadata memiliki field wajib.
- Parameters dapat dilog tanpa type error.
- Metrics dapat dilog.
- Evaluation report tersimpan sebagai artifact.
- Model artifact dapat diload kembali.
- Registered model version memiliki metadata yang benar.
- Training run reproducible dengan konfigurasi yang sama.
- Missing MLflow tracking directory ditangani dengan benar.

Quality checks:

- Pytest passed.
- Ruff passed.
- Format check passed.
- Pre-commit passed.
- GitHub Actions tetap green.

## 7. Expected Outputs

Setelah Sprint 3 selesai, repository diharapkan memiliki:

- MLflow experiment `transaction-risk-classification`.
- Training run dengan parameter dan metrics lengkap.
- Evaluation report sebagai MLflow artifact.
- Registered model `transaction-risk-model`.
- Production candidate dengan version metadata.
- Model yang dapat dimuat kembali melalui MLflow.
- Command training yang menghasilkan tracking run reproducible.
- Test suite untuk tracking dan registry flow.

## 8. Definition of Done

- [ ] MLflow dependency ditambahkan dan terdokumentasi.
- [ ] Tracking configuration tersedia.
- [ ] Experiment dapat dibuat atau digunakan kembali secara deterministik.
- [ ] Tracking URI dapat dikonfigurasi secara lokal.
- [ ] Training parameters tersimpan pada run.
- [ ] Validation metrics tersimpan pada run.
- [ ] Final test metrics tersimpan pada run.
- [ ] Threshold analysis tersimpan sebagai artifact.
- [ ] Dataset metadata tersimpan pada run.
- [ ] Model artifact tersimpan pada run.
- [ ] Production candidate terdaftar di Model Registry.
- [ ] Model version memiliki metadata yang lengkap.
- [ ] Promotion workflow candidate/staging/production terdokumentasi.
- [ ] Model registered dapat dimuat kembali.
- [ ] Training run dapat direproduksi dari configuration.
- [ ] Unit test tracking tersedia dan passed.
- [ ] Integration test MLflow tersedia dan passed.
- [ ] Ruff passed.
- [ ] Format check passed.
- [ ] Pre-commit passed.
- [ ] GitHub Actions tetap green.

## 9. Sprint 3 Completion Criteria

Sprint 3 dianggap selesai apabila setiap training run dapat ditelusuri dari dataset, konfigurasi, parameter, metrics, dan model artifact sampai registered model version. Deployment model ke API bukan bagian dari completion criteria Sprint 3 dan menjadi scope Sprint 4.
