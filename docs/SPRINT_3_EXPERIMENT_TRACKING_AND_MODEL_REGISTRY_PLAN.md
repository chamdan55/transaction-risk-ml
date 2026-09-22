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

### Status saat ini

Capability tracking dan registry sudah tersedia di codebase, tetapi Sprint 3 **belum boleh
ditandai selesai**. Status yang benar adalah **implemented, quality-gate pending**:

- TRM-003 memperbaiki migrasi MLflow 3.x dan sudah diverifikasi melalui unit test serta SQLite
  integration test.
- TRM-004 memperbaiki packaging, dependency profiles, constraints Python 3.12, dan CI smoke test;
  wheel lokal sudah berhasil dibangun dan di-import.
- End-to-end training pipeline yang membutuhkan Spark belum tervalidasi pada environment saat ini
  karena Java/Netty gagal membuat loopback connection.
- GitHub Actions dan pre-commit belum boleh disebut green tanpa bukti eksekusi pada commit yang
  sama.

Detail pekerjaan dan acceptance criteria berada di `docs/tickets/TRM-003-complete-mlflow-name-migration.md`
dan `docs/tickets/TRM-004-packaging-dependencies-ci.md`.

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

Local development menggunakan MLflow tracking URI berbasis SQLite. Tracking storage dan model artifacts tidak boleh masuk Git.

Versi dependency yang didukung adalah `mlflow>=3.0,<4.0`. Wrapper aplikasi tetap menerima nama
argumen internal `artifact_path`, lalu meneruskannya ke MLflow 3.x sebagai `name=artifact_path`.
Field internal `LoggedModel.artifact_path` sengaja dipertahankan agar referensi artifact dan
serialization tetap backward-compatible.

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
├── train_models.py
└── promote_model.py

tests/
├── unit/
│   ├── test_tracking_client.py
│   ├── test_tracking_metadata.py
│   ├── test_tracking_logging.py
│   └── test_tracking_registry.py
└── integration/
    └── test_mlflow_training_run.py
```

Nama modul dapat disesuaikan saat implementasi selama tracking dan registry responsibilities tetap terpisah.

### Evidence matrix

| Capability | Current evidence | Status |
|---|---|---|
| Local SQLite tracking | `configs/tracking.yaml` dan `MlflowTrackingClient` | Terverifikasi |
| Parameter/metric/artifact logging | `pipelines/train_models.py` dan unit tests | Terimplementasi; E2E Spark pending |
| Candidate registration dan load | SQLite integration test | Terverifikasi |
| Explicit candidate → staging/production promotion | `pipelines/promote_model.py` dan integration test | Terverifikasi |
| MLflow 3.x `name` migration | `ml/tracking/client.py`, registry, fakes | Terverifikasi |
| Feature-contract version pada metadata run | Belum dilog sebagai field eksplisit di pipeline | Pending TRM-006 |
| Model signature/input example | Runtime gagal karena inference masih meminta target | Pending TRM-015 |
| Full Spark training run | Blocked oleh loopback Java/Netty pada environment saat ini | Pending supported runtime |
| CI/pre-commit gate | Belum dieksekusi pada commit ini | Pending |

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
  uri: sqlite:///mlflow.db
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
- Model signature dan input example sudah dicoba pada TRM-006, tetapi runtime validation membuktikan
  model wrapper masih meminta target saat inference. Hard gate dan feature-only contract ditangani
  oleh TRM-015 sebelum TRM-007.
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

Status: **Completed.** Candidate alias dibuat saat model pemenang terdaftar. Promosi ke
`staging` atau `production` hanya dapat dilakukan secara eksplisit melalui
`python -m pipelines.promote_model` dan wajib menyertakan approver serta alasan.

### Step 8 — Integrate with Training Pipeline

Tujuan: menjalankan training dan tracking dari satu command.

Implementation:

- Mengintegrasikan MLflow ke `pipelines/train_models.py` atau membuat wrapper pipeline.
- Memulai run untuk setiap model atau satu parent run dengan nested model runs.
- Menjalankan existing candidate selection dari Sprint 2.
- Menjalankan final test evaluation setelah candidate dipilih.
- Menutup run secara aman ketika terjadi exception.
- Menyimpan run ID dan registered model version pada summary.

Status: **Implemented; end-to-end validation pending.** `python -m pipelines.train_models` membaca
`configs/tracking.yaml`, membuat parent run dan nested validation/final runs, mencatat parameter,
metadata dataset, threshold analysis, validation/final-test metrics, evaluation report, serta
mendaftarkan hanya model candidate. Eksekusi penuh yang memerlukan Spark belum dapat dibuktikan pada
environment saat ini.

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
- Pre-commit harus passed pada commit yang sama.
- GitHub Actions harus green pada commit yang sama.

Status: **Partially verified.** Unit test registry/client dan dua SQLite integration test berhasil.
Ruff, format check, compile check, serta wheel smoke test juga berhasil. Test yang memulai Spark
belum dapat dijalankan pada sandbox Windows karena loopback connection Java/Netty gagal; kondisi ini
harus dipisahkan dari hasil assertion test. Pre-commit dan GitHub Actions masih pending evidence.

## 7. Expected Outputs

Setelah Sprint 3 selesai, repository diharapkan memiliki:

- MLflow experiment `transaction-risk-classification`.
- Training run dengan parameter dan metrics lengkap.
- Evaluation report sebagai MLflow artifact.
- Dataset lineage manifest, feature-contract version, model signature, dan input example sebagai
  MLflow artifacts/model metadata (TRM-006).
- Registered model `transaction-risk-model`.
- Production candidate dengan version metadata.
- Model yang dapat dimuat kembali melalui MLflow.
- Command training yang menghasilkan tracking run reproducible.
- Test suite untuk tracking dan registry flow.

## 8. Definition of Done

**Status implementasi: capability tersedia. Status Sprint 3: quality-gate pending.**

Capability utama Step 1–9 telah tersedia di codebase dan flow MLflow registry telah diverifikasi
melalui integration test SQLite. Namun metadata feature-contract/signature, verifikasi end-to-end
yang memerlukan runtime Spark normal, serta gate repository/CI masih terbuka.

- [x] MLflow dependency ditambahkan dan terdokumentasi.
- [x] Tracking configuration tersedia.
- [x] Experiment dapat dibuat atau digunakan kembali secara deterministik.
- [x] Tracking URI dapat dikonfigurasi secara lokal.
- [x] Training parameters tersimpan pada run.
- [x] Validation metrics tersimpan pada run.
- [x] Final test metrics tersimpan pada run.
- [x] Threshold analysis tersimpan sebagai artifact.
- [x] Dataset metadata tersimpan pada run.
- [x] Model artifact tersimpan pada run.
- [x] Production candidate terdaftar di Model Registry.
- [x] Model version memiliki metadata yang lengkap.
- [x] Promotion workflow candidate/staging/production terdokumentasi.
- [x] Model registered dapat dimuat kembali.
- [ ] Feature-contract version tercatat eksplisit pada run/model metadata.
- [ ] Model signature dan representative input example tervalidasi.
- [ ] Training run + tracking dapat direproduksi end-to-end dari configuration pada runtime Spark normal.
- [x] Unit test tracking tersedia dan passed.
- [x] Integration test MLflow tersedia dan passed.
- [x] Ruff passed.
- [x] Format check passed.
- [ ] Full pytest suite passed — belum tervalidasi; test Spark terblokir pada sandbox Windows oleh
  kegagalan loopback Java/Netty.
- [ ] Pre-commit passed pada commit yang sama.
- [ ] GitHub Actions tetap green pada commit yang sama.

## 9. Sprint 3 Completion Criteria

Sprint 3 dianggap selesai apabila setiap training run dapat ditelusuri dari dataset, konfigurasi, parameter, metrics, dan model artifact sampai registered model version. Deployment model ke API bukan bagian dari completion criteria Sprint 3 dan menjadi scope Sprint 4.
