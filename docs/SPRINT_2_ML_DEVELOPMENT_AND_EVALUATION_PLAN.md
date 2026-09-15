# Sprint 2 — ML Development & Evaluation

## 1. Sprint Goal

Membangun, membandingkan, dan memilih model klasifikasi transaction fraud berdasarkan dataset hasil Sprint 1, dengan evaluasi teknis, threshold analysis, dan proses yang dapat direproduksi.

Sprint 2 menggunakan output berikut dari Sprint 1:

```text
data/processed/features/train
data/processed/features/validation
data/processed/features/test
```

Sprint 2 tidak membaca atau memproses ulang raw dataset. Seluruh proses training harus menggunakan feature dataset yang telah divalidasi dan di-split secara kronologis oleh pipeline Sprint 1.

## 2. Scope

### In scope

- Training dataset contract.
- Feature selection dan preprocessing.
- Penanganan class imbalance.
- Training minimal tiga model.
- Model evaluation pada validation dan test dataset.
- Threshold analysis berbasis metrik dan kebutuhan bisnis.
- Pemilihan production candidate.
- Penyimpanan model artifact dan evaluation report.
- Reproducibility melalui konfigurasi dan random seed.
- Unit test dan integration test untuk training serta evaluation flow.

### Out of scope

- Deployment model ke API atau production service.
- Online inference.
- Model serving dan monitoring.
- MLflow atau experiment tracking platform.
- Hyperparameter optimization berskala besar.
- Retraining otomatis.

## 3. Model Strategy

Model yang dibandingkan:

| Model | Peran |
|---|---|
| Logistic Regression | Baseline yang mudah dijelaskan dan diverifikasi |
| Random Forest | Model non-linear pembanding |
| XGBoost | Kandidat utama untuk production |

Karena transaction fraud umumnya merupakan kelas minoritas, accuracy tidak digunakan sebagai metrik utama. Metrik utama yang digunakan adalah PR-AUC, dengan precision, recall, F1, ROC-AUC, dan confusion matrix sebagai metrik pendukung.

## 4. Feature and Leakage Policy

Training feature harus menggunakan `MODEL_FEATURE_COLUMNS` dari modul split Sprint 1. Kolom berikut tidak boleh digunakan sebagai model input:

- `transaction_id`.
- `timestamp`.
- `origin_account_id`.
- `destination_account_id`.
- `is_fraud` sebagai target.
- `is_flagged_fraud` sebagai label atau proxy label.

Fitur yang hanya tersedia setelah transaksi selesai harus ditinjau sebelum training. Secara khusus, `origin_balance_after` dan `destination_balance_after` harus diputuskan berdasarkan waktu prediksi model. Jika model dimaksudkan untuk memprediksi fraud sebelum transaksi diproses, fitur tersebut harus dikeluarkan atau dinyatakan sebagai fitur offline-only.

Keputusan feature availability dan leakage harus dicatat dalam konfigurasi atau evaluation report.

## 5. Target Repository Artifacts

Struktur implementasi yang direncanakan:

```text
configs/
└── model.yaml

ml/
├── training/
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── models.py
│   └── train.py
└── evaluation/
    ├── metrics.py
    ├── threshold.py
    └── report.py

pipelines/
└── train_models.py

tests/
├── unit/
│   ├── test_training_dataset.py
│   ├── test_preprocessing.py
│   └── test_evaluation.py
└── integration/
    └── test_model_training.py

artifacts/
└── models/
```

Nama file dapat disesuaikan saat implementasi selama tanggung jawab modul tetap terpisah dan outputnya konsisten.

## 6. Step-by-Step Implementation Plan

### Step 1 — Define Training Dataset Contract

Tujuan: memastikan training flow hanya menerima dataset hasil split Sprint 1.

Implementation:

- Membaca `train`, `validation`, dan `test` dari path konfigurasi.
- Memvalidasi keberadaan file dan schema setiap split.
- Memastikan target `is_fraud` tersedia.
- Memastikan kolom model input konsisten di ketiga split.
- Memastikan tidak ada overlap `transaction_id` antar split.
- Menyediakan row count dan class distribution sebagai dataset summary.

Output:

- Dataset loader.
- Dataset contract validation.
- Unit test untuk schema, target, dan split isolation.

### Step 2 — Add Model Configuration

Tujuan: menghilangkan parameter training yang hard-coded.

Implementation:

- Menambahkan `configs/model.yaml`.
- Mendefinisikan path input dan output.
- Mendefinisikan target column dan excluded columns.
- Mendefinisikan random seed.
- Mendefinisikan model parameters.
- Mendefinisikan threshold candidates.
- Mendefinisikan metric utama untuk model selection.

Contoh konfigurasi minimum:

```yaml
data:
  features_path: data/processed/features
  target_column: is_fraud

training:
  random_seed: 42
  primary_metric: pr_auc

evaluation:
  thresholds: [0.10, 0.20, 0.30, 0.40, 0.50]
```

### Step 3 — Implement Feature Preparation

Tujuan: menyiapkan input yang konsisten untuk semua model.

Implementation:

- Menggunakan daftar model feature dari Sprint 1.
- Memisahkan feature matrix dan target.
- Menentukan fitur numerik dan kategorikal.
- Melakukan encoding untuk fitur kategorikal.
- Menangani missing value sesuai kontrak dataset.
- Memastikan preprocessing di-fit hanya pada training data.
- Menggunakan preprocessing yang sama untuk validation dan test.

### Step 4 — Handle Class Imbalance

Tujuan: mencegah model mengabaikan kelas fraud yang minoritas.

Implementation:

- Mengukur distribusi target pada setiap split.
- Logistic Regression dan Random Forest menggunakan class weight jika sesuai.
- XGBoost menggunakan parameter imbalance yang terdokumentasi, seperti `scale_pos_weight`.
- Tidak melakukan oversampling sebelum chronological split.
- Mencatat strategi imbalance setiap model dalam report.

### Step 5 — Train Baseline Model

Tujuan: menyediakan baseline yang dapat digunakan untuk membandingkan model lain.

Implementation:

- Melatih Logistic Regression pada training dataset.
- Menghasilkan probabilitas prediksi pada validation dataset.
- Menghitung seluruh metrik yang ditentukan.
- Menyimpan model dan metrics baseline.
- Memastikan proses dapat dijalankan ulang dengan seed dan konfigurasi yang sama.

### Step 6 — Train Comparison Models

Tujuan: membandingkan baseline dengan model non-linear dan production candidate.

Implementation:

- Melatih Random Forest.
- Melatih XGBoost.
- Menggunakan preprocessing dan dataset contract yang sama.
- Menyimpan parameter, training summary, dan metrics setiap model.
- Menangani error training tanpa menghilangkan informasi model yang gagal.

### Step 7 — Evaluate Models

Tujuan: mengukur performa model secara konsisten.

Metrik minimum:

- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.
- Confusion matrix.

Aturan evaluasi:

- Validation set digunakan untuk perbandingan model dan pemilihan threshold.
- Test set hanya digunakan untuk final evaluation setelah model dan threshold dipilih.
- Tidak boleh memilih model berdasarkan test set.
- Hasil evaluasi disimpan dalam format yang dapat dibaca ulang, minimal JSON.

### Step 8 — Perform Threshold and Business Analysis

Tujuan: menentukan threshold yang sesuai dengan trade-off false positive dan false negative.

Implementation:

- Mengevaluasi beberapa threshold pada validation set.
- Menghasilkan precision, recall, F1, confusion matrix, dan jumlah prediksi fraud pada setiap threshold.
- Mendefinisikan rekomendasi threshold berdasarkan prioritas bisnis.
- Mencatat cost assumption jika business cost tersedia.
- Menyimpan threshold terpilih dalam evaluation report.

### Step 9 — Select Production Candidate and Run Final Test

Tujuan: menentukan kandidat model berdasarkan aturan yang transparan.

Implementation:

- Memilih production candidate berdasarkan primary metric dan constraint minimum.
- Menjalankan final evaluation pada test set satu kali.
- Menyimpan nama model, parameter, threshold, dan final metrics.
- Memastikan test set tidak digunakan untuk tuning setelah final result dibuat.

### Step 10 — Add Training Pipeline and Reproducibility Checks

Tujuan: menyediakan satu entry point yang dapat digunakan ulang.

Implementation:

- Menambahkan `pipelines/train_models.py`.
- Menambahkan target Makefile jika diperlukan, misalnya `make train-models`.
- Menyediakan input konfigurasi melalui file config.
- Menyimpan output ke direktori artifact yang konsisten.
- Memastikan run kedua dengan input dan konfigurasi yang sama menghasilkan schema dan metrics yang konsisten.

### Step 11 — Add Tests and Quality Gates

Minimum test coverage:

- Dataset loader membaca seluruh split.
- Target dan model feature tervalidasi.
- Kolom identifier dan target tidak masuk ke feature matrix.
- Preprocessing tidak mengalami data leakage.
- Model dapat dilatih pada sample dataset.
- Metrics dan threshold analysis menghasilkan output yang valid.
- Integration pipeline menghasilkan model artifact dan evaluation report.

Quality checks:

- Pytest passed.
- Ruff passed.
- Format check passed.
- Pre-commit passed.
- GitHub Actions tetap green.

## 7. Expected Outputs

Setelah Sprint 2 selesai, repository diharapkan memiliki:

- Tiga model yang telah dibandingkan.
- Baseline Logistic Regression.
- Validation metrics dan threshold analysis.
- Final test metrics untuk model terpilih.
- Production candidate yang terdokumentasi.
- Model artifact yang dapat dimuat kembali.
- Evaluation report yang dapat dibaca manusia dan mesin.
- Konfigurasi training yang dapat digunakan ulang.
- Unit dan integration test untuk training flow.

## 8. Definition of Done

- [x] Training dataset contract diimplementasikan.
- [x] Train, validation, dan test Parquet dapat dibaca kembali.
- [x] Model input tidak mengandung identifier, target, atau proxy label.
- [x] Feature availability dan leakage policy terdokumentasi.
- [x] Preprocessing konsisten antara train, validation, dan test.
- [x] Class imbalance strategy diterapkan dan terdokumentasi.
- [x] Logistic Regression berhasil dilatih sebagai baseline.
- [x] Random Forest berhasil dilatih.
- [x] XGBoost berhasil dilatih.
- [x] Minimal tiga model dibandingkan.
- [x] Precision, recall, F1, ROC-AUC, dan PR-AUC tersedia.
- [x] Confusion matrix tersedia.
- [x] Threshold analysis tersedia.
- [x] Production candidate ditentukan berdasarkan validation result.
- [x] Final evaluation dilakukan pada test set.
- [x] Model artifact dapat dimuat kembali.
- [x] Evaluation report dapat dibaca kembali.
- [x] Training dapat direproduksi dari konfigurasi.
- [x] Unit test tersedia dan passed.
- [x] Integration test tersedia dan passed.
- [x] Ruff passed.
- [x] Format check passed.
- [x] Pre-commit passed.
- [x] GitHub Actions tetap green.

## 9. Sprint 2 Completion Criteria

Sprint 2 dianggap selesai apabila seluruh Definition of Done terpenuhi dan hasil final evaluation sudah terdokumentasi. Model yang dipilih belum dianggap production-ready untuk deployment; deployment, serving, dan monitoring menjadi scope sprint lanjutan.
