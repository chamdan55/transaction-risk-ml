# Sprint 1 — Transaction Data Engineering Pipeline Design

## 1. Objective

Sprint 1 builds the data engineering foundation for the **Transaction Risk Scoring ML Platform**.

The target flow is:

```text
Public / Synthetic Raw Dataset
        │
        ▼
   Raw Ingestion
        │
        ▼
      PySpark
        │
        ├── Schema validation
        ├── Data quality checks
        ├── Cleaning
        ├── Normalization
        ├── Temporal enrichment
        ├── Behavioral feature engineering
        └── Train / Validation / Test split
        │
        ▼
     Parquet
        │
        ├── processed/
        └── features/
```

The output of this sprint must be a **reproducible feature dataset** that can be consumed by Sprint 2 for model training.

---

# 2. Dataset Strategy

## 2.1 Primary dataset: PaySim

For the initial implementation, use **PaySim** as the source dataset.

PaySim is a synthetic mobile-money transaction dataset generated from a simulator. It contains transaction-level attributes and a fraud label, making it suitable for demonstrating an end-to-end fraud/risk-scoring pipeline without using proprietary banking data.

Important characteristics of the dataset:

- Approximately 6.3 million transactions
- Synthetic transaction data
- Contains fraudulent and non-fraudulent transactions
- Includes transaction type
- Includes transaction amount
- Includes origin and destination account identifiers
- Includes account balance information
- Includes fraud labels

The raw dataset should remain isolated from the production-style feature pipeline.

### Important principle

We do **not** make the ML system dependent on PaySim's original column names.

Instead:

```text
PaySim
   │
   ▼
Canonical Transaction Schema
   │
   ▼
Feature Engineering
   │
   ▼
ML Dataset
```

This makes the project architecture reusable if another dataset is introduced later.

---

# 3. Canonical Transaction Schema

The pipeline should normalize the source dataset into the following logical schema.

| Column | Type | Required | Description |
|---|---|---:|---|
| `transaction_id` | string | Yes | Unique transaction identifier |
| `timestamp` | timestamp | Yes | Transaction event time |
| `transaction_type` | string | Yes | Transaction category |
| `amount` | double | Yes | Transaction amount |
| `origin_account_id` | string | Yes | Source account identifier |
| `destination_account_id` | string | Yes | Destination account identifier |
| `origin_balance_before` | double | Yes | Origin balance before transaction |
| `origin_balance_after` | double | Yes | Origin balance after transaction |
| `destination_balance_before` | double | Yes | Destination balance before transaction |
| `destination_balance_after` | double | Yes | Destination balance after transaction |
| `is_fraud` | integer | Yes | Target label: 1 = fraud, 0 = non-fraud |
| `is_flagged` | integer | No | Original rule-based fraud flag, if available |

## 3.1 Source mapping

For PaySim:

| Canonical field | PaySim field |
|---|---|
| `transaction_id` | generated from deterministic row/source identifier |
| `timestamp` | derived from `step` |
| `transaction_type` | `type` |
| `amount` | `amount` |
| `origin_account_id` | `nameOrig` |
| `destination_account_id` | `nameDest` |
| `origin_balance_before` | `oldbalanceOrg` |
| `origin_balance_after` | `newbalanceOrig` |
| `destination_balance_before` | `oldbalanceDest` |
| `destination_balance_after` | `newbalanceDest` |
| `is_fraud` | `isFraud` |
| `is_flagged` | `isFlaggedFraud` |

The exact source-to-canonical transformation must be implemented in code rather than manually modifying the dataset.

---

# 4. Data Layers

The pipeline uses three logical data layers.

## 4.1 Raw

```text
data/raw/
```

Contains the original downloaded/source dataset.

Rules:

- Never modify raw files in place.
- Raw data is not committed to Git.
- Raw data is treated as immutable input.
- Pipeline code reads from raw and creates downstream datasets.

Example:

```text
data/raw/
└── paysim.csv
```

---

## 4.2 Processed

```text
data/processed/
```

Contains cleaned and normalized transaction data.

Example:

```text
data/processed/
└── transactions/
    ├── part-*.parquet
    └── ...
```

This layer contains the canonical transaction schema.

---

## 4.3 Features

Although the current `.gitignore` ignores `data/processed/`, the feature dataset should conceptually be separated from the normalized transaction dataset.

Recommended logical structure:

```text
data/
├── raw/
├── processed/
│   └── transactions/
└── sample/
    └── transactions_sample.parquet
```

For the first implementation, feature datasets can live under `data/processed/features/`:

```text
data/processed/
├── transactions/
└── features/
```

This avoids introducing unnecessary storage infrastructure during Sprint 1.

---

# 5. Data Quality Rules

Data validation should happen immediately after ingestion and before feature engineering.

## 5.1 Schema validation

The pipeline must verify:

- Required columns exist.
- Column types are compatible.
- Target column exists.
- Numeric fields are numeric.
- Timestamp is valid.

Pandera should be used for explicit schema validation where practical.

---

## 5.2 Transaction ID

Requirements:

```text
transaction_id != null
transaction_id is unique
transaction_id is deterministic
```

A duplicate transaction ID is considered a data quality failure.

---

## 5.3 Amount

Requirements:

```text
amount >= 0
amount is not null
```

Negative transaction amounts should be rejected or explicitly handled as invalid records.

---

## 5.4 Balance fields

Requirements:

```text
origin_balance_before >= 0
origin_balance_after >= 0
destination_balance_before >= 0
destination_balance_after >= 0
```

Small inconsistencies may exist in simulated datasets. They should be measured rather than silently corrected.

---

## 5.5 Fraud label

Requirements:

```text
is_fraud ∈ {0, 1}
```

Null labels are not allowed in the training dataset.

---

## 5.6 Transaction type

`transaction_type` must belong to the known source categories.

Unknown categories should cause a validation failure rather than being silently converted to another class.

---

# 6. Cleaning Strategy

Cleaning should be deterministic and auditable.

## 6.1 Missing values

Do not blindly impute all missing values.

Recommended policy:

| Field | Strategy |
|---|---|
| IDs | Reject |
| Timestamp | Reject |
| Transaction type | Reject |
| Amount | Reject |
| Target label | Reject for training |
| Balance fields | Investigate first; then define explicit policy |

Every rejected record should be measurable.

Example quality output:

```text
input_records: 6,362,620
valid_records: ...
rejected_records: ...
duplicate_records: ...
missing_value_records: ...
```

---

# 7. Temporal Representation

PaySim provides a `step` value representing transaction time in hourly increments.

For the canonical schema:

```text
timestamp = base_timestamp + step hours
```

Example:

```text
base_timestamp = 2026-01-01 00:00:00
step = 10
timestamp = 2026-01-01 10:00:00
```

The exact base timestamp is synthetic and should be documented.

### Important limitation

Because the source provides simulated time rather than real-world timestamps, temporal features should be interpreted as **behavioral simulation features**, not real banking-calendar behavior.

---

# 8. Feature Engineering

Feature engineering is the most important part of Sprint 1.

The objective is to transform individual transactions into features describing:

1. Transaction characteristics
2. Account behavior
3. Merchant/destination behavior
4. Recent transaction activity
5. Historical transaction patterns

---

## 8.1 Transaction-level features

Basic features:

```text
amount
transaction_type
hour
day_of_week
```

Derived features:

```text
amount_log
is_night
is_large_transaction
```

Example:

```text
amount_log = log1p(amount)
```

This helps reduce the impact of highly skewed transaction amounts.

---

# 9. Balance-based Features

Useful derived features:

```text
origin_balance_change
destination_balance_change
origin_balance_ratio
amount_to_origin_balance_ratio
amount_to_destination_balance_ratio
```

Examples:

```text
origin_balance_change =
    origin_balance_after - origin_balance_before

destination_balance_change =
    destination_balance_after - destination_balance_before

amount_to_origin_balance_ratio =
    amount / (origin_balance_before + 1)
```

The `+1` denominator guard should be applied consistently and documented.

---

# 10. Behavioral Features

Behavioral features should use historical information available **before the current transaction**.

This is critical to avoid target leakage.

## 10.1 Recent transaction count

Examples:

```text
transactions_last_1h
transactions_last_24h
transactions_last_7d
```

Calculated per origin account.

---

## 10.2 Recent transaction amount

Examples:

```text
amount_sum_last_24h
amount_avg_last_7d
amount_max_last_30d
```

---

## 10.3 Account activity

Examples:

```text
unique_destinations_last_30d
unique_transaction_types_last_30d
```

---

## 10.4 Destination behavior

Examples:

```text
destination_transaction_count
destination_amount_sum
destination_amount_avg
```

These features help capture unusual behavior around destination accounts.

## 10.5 Step 5 initial behavioral features

The initial Step 5 implementation includes four historical account-behavior
features:

```text
transactions_last_1h
transactions_last_24h
amount_sum_last_24h
unique_destinations_last_30d
```

These features are calculated with PySpark range windows partitioned by
`origin_account_id` and ordered by `timestamp`. The window ends strictly
before the current transaction, so the current row and future rows cannot
contribute to its own behavioral features.

---

# 11. Spark Window Strategy

Behavioral features should be implemented using PySpark window operations where appropriate.

Conceptually:

```text
partitionBy(origin_account_id)
orderBy(timestamp)
```

For example:

```text
origin_account_id
        │
        ├── transaction 1
        ├── transaction 2
        ├── transaction 3
        └── transaction 4
                │
                ▼
      historical window
```

The current transaction must not contribute to its own historical features.

### Rule

For a transaction at time `T`:

```text
feature(T) = information available strictly before T
```

Never:

```text
feature(T) = information from T and future transactions
```

This is one of the most important ML engineering rules in this project.

---

# 12. Categorical Features

Initial categorical fields:

```text
transaction_type
```

Potential future categorical fields:

```text
origin_account_segment
destination_segment
```

For Sprint 1, do not over-engineer categorical encoding.

The canonical dataset should retain meaningful categorical values.

Encoding belongs primarily to the ML preprocessing stage in Sprint 2.

---

# 13. Train / Validation / Test Split

A random split is **not preferred** for the primary experiment because this is a temporal transaction-risk problem.

Use a chronological split.

Example:

```text
Historical data
─────────────────────────────────────────────────────────────>

|--------- Train ---------|---- Validation ----|---- Test ----|

        70%                     15%                 15%
```

The exact proportions can be adjusted after inspecting the dataset.

Recommended rule:

```text
Train:
    earliest transactions

Validation:
    subsequent transactions

Test:
    latest transactions
```

This better simulates production:

```text
Past data → train model
Recent data → validate
Future unseen data → test
```

---

# 14. Preventing Data Leakage

The pipeline must explicitly prevent leakage.

## Forbidden

Features that use:

- Future transactions
- Future account behavior
- The fraud label
- Post-transaction information that would not be available at prediction time

Examples of problematic features:

```text
future_transaction_count
future_fraud_count
is_fraud
```

The target:

```text
is_fraud
```

must never be included in model features.

---

# 15. Output Dataset

The final ML-ready dataset should be stored as Parquet.

Example:

```text
data/processed/features/
├── train/
├── validation/
└── test/
```

Each dataset should contain:

```text
transaction_id
timestamp
feature_1
feature_2
...
feature_n
is_fraud
```

The exact feature list will be finalized after exploratory analysis.

---

# 16. Parquet Partitioning

Do not blindly partition by high-cardinality columns such as:

```text
origin_account_id
transaction_id
```

This can create excessive small files.

For the first implementation, partitioning should be modest.

Possible partition:

```text
year
month
```

or:

```text
dataset_split
```

The implementation should prioritize practical file sizes and reproducibility over theoretical partitioning complexity.

---

# 17. Reproducibility

The pipeline must be deterministic.

Configuration should control:

```text
random_seed
dataset_path
output_path
base_timestamp
train_ratio
validation_ratio
test_ratio
```

Example configuration:

```yaml
data:
  input_path: data/raw/paysim.csv
  output_path: data/processed

split:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15

reproducibility:
  seed: 42

time:
  base_timestamp: "2026-01-01T00:00:00"
```

The configuration should be version-controlled.

---

# 18. Proposed Project Structure After Sprint 1

```text
transaction-risk-ml/
├── app/
│   ├── api/
│   │   └── routes/
│   ├── core/
│   ├── inference/
│   └── main.py
│
├── ml/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   └── cleaning.py
│   │
│   ├── features/
│   │   ├── transaction_features.py
│   │   ├── behavioral_features.py
│   │   └── pipeline.py
│   │
│   ├── training/
│   │
│   └── monitoring/
│
├── pipelines/
│   └── build_features.py
│
├── configs/
│   └── data.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── model/
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── deployment/
│   ├── docker/
│   └── kubernetes/
│
├── monitoring/
│   ├── prometheus/
│   └── grafana/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── pyproject.toml
├── Makefile
└── README.md
```

---

# 19. Sprint 1 Implementation Breakdown

## Step 5: Feature Engineering
### __Objective__
Mengubah canonical transaction dataset:

> data/processed/canonical/

menjadi feature dataset yang siap digunakan untuk:
```
                 Canonical Transactions
                         │
                         ▼
                Feature Engineering
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Amount Features  Balance Features  Transaction
                                        Behavior
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Feature Dataset
                         │
                         ▼
                  Model Training
```
### __Prinsip utama__

Kita akan menghindari feature yang menyebabkan data leakage.

Karena target kita adalah __`is_fraud`__, feature engineering hanya boleh menggunakan informasi yang secara logis tersedia pada atau sebelum transaksi tersebut diproses.

---
### Step 5.1 — Tentukan Feature Contract

Sebelum coding, kita define dulu feature yang akan kita hasilkan.

Aku menyarankan initial feature set berikut.

#### __A. Transaction features__
| Feature | Deskripsi |
| :--- | ---: |
| `amount` | Nilai transaksi |
|`amount_log` | __`log1p(amount)`__ |
|`transaction_type` | Tipe transaksi |
|`is_cash_in` | Indicator `CASH_IN` |
|`is_cash_out` | Indicator `CASH_OUT` |
|`is_debit` | Indicator `DEBIT` |
|`is_payment` | Indicator `PAYMENT` |
|`is_transfer` | Indicator `TRANSFER` |

#### __B. Origin balance features__
| Feature | Deskripsi |
| :--- | ---: |
|`origin_balance_before` | Saldo sebelum transaksi
|`origin_balance_after` | Saldo setelah transaksi
|`origin_balance_delta` | `after - before`
|`origin_balance_change_ratio` | Perubahan relatif terhadap saldo awal
|`amount_to_origin_balance_ratio` | Amount dibanding saldo awal

Contoh:

> origin_balance_delta = origin_balance_after - origin_balance_before

dan:

> amount_to_origin_balance_ratio = amount / origin_balance_before

Untuk denominator `0`, __jangan menghasilkan infinity__. Kita perlu menentukan policy eksplisit, misalnya `0.0` atau `NULL`, lalu didokumentasikan.

Untuk risk modeling, aku lebih menyukai:

> NULL → kemudian ditangani pada preprocessing/model pipeline

karena:

> origin_balance_before = 0

memang memiliki makna bisnis tersendiri.

#### __C. Destination balance features__

Analogous:
```
destination_balance_before
destination_balance_after
destination_balance_delta
amount_to_destination_balance_ratio
```

#### __D. Balance consistency features__

Ini menarik karena kita sudah melakukan domain profiling di Step 4.

Kita bisa membawa hasil domain knowledge tersebut menjadi feature.

Misalnya:
```
origin_balance_mismatch
destination_balance_mismatch
```
sehingga model dapat belajar bahwa transaksi dengan balance behavior tertentu mungkin memiliki risk yang berbeda.

#### __E. Transaction/account behavior__

Untuk tahap awal, kita jangan langsung membuat fitur yang membutuhkan window aggregation kompleks.

Kita bisa mulai dari:
```
is_zero_origin_balance_before
is_zero_destination_balance_before
origin_balance_depleted
destination_balance_increased
```
Misalnya:
```
origin_balance_depleted =
    origin_balance_after == 0
```
Ini cukup meaningful untuk transaction-risk modeling.

Selain indicator transaksi, Step 5 juga mengimplementasikan historical
behavior features berikut:

```text
transactions_last_1h
transactions_last_24h
amount_sum_last_24h
unique_destinations_last_30d
```

Semua historical features hanya menggunakan transaksi strictly sebelum
transaksi saat ini. Transaksi dengan timestamp yang sama juga tidak ikut
dihitung agar tidak ada kontribusi current-row yang ambigu.

---
### Step 5.2 — Timestamp Features

Kita punya:

> timestamp

Maka kita dapat derive:
```
transaction_hour
transaction_day
transaction_day_of_week
```
Tetapi __jangan langsung memasukkan raw timestamp ke model__.

Untuk model klasik, lebih baik kita derive:
```
transaction_hour
transaction_day_of_week
```
dan nantinya kita bisa mempertimbangkan cyclical encoding:
```
hour_sin
hour_cos
```
Namun untuk Step 5 initial implementation, cukup:
```
transaction_hour
transaction_day_of_week
```

---
### Step 5.3 — Target

__`is_fraud` bukan feature__.

Ia adalah:

> target

Jadi dataset konseptual kita:
```
features:
    amount
    amount_log
    transaction_type
    ...

target:
    is_fraud
```
Ini penting untuk menjaga boundary antara:
```
feature engineering
        ↓
training
```

---
### Step 5.4 — Proposed Feature Dataset

Output akhirnya kira-kira:

> data/processed/features/

dengan schema seperti:
```
transaction_id
timestamp

transaction_type

amount
amount_log

origin_balance_before
origin_balance_after
origin_balance_delta
amount_to_origin_balance_ratio

destination_balance_before
destination_balance_after
destination_balance_delta
amount_to_destination_balance_ratio

origin_balance_mismatch
destination_balance_mismatch

is_zero_origin_balance_before
is_zero_destination_balance_before
origin_balance_depleted
destination_balance_increased

transactions_last_1h
transactions_last_24h
amount_sum_last_24h
unique_destinations_last_30d

transaction_hour
transaction_day_of_week

is_fraud
```
Account IDs kemungkinan __belum kita masukkan langsung sebagai model feature__.

Ini keputusan yang disengaja.

`origin_account_id` dan `destination_account_id` memiliki cardinality sangat tinggi dan naive encoding bisa menghasilkan feature space yang buruk serta berpotensi membuat model menghafal entity tertentu.

Nanti kita bisa membuat __behavioral/account-level__ features dengan window aggregation secara khusus.

---
### Step 5.5 — Struktur Code

Aku sarankan mulai dengan:
```
ml/
└── features/
    ├── __init__.py
    ├── transaction.py
    ├── behavior.py
    └── schema.py
```
dan pipeline:
```
pipelines/
└── build_features.py
```
Tests:
```
tests/
└── unit/
    └── test_features.py
```
Jadi separation-nya:
```
ml/data/
    ↓
data ingestion + validation

ml/features/
    ↓
feature transformation

pipelines/
    ↓
orchestration / executable workflow

tests/
    ↓
verification
```
Ini jauh lebih scalable daripada menaruh seluruh feature logic di `pipelines/build_features`.py.

---
### Step 5.6 — PySpark Strategy

Karena kita sudah sepakat menggunakan __PySpark__, feature engineering juga kita lakukan menggunakan Spark DataFrame API.

Contohnya secara konsep:
```python
df = df.withColumn("amount_log", F.log1p("amount")).withColumn(
    "origin_balance_delta",
    F.col("origin_balance_after") - F.col("origin_balance_before"),
)
```

Bukan:
```python
df.toPandas()
```
Jadi pipeline kita tetap:
```
Parquet
  ↓
Spark DataFrame
  ↓
Spark transformations
  ↓
Parquet
```
Tidak ada conversion ke Pandas.

Ini konsisten dengan positioning project sebagai __large-scale transaction ML pipeline__.

---
### Step 5.7 — Feature Engineering Rules

Kita juga perlu membuat beberapa helper function supaya logic tidak menjadi monolithic.

Misalnya:
```
build_transaction_features()
build_balance_features()
build_behavior_features()
build_time_features()
```
Kemudian:
```python
def build_features(df):
    df = build_transaction_features(df)
    df = build_balance_features(df)
    df = build_behavior_features(df)
    df = build_time_features(df)

    return df
```
Dengan begitu nantinya testing bisa granular.

---
### Step 5.8 — Testing

Minimal kita ingin test:

#### __Amount__
```
amount = 100
→ amount_log = log1p(100)
```
#### __Balance delta__
```
before = 1000
after = 700
→ delta = -300
```
#### __Ratio__
```
amount = 100
origin_balance_before = 1000
→ ratio = 0.1
```
#### __Zero denominator__
```
origin_balance_before = 0
→ ratio tidak boleh infinity
```
#### __Timestamp__
```
timestamp = known datetime
→ hour benar
→ day_of_week benar
```
#### __Transaction type__
```
PAYMENT
→ is_payment = 1
→ is_transfer = 0
```
#### __Target preservation__

Pastikan:

> is_fraud

tetap ada dan tidak berubah.

#### __Behavioral leakage__

Untuk historical features, test juga harus memastikan:

```text
transaksi masa depan tidak memengaruhi transaksi saat ini
transaksi pada timestamp yang sama tidak dihitung sebagai histori
```

---
### Step 5.9 — Feature Schema Validation

Kita juga sebaiknya punya:
```
FEATURE_SCHEMA
```
seperti kita punya:
```
PAYSIM_SCHEMA
CANONICAL_TRANSACTION_SCHEMA
```
Sehingga pipeline berikutnya bisa melakukan:
```
canonical schema
        ↓
feature transformation
        ↓
feature schema validation
        ↓
training
```
Ini akan sangat berguna ketika nanti kita masuk ke: __Step 6 — Dataset Splitting & Training Preparation__.

---
### Step 5.10 — Pipeline

Nantinya command-nya:
```pwsh
python -m pipelines.build_features
```
Expected flow:
```
Reading canonical dataset
        ↓
Validating canonical input
        ↓
Building transaction features
        ↓
Building balance features
        ↓
Building behavioral features
        ↓
Building timestamp features
        ↓
Validating feature schema
        ↓
Writing feature dataset
        ↓
Feature engineering completed
```
Output:
> data/processed/features/

### Step 5 Definition of Done

Step 5 baru kita anggap __DONE__ kalau:
- [ ✓ ]  Feature contract didefinisikan
- [ ✓ ]  Feature transformations implemented
- [ ✓ ]  PySpark-only transformation
- [ ✓ ]  Feature schema defined
- [ ✓ ]  Feature pipeline implemented
- [ ✓ ]  Feature dataset successfully written
- [ ✓ ]  No NaN/infinity yang tidak terkontrol
- [ ✓ ]  Target is_fraud preserved
- [ ✓ ]  No obvious target leakage
- [ ✓ ]  Unit tests implemented
- [ ✓ ]  Unit tests passed
- [ ✓ ]  Ruff passed
- [ ✓ ]  pre-commit passed
- [ ✓ ]  Pipeline berhasil dijalankan terhadap full 6.36M rows
- [ ✓ ]  Feature output dapat dibaca kembali oleh Spark

---

## Step 6 — Dataset Splitting & Training Preparation

Step 6 menggunakan feature dataset hasil Step 5 dan menyiapkan dataset yang
siap dikonsumsi oleh Sprint 2. Step ini belum melakukan training model.

### Step 6.1 — Chronological split

Create:

```text
ml/data/split.py
```

Dataset dibagi berdasarkan urutan waktu, bukan random split:

```text
earliest transactions   → train
subsequent transactions → validation
latest transactions     → test
```

Default ratio dikontrol oleh `configs/data.yaml`:

```yaml
split:
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
```

Boundary split harus deterministik dan tidak boleh mencampurkan transaksi
masa depan ke dalam train dataset.

### Step 6.2 — Target and metadata boundary

Dataset split harus mempertahankan:

```text
is_fraud
```

sebagai target, bukan sebagai model feature. Kolom berikut diperlakukan
sebagai identifier/metadata dan tidak boleh masuk ke feature vector secara
naive:

```text
transaction_id
origin_account_id
destination_account_id
timestamp
```

Feature columns harus berasal dari explicit feature contract Step 5.

### Step 6.3 — Split output

Output disimpan sebagai Parquet:

```text
data/processed/
├── canonical/
└── features/
    ├── part-*.parquet
    ├── train/
    ├── validation/
    └── test/
```

Dataset Parquet pada root `features/` merupakan output feature assembly Step 5. Dataset `train/`,
`validation/`, dan `test/` merupakan output Step 6.

### Step 6.4 — Split validation

Validasi minimum:

```text
train row count + validation row count + test row count = all row count
max(train timestamp) <= min(validation timestamp)
max(validation timestamp) <= min(test timestamp)
schema konsisten antar split
is_fraud tetap tersedia
transaction_id tidak duplikat antar split
```

### Step 6 Definition of Done

- [✓] `ml/data/split.py` implemented
- [✓] Chronological split implemented
- [✓] Split ratios read from configuration
- [✓] Train/validation/test Parquet written
- [✓] Row counts reconcile with the all-feature dataset
- [✓] Temporal boundaries validated
- [✓] Target preserved
- [✓] No identifier leakage into the feature vector
- [✓] Unit tests implemented
- [✓] Integration test implemented

---

## Step 7 — End-to-End Data Pipeline

Step 7 menyatukan seluruh proses Sprint 1 menjadi satu workflow yang dapat
dijalankan ulang secara reproducible.

### Step 7.1 — Pipeline orchestrator

Create:

```text
pipelines/run_pipeline.py
```

Command utama:

```pwsh
python -m pipelines.run_pipeline
```

Orchestrator menjalankan:

```text
raw ingestion
    ↓
raw validation
    ↓
canonical transformation
    ↓
canonical validation
    ↓
domain validation and profiling
    ↓
feature engineering
    ↓
feature validation
    ↓
chronological split
    ↓
train/validation/test Parquet
```

### Step 7.2 — Failure semantics

Pipeline harus fail fast. Jika validasi raw atau canonical gagal, proses
berhenti dan tidak boleh menghasilkan dataset downstream yang dianggap valid.

### Step 7.3 — Idempotency

Pipeline harus aman dijalankan ulang. Untuk local development, output
menggunakan `mode("overwrite")` atau strategi setara sehingga tidak terjadi
append data atau duplikasi output.

### Step 7.4 — Logging and summary

Pipeline mencatat tahapan dan summary minimum:

```text
input rows
canonical rows
feature rows
train/validation/test rows
feature column count
target column
output paths
```

### Step 7.5 — End-to-end test

Gunakan fixture kecil 5–20 transaksi di `data/sample/`. Test harus
memverifikasi bahwa raw-to-split workflow dapat berjalan, output tersedia,
row count terjaga, schema konsisten, dan target tetap ada.

### Step 7 Definition of Done

- [✓] `pipelines/run_pipeline.py` implemented
- [✓] Fail-fast behavior verified
- [✓] Idempotent execution verified
- [✓] Pipeline logging implemented
- [✓] End-to-end integration test implemented and passed
- [✓] Makefile command `make pipeline` tersedia
- [✓] Full pipeline completes successfully

---

# 20. Testing Strategy

Sprint 1 must include tests.

### Unit tests

Test:

```text
amount_log
balance_change
balance_ratio
timestamp conversion
schema validation
```

### Feature tests

Verify:

```text
current transaction is excluded from historical windows
future transactions are excluded
feature values are deterministic
```

### Data pipeline integration test

Use a small dataset under:

```text
data/sample/
```

Pipeline should be able to execute end-to-end:

```text
sample input
    ↓
PySpark
    ↓
validation
    ↓
cleaning
    ↓
features
    ↓
chronological split
    ↓
Parquet output
```

The integration test should use the small sample rather than the full dataset.

---

# 21. Definition of Done

Sprint 1 is complete when:

- [ ✓ ]  Raw dataset is available locally
- [ ✓ ]  Canonical transaction schema is implemented
- [ ✓ ]  PySpark ingestion works
- [ ✓ ]  Data quality validation works
- [ ✓ ]  Cleaning/transformation works
- [ ✓ ]  Transaction-level features work
- [ ✓ ]  Behavioral features work
- [ ✓ ]  Leakage prevention is tested
- [ ✓ ]  Chronological train/validation/test split works
- [ ✓ ]  Output is stored as Parquet
- [ ✓ ]  Sample dataset exists under `data/sample/`
- [ ✓ ]  Unit tests exist
- [ ✓ ]  Integration test exists
- [ ✓ ]  Ruff passes
- [ ✓ ]  Pytest passes
- [ ] GitHub Actions remains green
- [ ✓ ]  Pipeline can be reproduced from configuration

---

## Definition of Done per Scope — Sprint 1

Kalau kita mengikuti plan ini, Sprint 1 baru benar-benar selesai ketika:

### __Data Engineering__
- [ ✓ ] Raw dataset ingestion menggunakan PySpark
- [ ✓ ] Raw schema validation
- [ ✓ ] Canonical transaction schema
- [ ✓ ] Canonical transformation
- [ ✓ ] Canonical dataset Parquet
- [ ✓ ] Canonical validation
- [ ✓ ] Domain validation
- [ ✓ ] Domain profiling

### __Feature Engineering__
- [ ✓ ] Timestamp features
- [ ✓ ] Amount features
- [ ✓ ] Balance features
- [ ✓ ] Behavioral features
- [ ✓ ] Feature dataset assembly
- [ ✓ ] Feature dataset validation
- [ ✓ ] Feature Parquet

### __Dataset Splitting__
- [ ✓ ] Chronological train/validation/test split
- [ ✓ ] Split validation
- [ ✓ ] Train/validation/test Parquet

### __Pipeline Engineering__
- [ ✓ ] End-to-end orchestrator
- [ ✓ ] Fail-fast behavior
- [ ✓ ] Idempotent execution
- [ ✓ ] Pipeline logging
- [ ✓ ] Integration test
- [ ✓ ] Makefile pipeline command

### __Quality__
- [ ✓ ] pytest
- [ ✓ ] Ruff
- [ ✓ ] formatting
- [ ✓ ] pre-commit
- [ ✓ ] GitHub Actions CI
- [ ✓ ] final end-to-end pipeline green

---
# 22. Sprint 1 Success Criteria

The most important outcome is not the number of features.

The success criterion is:

> Given the same raw input and configuration, the pipeline produces a validated, deterministic, leakage-safe Parquet dataset suitable for ML training.

The resulting architecture should demonstrate that the project can handle the **data engineering side of a production ML lifecycle**, rather than only demonstrating model training.

---

# 23. What Comes Next

After Sprint 1:

```text
Raw Transactions
       ↓
     PySpark
       ↓
Validation + Cleaning
       ↓
Feature Engineering
       ↓
Train / Validation / Test
       ↓
Parquet
       │
       ▼
   Sprint 2
ML Development
       ↓
Logistic Regression
Random Forest
XGBoost
       ↓
Evaluation
       ↓
Model Selection
```

Sprint 2 will use the output of this pipeline rather than reading the raw dataset directly.
