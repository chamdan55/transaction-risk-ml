# Step 6 — Feature Dataset Assembly
## Objective
Menghasilkan __dataset feature__ yang siap dikonsumsi oleh ML pipeline.

Di Step 5 kita baru menghasilkan individual feature transformations. Step 6 tugasnya menyatukan semuanya menjadi satu dataset yang mempunyai:
- stable schema
- deterministic feature columns
- target label
- no accidental raw fields
- reproducible output
- Parquet format

Konsepnya:
```
Canonical Dataset
       │
       ├── Timestamp Features
       │
       ├── Amount Features
       │
       ├── Balance Features
       │
       └── Transaction Features
                │
                ▼
       Feature Dataset
                │
                ├── features
                └── target
                │
                ▼
       data/processed/features
```

---
## Step 6.1 — Define Feature Schema

Kita perlu menentukan secara eksplisit feature contract.

Misalnya:
```
transaction_id
transaction_type
timestamp-derived features
amount-derived features
balance-derived features
is_fraud
```
Contoh:
```
transaction_id
transaction_type
transaction_hour
transaction_day_of_week
amount
amount_log
origin_balance_before
origin_balance_after
destination_balance_before
destination_balance_after
origin_balance_ratio
destination_balance_ratio
balance_delta
is_fraud
```
**Penting**: jangan memasukkan seluruh kolom canonical secara otomatis.

Kita ingin menghindari:
```python
df.select("*")
```
karena feature set harus eksplisit.

### Deliverables
Tambahkan semacam feature contract/configuration:
> ml/data/feature_schema.py

atau jika struktur existing lebih cocok:
> ml/features/schema.py

---
## Step 6.2 — Implement Feature Assembly

Buat satu function yang menjadi entry point.

Misalnya secara konsep:
```python
def build_feature_dataset(df: DataFrame) -> DataFrame: ...
```

Pipeline-nya:
```
canonical_df
      │
      ▼
timestamp features
      │
      ▼
amount features
      │
      ▼
balance features
      │
      ▼
feature selection
      │
      ▼
feature dataset
```
Idealnya jangan membuat transformation logic baru di sini.

Step 6 __hanya orchestration__.

Contoh konsep:
```py
def build_feature_dataset(df):
    df = add_timestamp_features(df)
    df = add_amount_features(df)
    df = add_balance_features(df)

    return df.select(*FEATURE_COLUMNS)
```
Dengan demikian tanggung jawab tetap terpisah:
```
timestamp_features.py
        │
amount_features.py
        │
balance_features.py
        │
        ▼
feature_pipeline.py
```

---
## Step 6.3 — Decide Target Separation

Kita perlu membedakan:
```
features
target
metadata
```
Untuk fraud/risk ML:
```
X = feature columns
y = is_fraud
```
Jadi `is_fraud` __bukan feature__.

Contoh:
```
FEATURE_COLUMNS
├── transaction_type
├── transaction_hour
├── transaction_day_of_week
├── amount
├── amount_log
├── ...
└── balance features

TARGET_COLUMN
└── is_fraud
```
`transaction_id` juga sebaiknya dianggap sebagai __identifier__, bukan ML feature.

Ini penting untuk mencegah:

> data leakage / meaningless identifier learning.

---
## Step 6.4 — Feature Dataset Output

Pipeline harus menghasilkan:
```
data/
└── processed/
    ├── canonical/
    └── features/
```
Format:
> Parquet

__Kenapa Parquet?__

Karena ini merupakan intermediate ML dataset dan kita sudah menggunakan Spark.

Pipeline:
```
CSV
 ↓
PySpark
 ↓
Canonical Parquet
 ↓
Feature Engineering
 ↓
Feature Parquet
```
Tidak perlu kembali ke CSV.

---
## Step 6.5 — Feature Dataset Validation

Setelah dataset feature dibuat, validasi:

__Structural__
```
row count sama dengan canonical
```
__Schema__
```
expected columns exist
expected datatypes
```
__Null__
```
feature columns tidak unexpected null
```
__Target__
```
is_fraud ∈ {0,1}
```
__Identifier__
```
transaction_id unique
```
__Numeric sanity__
Contoh:
```
amount >= 0
amount_log finite
ratio finite
```
Khusus ratio, kita harus memperhatikan:
```
division by zero
```

---
## Step 6.6 — Unit Tests
Buat:
> tests/unit/test_feature_pipeline.py

Minimal test:

__Test 1 — semua feature muncul__
```
input
 ↓
build_feature_dataset
 ↓
expected columns
```
__Test 2 — target tetap tersedia__
```
is_fraud exists
```
__Test 3 — identifier tetap tersedia__
```
transaction_id exists
```
__Test 4 — row count preserved__
```
input rows == output rows
```
__Test 5 — feature values benar__
Gunakan small deterministic dataset.

---
## Step 6.7 — Feature Pipeline Script

Buat:
> pipelines/build_features.py

Flow:
```
create Spark session
       ↓
read canonical dataset
       ↓
validate canonical dataset
       ↓
build features
       ↓
validate feature dataset
       ↓
write Parquet
       ↓
log summary
       ↓
stop Spark
```
Command:
```pwsh
python -m pipelines.build_features
```
Expected:
```
Reading canonical dataset
Building features
Feature dataset schema
Writing feature dataset
Feature dataset successfully written
```

---
## Step 6.8 — Run Quality Gates
Sebelum Step 6 dianggap selesai:
```pwsh
pytest
make lint
make format-check
pre-commit run --all-files
```
Jika semuanya green:
> Step 6 **DONE**

---
# Step 7 — End-to-End Data Pipeline

Step 7 adalah penutup Sprint 1.

Tujuannya bukan menambahkan feature baru.

Tujuannya memastikan seluruh data pipeline dapat berjalan sebagai satu reproducible workflow.

---
## Step 7.1 — Define Pipeline Contract

Kita dokumentasikan flow final:
```
Raw PaySim
    │
    ▼
Ingestion
    │
    ▼
Schema Validation
    │
    ▼
Canonical Transformation
    │
    ▼
Canonical Validation
    │
    ▼
Domain Validation
    │
    ▼
Feature Engineering
    │
    ▼
Feature Dataset Validation
    │
    ▼
Feature Parquet
```
Ini nantinya menjadi backbone project.

---
## Step 7.2 — Create Pipeline Orchestrator
Buat:
> pipelines/run_pipeline.py

Tanggung jawabnya:
```
Step 1
   ↓
Step 2
   ↓
Step 3
   ↓
Step 4
   ↓
Step 5
   ↓
Step 6
```
Sehingga user cukup menjalankan:
```pwsh
python -m pipelines.run_pipeline
```
Tidak perlu:
```pwsh
python -m pipelines.validate_paysim
python -m pipelines.canonicalize_paysim
python -m pipelines.validate_canonical
python -m pipelines.profile_canonical
python -m pipelines.build_features
```
satu per satu.

---
## Step 7.3 — Pipeline Failure Semantics
Ini penting untuk portfolio MLE.

Pipeline harus fail fast.

Misalnya:
```
Raw validation FAILED
       │
       ▼
STOP
```
Jangan:
```
validation failed
       ↓
tetap generate features
       ↓
tetap write dataset
```
Karena itu berbahaya pada production ML pipeline.

---
## Step 7.4 — Idempotency

Pipeline harus aman ketika dijalankan ulang.

Contoh:
```pwsh
python -m pipelines.run_pipeline
```
kemudian:
```pwsh
python -m pipelines.run_pipeline
```
hasilnya harus deterministic.

Tidak boleh menghasilkan:
```
canonical/
canonical/
canonical/
canonical/
```
atau append data secara tidak sengaja.

Kita perlu menentukan strategy:

> overwrite

untuk local development.

Misalnya:
```py
.write.mode("overwrite")
```
Nanti ketika masuk production architecture, strategy ini bisa berubah menjadi:
```
partitioned writes
incremental processing
data versioning
```
tetapi belum perlu sekarang.

---
## Step 7.5 — Pipeline Logging

Kita ingin log yang cukup informatif.

Contoh:
```
[1/6] Reading raw dataset
[2/6] Validating raw dataset
[3/6] Building canonical dataset
[4/6] Validating canonical dataset
[5/6] Building features
[6/6] Writing feature dataset

Pipeline completed successfully
```

Tambahkan summary:
```
Input rows       : 6,362,620
Canonical rows   : 6,362,620
Feature rows     : 6,362,620
Feature columns  : XX
Target column    : is_fraud
Output           : data/processed/features
```
Ini bagus banget untuk portfolio karena menunjukkan bahwa kita memikirkan **observability**, walaupun masih local pipeline.

---
## Step 7.6 — End-to-End Test
Buat:
> tests/integration/test_pipeline.py

Tetapi __jangan menggunakan 6.3 juta rows__.

Gunakan tiny fixture:
> 5–20 transactions

Flow:
```
fixture
 ↓
ingestion
 ↓
canonicalization
 ↓
validation
 ↓
feature engineering
 ↓
output
```
Test:
```
pipeline completes
output exists
row count preserved
schema correct
target exists
```
Ini berbeda dengan unit test.

__Unit test__
```
function → function
```

__Integration test__
```
pipeline component A
        ↓
component B
        ↓
component C
```

---
## Step 7.7 — Makefile Integration
Tambahkan command:
```make
pipeline:
    python -m pipelines.run_pipeline
```
Sehingga:
```pwsh
make pipeline
```
menjalankan entire data pipeline.

Idealnya Makefile akhirnya mempunyai:
```
make test
make lint
make format-check
make pipeline
```

---
## Step 7.8 — Final Sprint 1 Quality Gate
Jalankan:
```pwsh
pytest
```
kemudian:
```pwsh
make lint
make format-check
pre-commit run --all-files
```
dan terakhir:
```pwsh
make pipeline
```
Semua harus:
> **GREEN**

---
## Step 7.9 — Sprint 1 Documentation
Update:
> README.md

menjadi mencakup:
- Architecture
- Data Flow
- Dataset
- Canonical Schema
- Domain Validation
- Feature Engineering
- Feature Dataset
- How to Run
- Testing
- Quality Gates

Tambahkan diagram:
```
                    ┌─────────────────┐
                    │  PaySim CSV     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ PySpark Ingest  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Schema Validate │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Canonicalize    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Domain Validate │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Feature Engineer│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Feature Validate│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Feature Parquet │
                    └─────────────────┘
```

---
# Final Definition of Done — Sprint 1

Kalau kita mengikuti plan ini, Sprint 1 baru benar-benar selesai ketika:

## __Data Engineering__
- [ ✓ ] Raw dataset ingestion menggunakan PySpark
- [ ✓ ] Raw schema validation
- [ ✓ ] Canonical transaction schema
- [ ✓ ] Canonical transformation
- [ ✓ ] Canonical dataset Parquet
- [ ✓ ] Canonical validation
- [ ✓ ] Domain validation
- [ ✓ ] Domain profiling

## __Feature Engineering__
- [ ✓ ] Timestamp features
- [ ✓ ] Amount features
- [ ✓ ] Balance features
- [ ] Feature dataset assembly
- [ ] Feature dataset validation
- [ ] Feature Parquet

## Pipeline Engineering
- [ ] End-to-end orchestrator
- [ ] Fail-fast behavior
- [ ] Idempotent execution
- [ ] Pipeline logging
- [ ] Integration test
- [ ] Makefile pipeline command

## Quality
- [ ✓ ] pytest
- [ ✓ ] Ruff
- [ ✓ ] formatting
- [ ✓ ] pre-commit
- [ ✓ ] GitHub Actions CI
- [ ] final end-to-end pipeline green
