# SPRINT 0 — PROJECT FOUNDATION
- [✓] Create GitHub repository
- [✓] Initialize Python 3.12 environment
- [✓] Create pyproject.toml
- [✓] Create project directory structure
- [✓] Create FastAPI application
- [✓] Create /health endpoint
- [✓] Create configuration layer
- [✓] Create logging layer
- [✓] Configure Ruff
- [✓] Configure Pytest
- [✓] Write first unit test
- [✓] Create Makefile
- [✓] Configure pre-commit
- [✓] Create .gitignore
- [✓] Create .env.example
- [✓] Create GitHub Actions CI
- [✓] Write initial README
- [✓] Verify local setup
- [✓] Verify CI
- [✓] Commit & push


# SPRINT 1 — Transaction Data Engineering Design

## Step 1 — Dependency + Dataset Setup
- [✓] trm-env aktif
- [✓] PySpark installed
- [✓] Pandas installed
- [✓] PyArrow installed
- [✓] Pandera installed
- [✓] PyYAML installed
- [✓] pip check bersih
- [✓] data/raw/ tersedia
- [✓] PaySim downloaded
- [✓] paysim.csv tersedia
- [✓] raw dataset tidak masuk Git
- [✓] configs/data.yaml dibuat
- [✓] data/README.md dibuat
- [✓] .gitignore diperbarui
- [✓] pre-commit passed
- [✓] pytest passed
- [✓] ruff passed
- [✓] format check passed

## Step 2 — PySpark Ingestion + Schema Validation
- [✓] ml/data/schema.py
- [✓] ml/data/spark.py
- [✓] ml/data/ingestion.py
- [✓] ml/data/validation.py
- [✓] pipelines/validate_paysim.py
- [✓] Explicit Spark schema
- [✓] PySpark berhasil membaca PaySim
- [✓] Schema berhasil diverifikasi
- [✓] Null validation
- [✓] Amount validation
- [✓] Fraud label validation
- [✓] Step validation
- [✓] ValidationResult
- [✓] Unit test valid dataset
- [✓] Unit test invalid dataset
- [✓] pytest PASS
- [✓] ruff PASS
- [✓] format check PASS
- [✓] pre-commit PASS

## Step 3: Canonical Transaction Transformation.
- [✓] ml/data/transformation.py
- [✓] Canonical schema defined
- [✓] PaySim → canonical mapping
- [✓] Deterministic transaction_id
- [✓] step → timestamp
- [✓] YAML config digunakan
- [✓] Canonical data ditulis sebagai Parquet
- [✓] Transformation unit tests
- [✓] make lint
- [✓] make test
- [✓] Pipeline berhasil dengan full PaySim dataset
- [✓] data/processed/ tidak masuk Git

## Step 4: Data Quality Validation & Canonical Dataset Verification
- [ ] canonical Parquet dapat dibaca kembali
- [ ] schema valid
- [ ] row count = 6,362,620
- [ ] required fields tidak NULL
- [ ] transaction_id valid dan unique
- [ ] timestamp valid
- [ ] amount valid
- [ ] fraud labels valid
- [ ] fraud distribution berhasil dihitung
- [ ] automated tests pass
- [ ] Ruff pass
- [ ] pipeline validation berhasil
