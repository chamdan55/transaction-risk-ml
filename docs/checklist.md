# SPRINT 0 — PROJECT FOUNDATION
- [x] Create GitHub repository
- [x] Initialize Python 3.12 environment
- [x] Create pyproject.toml
- [x] Create project directory structure
- [x] Create FastAPI application
- [x] Create /health endpoint
- [x] Create configuration layer
- [x] Create logging layer
- [x] Configure Ruff
- [x] Configure Pytest
- [x] Write first unit test
- [x] Create Makefile
- [x] Configure pre-commit
- [x] Create .gitignore
- [x] Create .env.example
- [x] Create GitHub Actions CI
- [x] Write initial README
- [x] Verify local setup
- [x] Verify CI
- [x] Commit & push


# SPRINT 1 — Transaction Data Engineering Design

## Step 1 — Dependency + Dataset Setup
- [x] trm-env aktif
- [x] PySpark installed
- [x] Pandas installed
- [x] PyArrow installed
- [x] Pandera installed
- [x] PyYAML installed
- [x] pip check bersih
- [x] data/raw/ tersedia
- [x] PaySim downloaded
- [x] paysim.csv tersedia
- [x] raw dataset tidak masuk Git
- [x] configs/data.yaml dibuat
- [x] data/README.md dibuat
- [x] .gitignore diperbarui
- [x] pre-commit passed
- [x] pytest passed
- [x] ruff passed
- [x] format check passed

## Step 2 — PySpark Ingestion + Schema Validation
- [x] ml/data/schema.py
- [x] ml/data/spark.py
- [x] ml/data/ingestion.py
- [x] ml/data/validation.py
- [x] pipelines/validate_paysim.py
- [x] Explicit Spark schema
- [x] PySpark berhasil membaca PaySim
- [x] Schema berhasil diverifikasi
- [x] Null validation
- [x] Amount validation
- [x] Fraud label validation
- [x] Step validation
- [x] ValidationResult
- [x] Unit test valid dataset
- [x] Unit test invalid dataset
- [x] pytest PASS
- [x] ruff PASS
- [x] format check PASS
- [x] pre-commit PASS
