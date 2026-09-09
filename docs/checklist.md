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
- [ ] ml/data/schema.py
- [ ] ml/data/spark.py
- [ ] ml/data/ingestion.py
- [ ] ml/data/validation.py
- [ ] pipelines/validate_paysim.py
- [ ] Explicit Spark schema
- [ ] PySpark berhasil membaca PaySim
- [ ] Schema berhasil diverifikasi
- [ ] Null validation
- [ ] Amount validation
- [ ] Fraud label validation
- [ ] Step validation
- [ ] ValidationResult
- [ ] Unit test valid dataset
- [ ] Unit test invalid dataset
- [ ] pytest PASS
- [ ] ruff PASS
- [ ] format check PASS
- [ ] pre-commit PASS
