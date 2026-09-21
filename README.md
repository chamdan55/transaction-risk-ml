# Transaction Risk ML

End-to-end machine learning platform for transaction risk scoring, designed to demonstrate production-oriented Machine Learning Engineering practices across the complete ML lifecycle.

> **Current status:** Sprint 0, Sprint 1, and Sprint 2 completed. The project now has a reproducible data-to-model training and evaluation flow.

---

## 1. Project Overview

**Transaction Risk ML** is a portfolio project for building a production-oriented transaction risk scoring system.

The project is intentionally designed around the end-to-end Machine Learning lifecycle:

```text
Data Ingestion
      ↓
Schema Validation
      ↓
Canonical Transformation
      ↓
Domain Validation & Profiling
      ↓
Feature Engineering
      ↓
Dataset Preparation
      ↓
Model Development
      ↓
Model Evaluation
      ↓
Model Packaging
      ↓
API Deployment
      ↓
Monitoring
```
The goal is not only to train a model, but to demonstrate how an ML system can be engineered, tested, packaged, deployed, and monitored as a production-oriented application.

---

## 2. Project Goals

This project is intended to demonstrate practical skills relevant to a Machine Learning Engineer role, particularly:
- End-to-end machine learning lifecycle
- Large-scale data processing with PySpark
- Data ingestion and schema validation
- Canonical data modeling
- Transaction domain validation
- Data quality and profiling
- Feature engineering
- Machine learning model development
- Model evaluation
- API-based model serving
- Containerization
- CI/CD
- Model and system monitoring
- Software engineering best practices

Cloud computing is intentionally excluded from the current project scope so the core ML engineering lifecycle can be developed and demonstrated locally.

---

## 3. Dataset

The current project uses the __PaySim__ synthetic mobile money transaction dataset.

The dataset contains approximately __6.36 million transactions__ and includes:
- Transaction step
- Transaction type
- Transaction amount
- Origin account
- Destination account
- Origin balances before/after transaction
- Destination balances before/after transaction
- Fraud label
- Flagged-fraud label

The raw dataset is stored locally and is intentionally excluded from Git.
```text
data/
├── raw/
└── processed/
```
See ```.gitignore``` for the project's data/artifact exclusion rules.

---

## 4. Architecture

The current architecture separates data engineering, ML logic, pipelines, tests, and application serving.
```
                  ┌─────────────────────┐
                  │     PaySim CSV      │
                  │      Raw Data       │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ PySpark Ingestion   │
                  │ + Schema Validation │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Canonical           │
                  │ Transformation      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Canonical Parquet   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Domain Validation   │
                  │ + Data Profiling    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Feature Engineering │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Training Dataset    │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Model Development   │
                  │ + Evaluation        │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Model Artifact      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ FastAPI Model API   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Monitoring          │
                  └─────────────────────┘
```

---

## 5. Technology Stack
### Core
- Python 3.12
- PySpark 4.2
- FastAPI
- Pydantic / Pydantic Settings
- Uvicorn

### Data & ML
- PySpark
- scikit-learn
- XGBoost
- Joblib
- PyTorch
- Additional ML/AI libraries will be introduced only when required by later sprints.

### Testing & Code Quality
- pytest
- Ruff
- pre-commit
- GitHub Actions

### Packaging & Runtime
- ```pyproject.toml```
- Conda for local development environment
- Docker / container runtime planned for later deployment stages

### Local Java Runtime
PySpark 4.2 currently runs against:
- Java 17.0.20
- Hadoop runtime bundled with PySpark: 3.5.0

Java/Hadoop are runtime dependencies of Spark rather than Python dependencies and are therefore not declared in ```pyproject.toml```.

---

## 6. Repository Structure

Current structure:
```text
transaction-risk-ml/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── ml/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── constants.py
│       ├── domain_rules.py
│       ├── schema.py
│       ├── spark.py
│       └── validation.py
│
├── pipelines/
│   ├── __init__.py
│   ├── canonicalize_paysim.py
│   ├── profile_canonical.py
│   └── validate_canonical.py
│
├── tests/
│   └── unit/
│       ├── test_config.py
│       ├── test_domain_rules.py
│       ├── test_health.py
│       └── test_validation.py
│
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── pyproject.toml
└── README.md
```
The structure will evolve as feature engineering, training, serving, and monitoring components are added.

Sprint 2 additions include:

```text
configs/model.yaml
ml/
├── evaluation/
└── training/
pipelines/train_models.py
tests/
├── unit/
└── integration/test_training_pipeline.py
data/sample/paysim_sample.csv
```

---

## 7. Development Progress
### Sprint 0 — Project Foundation
Status: __Completed__

Sprint 0 established the engineering foundation of the project.

Completed:
- Python project configuration
- ```pyproject.toml```
- Conda development environment
- FastAPI application skeleton
- Configuration management
- Logging configuration
- Health endpoint
- Unit testing setup
- Ruff linting
- Ruff formatting
- Makefile commands
- pre-commit hooks
- GitHub Actions CI
- Python ```.gitignore```
- Initial project documentation

Quality gates currently include:
```pwsh
make test
make lint
make format-check
pre-commit run --all-files
```

---

## 8. Sprint 1 — Data Foundation
### Step 1 — Dependency + Dataset Setup
Status: __Completed__

Established:
- ML/data-processing dependencies
- PaySim dataset
- Local raw-data convention
- Data artifact exclusion from Git

Raw data:
> data/raw/paysim.csv

---

### Step 2 — PySpark Ingestion + Schema Validation

Status: __Completed__

Implemented:
- PySpark session factory
- Explicit PaySim schema
- Raw dataset ingestion
- Schema validation
- Basic data quality validation
- Unit tests for validation
- Pipeline validation command

Validation covers:
- Required columns
- Null values
- Transaction amount
- Fraud labels
- Flagged-fraud labels
- Transaction step

The current dataset contains:
```
Rows: 6,362,620
Null rows: 0
Invalid amount: 0
Invalid fraud label: 0
Invalid flagged-fraud label: 0
Invalid step: 0
```

---

### Step 3 — Canonical Transaction Transformation
Status: __Completed__

The raw PaySim schema is transformed into a domain-oriented canonical transaction schema.

Current canonical schema:
```
transaction_id
timestamp
transaction_type
origin_account_id
destination_account_id
amount
origin_balance_before
origin_balance_after
destination_balance_before
destination_balance_after
is_fraud
is_flagged_fraud
```
The canonical dataset is stored as Parquet:

> data/processed/canonical/

The transformation establishes a stable internal representation that can be used by downstream validation, feature engineering, and model-training pipelines.

---

### Step 4 — Domain Validation & Profiling

Status: __Completed__

Step 4 introduced transaction-domain rules rather than relying only on structural schema validation.

Implemented:
- Negative balance detection
- Origin balance consistency checks
- Destination balance consistency checks
- Transaction-type-aware balance rules
- Domain rule unit tests
- Canonical validation pipeline integration
- Balance behavior profiling

#### __Domain validation result__

The current PaySim canonical dataset produces:
```
negative_balance_count: 0
origin_balance_mismatch_count: 337,903
destination_balance_mismatch_count: 3,712,723
```
These mismatches are __not automatically treated as dataset-invalid rows__.

Instead, the project profiles the behavior by transaction type so that downstream feature engineering can distinguish expected dataset behavior from genuine data-quality violations.

#### __Balance profiling__

The profiling stage identified different balance behaviors across:
- __```CASH_IN```__
- __```CASH_OUT```__
- __```DEBIT```__
- __```PAYMENT```__
- __```TRANSFER```__

This is important because transaction consistency cannot be modeled safely using one universal balance equation for every transaction type.

---

## 9. Sprint 2 — ML Development & Evaluation

Status: __Completed__

Sprint 2 implemented a reproducible model development and evaluation flow using the feature splits produced by Sprint 1.

Implemented:
- Training dataset contract for train/validation/test Parquet splits
- Feature preprocessing with numeric imputation/scaling and categorical encoding
- Class imbalance analysis and weighting
- Logistic Regression baseline
- Random Forest comparison model
- XGBoost comparison model
- Precision, recall, F1, ROC-AUC, and PR-AUC evaluation
- Confusion matrix and threshold analysis
- Business-cost threshold selection
- Validation-based production candidate selection
- Final test evaluation using the selected model and threshold
- Joblib model artifacts and JSON evaluation report
- Reproducible training configuration in `configs/model.yaml`
- Unit and integration tests for the training flow

Run the model training pipeline:

```pwsh
python -m pipelines.train_models
```

Expected outputs:

```text
artifacts/
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── xgboost.joblib
└── evaluation_report.json
```

See the complete implementation plan in [`docs/SPRINT_2_ML_DEVELOPMENT_AND_EVALUATION_PLAN.md`](docs/SPRINT_2_ML_DEVELOPMENT_AND_EVALUATION_PLAN.md).

---

## 10. Current Data Quality Philosophy

The project intentionally separates three concepts:

### __Structural validity__

Does the dataset conform to the expected schema?

Examples:
- Required columns
- Data types
- Nullability
- Labels
- Transaction IDs

### __Domain validity__

Does a transaction satisfy expected business/domain constraints?

Examples:
- No negative balances
- Origin balance consistency
- Destination balance consistency

### __Statistical / behavioral profiling__

What does the dataset actually look like?

Examples:
- Balance mismatch rates by transaction type
- Transaction distributions
- Fraud distribution
- Amount distribution
- Account behavior

This separation prevents the pipeline from incorrectly treating every observed business-rule mismatch as a hard data-quality failure.

---

## 11. Current Pipeline Commands

Activate the Conda environment:
~~~pwsh
conda activate trm-env
~~~

Run raw PaySim validation:
```pwsh
python -m pipelines.validate_paysim
```

Run canonical transformation:
```pwsh
python -m pipelines.canonicalize_paysim
```
Run canonical validation and domain validation:
```pwsh
python -m pipelines.validate_canonical
```
Run canonical profiling:
```pwsh
python -m pipelines.profile_canonical
```
Run model training and evaluation:
```pwsh
python -m pipelines.train_models
```
Run unit tests:
```pwsh
pytest
```
Run linting:
```pwsh
make lint
```
Run formatting check:
```pwsh
make format-check
```
Run all pre-commit checks:
```pwsh
make pre-commit
```

---

## 12. Engineering Quality Gates

Before considering a change complete, the project aims to keep the following green:
```
pytest
ruff check
ruff format --check
pre-commit
GitHub Actions CI
```
For data-processing changes, the relevant pipeline should also execute successfully.

---

## 13. Known Local Development Considerations

The current development environment is Windows + Conda.

PySpark requires a compatible Java runtime. The current environment uses:
```
Python 3.12.14
PySpark 4.2.0
Java 17.0.20
Hadoop runtime 3.5.0
```
Windows-specific Hadoop tooling such as __`winutils.exe`__ may be required for certain local filesystem operations.

The project does not treat the Windows Hadoop setup as the target production architecture.

A containerized Spark execution environment is planned for development/production parity on a machine that supports containerization.

---

## 14. Planned Roadmap

The next stages will continue the ML lifecycle.
```
Sprint 0
Project Foundation
        ✓
        │
        ▼
Sprint 1
Data Foundation
        ✓ Step 1
        ✓ Step 2
        ✓ Step 3
        ✓ Step 4
        │
        ▼
Sprint 2
ML Development & Evaluation
        ✓ Completed
        │
        ▼
Sprint 3
Experiment Tracking + Model Registry
        ✓ Completed
        │
        ▼
Model Development
        │
        ▼
Model Evaluation
        │
        ▼
Model Packaging
        │
        ▼
Inference API
        │
        ▼
Containerized Deployment
        │
        ▼
Monitoring
        │
        ▼
End-to-End ML System
```
The exact implementation will be introduced incrementally through subsequent sprint specifications rather than building the entire system upfront.

---

## 15. Project Principles
### 1. Production-oriented, not notebook-oriented

The project prioritizes reusable modules and executable pipelines over one-off notebook experimentation.

### 2. Data quality before modeling

Model development should not begin before the input data contract and domain behavior are understood.

### 3. Explicit contracts

Schemas, validation rules, configurations, and model interfaces should be explicit and testable.

### 4. Testable ML components

Data transformations, validation rules, feature engineering, and model interfaces should have automated tests where practical.

### 5. Reproducibility

Pipeline behavior and model development should be reproducible from version-controlled code and configuration.

### 6. Separation of concerns

Data processing, domain logic, ML logic, API serving, and infrastructure concerns should remain separated.

### 7. Observability

The final system should expose enough information to understand both model behavior and system behavior after deployment.

---

## 16. Target Outcome

The final project should demonstrate the ability to take a transaction-risk ML use case from raw data to a production-oriented ML service:
```
Raw Transactions
       ↓
Validated Data
       ↓
Canonical Data
       ↓
Domain-Aware Features
       ↓
Training Dataset
       ↓
Trained Model
       ↓
Evaluated Model
       ↓
Model Artifact
       ↓
Inference API
       ↓
Containerized Service
       ↓
Monitoring
```
The primary objective is to demonstrate Machine Learning Engineering capability across the complete lifecycle, rather than focusing solely on model accuracy.
