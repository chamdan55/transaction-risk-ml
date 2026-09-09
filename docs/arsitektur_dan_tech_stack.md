Nama kerja project:

Transaction Risk Scoring — End-to-End ML Platform

Tujuan akhirnya:

Build, deploy, monitor, and continuously improve a production-ready machine learning system for transaction risk scoring.

Fokus utamanya adalah membuktikan end-to-end ML lifecycle tanpa cloud.

1. High-Level Architecture

Arsitektur final yang saya rekomendasikan:

                           ┌──────────────────────┐
                           │       GitHub         │
                           │  Source + CI/CD      │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │   GitHub Actions     │
                           │ Test / Lint / Build  │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │    Docker Images     │
                           └──────────┬───────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │       Local Kubernetes          │
                    │       kind / Minikube           │
                    │                                 │
                    │   ┌─────────────────────────┐   │
                    │   │    FastAPI Inference    │   │
                    │   │                         │   │
                    │   │  /predict               │   │
                    │   │  /health                │   │
                    │   │  /model/info            │   │
                    │   └────────────┬────────────┘   │
                    │                │                │
                    │                ▼                │
                    │   ┌─────────────────────────┐   │
                    │   │     ML Model            │   │
                    │   │     XGBoost             │   │
                    │   └─────────────────────────┘   │
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
          ┌───────────────────┐             ┌───────────────────┐
          │    Prometheus     │             │    Prediction     │
          │ System Metrics    │             │       Logs        │
          └─────────┬─────────┘             └─────────┬─────────┘
                    │                                 │
                    ▼                                 ▼
          ┌───────────────────┐             ┌───────────────────┐
          │      Grafana      │             │     Evidently     │
          │    Dashboards     │             │ Drift / Quality   │
          └───────────────────┘             └─────────┬─────────┘
                                                      │
                                                      ▼
                                            ┌───────────────────┐
                                            │ Retraining Trigger│
                                            └─────────┬─────────┘
                                                      │
                                                      ▼
                                            ┌───────────────────┐
                                            │ Training Pipeline │
                                            └─────────┬─────────┘
                                                      │
                                                      ▼
                                            ┌───────────────────┐
                                            │       MLflow      │
                                            │ Experiment +      │
                                            │ Model Registry    │
                                            └───────────────────┘

Ada empat subsystem utama:

ML Pipeline
Model Serving
ML Monitoring
MLOps Automation
2. Final Tech Stack

Saya sarankan kita tidak menambahkan teknologi hanya supaya terlihat banyak.

Final stack:

Layer	Technology	Purpose
Language	Python 3.12	Core language
Data processing	Pandas	Data preparation
ML	scikit-learn	Baseline + preprocessing
ML	XGBoost	Main production model
Data validation	Pandera	Dataset/schema validation
Experiment tracking	MLflow	Experiments + metrics
Model registry	MLflow	Model versioning
API	FastAPI	Model serving
API schema	Pydantic	Request/response validation
Testing	Pytest	Unit/integration/model tests
Code quality	Ruff	Linting + formatting
Container	Docker	Packaging
Orchestration	Kubernetes	Deployment
Local Kubernetes	kind	Local cluster
System monitoring	Prometheus	Metrics
Dashboard	Grafana	Visualization
ML monitoring	Evidently	Drift/data/model monitoring
CI/CD	GitHub Actions	Automation
Version control	Git + GitHub	Source control
Storage	PostgreSQL	Metadata / application data
Artifact storage	Local filesystem / Docker volume	Model/artifacts
Yang sengaja tidak kita gunakan

Untuk menjaga scope:

AWS
Azure
GCP
Kafka
Spark
Airflow
Kubeflow
Terraform
MLflow alternatives
LangChain
LLM

Bukan karena teknologi tersebut tidak bagus, tetapi karena mereka tidak diperlukan untuk membuktikan objective project.

3. Kenapa XGBoost?

Saya pilih XGBoost sebagai production model, dengan scikit-learn sebagai baseline.

Strukturnya:

Baseline
   │
   ├── Logistic Regression
   │
   └── Random Forest
          │
          ▼
     XGBoost
          │
          ▼
   Production Candidate

Ini memberikan cerita yang bagus:

Baseline → experimentation → model comparison → production candidate.

Dan XGBoost sangat masuk akal untuk dataset tabular seperti transaction risk.

4. Data Layer

Kita gunakan dataset publik atau synthetic transaction dataset.

Saya lebih suka pendekatan:

Training data
data/
├── raw/
├── processed/
└── reference/

Pipeline:

Raw CSV
   │
   ▼
Schema Validation
   │
   ▼
Data Cleaning
   │
   ▼
Feature Engineering
   │
   ▼
Processed Dataset
   │
   ├── train
   ├── validation
   └── test
Contoh feature
transaction_amount
transaction_hour
transaction_day
merchant_category
transaction_frequency
location_distance
account_age
previous_transaction_amount
device_change
international_transaction

Target:

is_risky
5. Data Validation

Kita akan menggunakan Pandera.

Misalnya:

transaction_amount
→ numeric
→ >= 0

transaction_hour
→ integer
→ 0–23

merchant_category
→ string
→ not null

is_risky
→ 0 / 1

Ini penting karena requirement vacancy menyebut:

data governance, security, and best engineering practices

Data validation adalah salah satu bentuk engineering discipline yang konkret.

6. ML Training Architecture

Training pipeline:

                 ┌─────────────┐
                 │ Raw Dataset │
                 └──────┬──────┘
                        ▼
                ┌───────────────┐
                │ Data Validate │
                └──────┬────────┘
                       ▼
                ┌───────────────┐
                │ Preprocessing │
                └──────┬────────┘
                       ▼
                ┌───────────────┐
                │Feature Engineer│
                └──────┬────────┘
                       ▼
              ┌──────────────────┐
              │ Train / Val / Test│
              └─────────┬────────┘
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
       Logistic      Random       XGBoost
      Regression     Forest
           │            │            │
           └────────────┼────────────┘
                        ▼
                 Model Evaluation
                        │
                        ▼
                  MLflow Tracking
                        │
                        ▼
                  Model Selection
                        │
                        ▼
                 Model Registry
7. MLflow Architecture

MLflow menjadi pusat experiment tracking.

Misalnya:

Experiment:
transaction-risk-classification

Run 001
├── algorithm = logistic_regression
├── learning_rate = -
├── max_depth = -
├── F1 = 0.72
└── ROC-AUC = 0.81

Run 002
├── algorithm = random_forest
├── n_estimators = 300
├── F1 = 0.81
└── ROC-AUC = 0.89

Run 003
├── algorithm = xgboost
├── max_depth = 6
├── learning_rate = 0.05
├── F1 = 0.88
└── ROC-AUC = 0.94

Kemudian:

MLflow Model Registry

transaction-risk-model

v1 → Production
v2 → Staging
v3 → Candidate

Ini akan menjadi salah satu bukti utama bahwa kamu memahami model lifecycle.

8. Model Serving

Model production disajikan melalui FastAPI.

                    Client
                      │
                      │ POST /predict
                      ▼
              ┌───────────────┐
              │    FastAPI    │
              └───────┬───────┘
                      │
                      ▼
              Pydantic Validation
                      │
                      ▼
               Feature Pipeline
                      │
                      ▼
                ML Predictor
                      │
                      ▼
                 XGBoost
                      │
                      ▼
                  Prediction

Response:

{
  "prediction": 1,
  "risk_score": 0.91,
  "risk_level": "HIGH",
  "model_name": "transaction-risk-model",
  "model_version": "3"
}
9. API Design

Minimal endpoint:

POST /predict

Operational endpoints:

GET /health
GET /ready
GET /model/info
GET /metrics

Contoh:

GET /model/info

{
  "model_name": "transaction-risk-model",
  "version": "3",
  "algorithm": "xgboost",
  "trained_at": "2026-09-08",
  "status": "production"
}

Ini membuat model serving terasa seperti production backend service.

Dan ini sangat sesuai dengan pengalamanmu di FastAPI.

10. Kubernetes Architecture

Di Kubernetes kita tidak perlu membuat cluster yang kompleks.

Kubernetes Cluster
│
├── namespace: ml-platform
│
├── inference-api
│   ├── Deployment
│   ├── Service
│   └── ConfigMap
│
├── monitoring
│   ├── Prometheus
│   └── Grafana
│
└── mlflow
    ├── Deployment
    └── Service

Untuk local development:

kind

Jadi:

Docker image
       ↓
kind cluster
       ↓
Kubernetes Deployment
       ↓
FastAPI Pod

Ini cukup untuk membuktikan orchestration skill.

11. Monitoring Architecture

Kita sengaja pisahkan:

System monitoring
FastAPI
   │
   ▼
Prometheus
   │
   ▼
Grafana

Metrics:

request_count
request_latency
error_count
prediction_count
HTTP status

Dashboard:

┌────────────────────────────────────┐
│ Transaction Risk API               │
├────────────────────────────────────┤
│ Requests              128,421      │
│ Error Rate              0.21%      │
│ P95 Latency             74 ms      │
│ Requests/min             183       │
├────────────────────────────────────┤
│ Prediction Distribution            │
│ LOW        ███████████  71%        │
│ MEDIUM     ████          19%        │
│ HIGH       ██            10%       │
└────────────────────────────────────┘
12. ML Monitoring

System monitoring ≠ ML monitoring.

Kita buat pipeline terpisah:

Prediction Logs
      │
      ▼
Evidently
      │
      ├── Data Drift
      ├── Prediction Drift
      ├── Data Quality
      └── Model Performance

Contohnya:

Feature                  Drift
────────────────────────────────
transaction_amount       42%
transaction_hour          3%
merchant_category        18%
location_distance        27%

Overall Drift: DETECTED
13. Retraining Architecture

Ini bagian yang akan membuat project jauh lebih kuat.

               Production Data
                      │
                      ▼
               Monitoring Job
                      │
              ┌───────┴───────┐
              │               │
         Data Drift      Performance
              │               │
              └───────┬───────┘
                      ▼
                Threshold?
                 /       \
               NO         YES
               │           │
               ▼           ▼
             Stop      Retraining
                           │
                           ▼
                     New Model
                           │
                           ▼
                       Evaluate
                           │
                  ┌────────┴────────┐
                  │                 │
              Better?            Worse?
                  │                 │
                  ▼                 ▼
             Candidate            Reject
                  │
                  ▼
             Model Registry
                  │
                  ▼
              Production

Dengan demikian kita punya:

training → deployment → monitoring → retraining → redeployment

Ini benar-benar mencakup lifecycle yang diminta vacancy.

14. CI/CD Architecture

GitHub Actions:

Developer
    │
    ▼
git push
    │
    ▼
GitHub
    │
    ▼
GitHub Actions
    │
    ├── Ruff
    ├── Pytest
    ├── Model tests
    ├── Integration tests
    │
    ▼
Docker Build
    │
    ▼
Container Test
    │
    ▼
Kubernetes Deployment

Untuk tahap awal kita bisa melakukan deployment ke local kind cluster dari CI secara terbatas atau menggunakan CI untuk build/test saja dan deployment secara lokal.

Saya lebih menyarankan CI terlebih dahulu, CD kemudian, supaya tidak over-engineer.

15. Testing Strategy

Ini juga jangan dilewatkan.

Unit test
test_feature_engineering()
test_data_validation()
test_prediction()
Integration test
API
 ↓
Preprocessing
 ↓
Model
 ↓
Response
Model test

Contohnya:

model F1 >= baseline
recall >= minimum threshold
prediction schema valid
API test
POST /predict
→ 200

Invalid request
→ 422

Model unavailable
→ 503

Jadi ML project-nya punya software engineering discipline.

16. Repository Architecture Final

Saya akan sedikit memperbaiki struktur repository sebelumnya supaya separation of concerns lebih jelas:

transaction-risk-ml/
│
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py
│   │   │   ├── prediction.py
│   │   │   └── model.py
│   │   └── schemas.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   │
│   ├── inference/
│   │   ├── predictor.py
│   │   └── model_loader.py
│   │
│   └── main.py
│
├── ml/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   └── engineering.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── experiment.py
│   │
│   └── monitoring/
│       ├── drift.py
│       └── performance.py
│
├── pipelines/
│   ├── training_pipeline.py
│   ├── monitoring_pipeline.py
│   └── retraining_pipeline.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── model/
│
├── deployment/
│   ├── docker/
│   │   └── Dockerfile
│   │
│   └── kubernetes/
│       ├── namespace.yaml
│       ├── api-deployment.yaml
│       ├── api-service.yaml
│       ├── mlflow.yaml
│       └── monitoring/
│
├── monitoring/
│   ├── prometheus/
│   └── grafana/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_experiment.ipynb
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── cd.yml
│
├── configs/
│   ├── model.yaml
│   └── monitoring.yaml
│
├── pyproject.toml
├── docker-compose.yml
├── Makefile
└── README.md

Notebook hanya untuk exploration/visualization.

Production logic tidak boleh berada di notebook.

Ini poin kecil tetapi penting untuk menunjukkan engineering maturity.

17. Development Environment

Untuk development sehari-hari:

Windows
   │
   ├── Python
   ├── Git
   ├── Docker/Podman
   │
   └── kind
          │
          ▼
    Kubernetes cluster

Local services:

localhost
│
├── FastAPI
├── MLflow
├── Prometheus
└── Grafana

Semuanya dapat dijalankan dengan:

make up

dan dihentikan dengan:

make down

Jadi project mudah direproduksi oleh recruiter/engineer lain.

18. Architecture Principles

Saya ingin kita menetapkan beberapa prinsip dari awal.

1. Reproducibility

Training yang sama → environment + data + config yang sama → hasil yang dapat direproduksi.

2. Separation of concerns
Data
≠
Training
≠
Inference
≠
Monitoring
3. Configuration-driven

Hyperparameter tidak hard-coded di banyak tempat.

Misalnya:

configs/model.yaml
4. Testability

Semua komponen penting harus bisa dites tanpa menjalankan keseluruhan platform.

5. Observability

Production model tidak boleh dianggap selesai hanya karena endpoint /predict berhasil.

6. Model promotion harus berdasarkan metric

Bukan:

model baru → otomatis production.

Tetapi:

Candidate
    ↓
Evaluation
    ↓
Quality Gate
    ↓
Promotion
19. Mapping ke Requirement Vacancy

Kalau kita selesai sampai architecture ini, mapping-nya sangat kuat:

"Design, develop, deploy ML models"
             ↓
        XGBoost + FastAPI

"Scalable ML pipelines"
             ↓
    Training / Monitoring /
       Retraining pipelines

"End-to-end ML lifecycle"
             ↓
Data → Train → Evaluate → Deploy
→ Monitor → Retrain → Deploy

"Model versioning"
             ↓
          MLflow

"Experiment tracking"
             ↓
          MLflow

"CI/CD"
             ↓
     GitHub Actions

"Production monitoring"
             ↓
 Prometheus + Grafana

"Model monitoring"
             ↓
        Evidently

"Model drift"
             ↓
        Evidently

"Automated retraining"
             ↓
    Retraining Pipeline

"APIs"
             ↓
         FastAPI

"Containers"
             ↓
          Docker

"Orchestration"
             ↓
        Kubernetes

"Software engineering"
             ↓
Pytest + Ruff + Git + CI/CD

Dan yang paling penting:

Requirement yang menjadi hero feature project ini:

Understanding of the end-to-end machine learning lifecycle, including data preparation, model development, evaluation, deployment, and monitoring

Kita bahkan akan membuat lifecycle itu terlihat jelas di README:

             END-TO-END ML LIFECYCLE

       ┌──────────────────────┐
       │ 1. Data Preparation  │
       └──────────┬───────────┘
                  ↓
       ┌──────────────────────┐
       │ 2. Model Development │
       └──────────┬───────────┘
                  ↓
       ┌──────────────────────┐
       │ 3. Model Evaluation  │
       └──────────┬───────────┘
                  ↓
       ┌──────────────────────┐
       │ 4. Model Deployment  │
       └──────────┬───────────┘
                  ↓
       ┌──────────────────────┐
       │ 5. Model Monitoring  │
       └──────────┬───────────┘
                  ↓
       ┌──────────────────────┐
       │ 6. Retraining        │
       └──────────┬───────────┘
                  │
                  └──────→ back to 2
20. Scope final yang saya sarankan

Saya akan menetapkan MVP → Production-like → Advanced supaya kita tidak terjebak over-engineering.

MVP
Dataset
Data validation
Feature engineering
Baseline models
XGBoost
Evaluation
MLflow
FastAPI
Docker
Production-like
Kubernetes
Pytest
CI
Prometheus
Grafana
Evidently
Model registry
Model versioning
Advanced
Automated drift detection
Automated retraining
Model quality gate
Automated model promotion
CD
Load testing
Failure/recovery testing

Saya menyarankan kita berhenti menambah teknologi setelah Advanced scope ini. Dengan itu project sudah cukup besar untuk menunjukkan kemampuan MLE secara serius, tetapi masih realistis untuk dikerjakan seorang diri.

Dan ada satu keputusan arsitektur yang menurut saya penting: jangan mulai dari Kubernetes/MLflow dulu. Kita mulai dari ML lifecycle yang benar, lalu setiap layer infrastructure ditambahkan untuk menyelesaikan masalah nyata. Dengan begitu, saat interview kamu bisa menjelaskan mengapa setiap komponen ada, bukan sekadar menyebut daftar tools.

Langkah berikutnya paling logis adalah membuat Project Specification v1: problem statement, dataset, data schema, feature definitions, target variable, success metrics, ML experiment plan, API contract, monitoring metrics, dan Definition of Done untuk setiap sprint.
