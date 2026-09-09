# Sprint Plan

Saya sarankan 7 sprint, bukan 6, karena Data Engineering layer sekarang cukup signifikan.

## Sprint 0 — Project Foundation
Goal
Mempersiapkan repository dan engineering foundation.

Tasks
 - Git repository
 - Project structure
Python environment
pyproject.toml
Ruff
Pytest
Pre-commit
Configuration management
Logging
Makefile
README skeleton
Output
Repository siap dikembangkan
DoD
application dapat dijalankan
linting berjalan
test suite berjalan
configuration tidak hard-coded
basic logging tersedia
Sprint 1 — Data Engineering Pipeline
Goal

Membangun ingestion dan preprocessing menggunakan PySpark.

Pipeline
Raw Transaction Data
        ↓
Schema Validation
        ↓
Data Cleaning
        ↓
Transformation
        ↓
Feature Engineering
        ↓
Parquet
Feature categories

Transaction-level

amount
hour
day_of_week
merchant_category
location
device

Behavioral

transactions_last_1h
transactions_last_24h
avg_amount_last_7d
avg_amount_last_30d
unique_merchants_last_30d
Output
data/processed/
    train/
    validation/
    test/

Format:

Parquet
DoD
pipeline dapat dijalankan dari command line
schema tervalidasi
output deterministic
output berupa Parquet
tidak ada business logic penting di notebook
unit test untuk transformation tersedia
Sprint 2 — ML Development & Evaluation
Goal

Membangun model dan menentukan production candidate.

Models
Logistic Regression
Random Forest
XGBoost
Evaluation
Precision
Recall
F1
ROC-AUC
PR-AUC
Confusion Matrix

Tambahkan business-oriented threshold analysis.

Misalnya:

Threshold 0.30
Recall = 94%
Precision = 62%

Threshold 0.50
Recall = 88%
Precision = 78%

Threshold 0.70
Recall = 71%
Precision = 91%
DoD
minimal 3 model dibandingkan
baseline ditentukan
production candidate ditentukan berdasarkan metric
experiment dapat direproduksi
Sprint 3 — Experiment Tracking & Model Registry
Goal

Mengintegrasikan MLflow.

Training
   ↓
MLflow Experiment
   ├── Parameters
   ├── Metrics
   ├── Artifacts
   └── Model
          ↓
     Model Registry
DoD

Setiap training run menyimpan:

model type
hyperparameters
dataset version
metrics
training timestamp
model artifact

Model:

Candidate
    ↓
Staging
    ↓
Production
Sprint 4 — Model Serving
Goal

Menyediakan model sebagai production-like API.

Architecture:

Client
  ↓
FastAPI
  ↓
Request Validation
  ↓
Feature Preparation
  ↓
Model
  ↓
Prediction
Endpoints
POST /predict
GET /health
GET /ready
GET /model/info
GET /metrics
DoD
API dapat melakukan prediction
invalid input ditolak
model version dapat diketahui
health/readiness endpoint tersedia
integration test tersedia
Sprint 5 — Containerization & Kubernetes
Goal

Deploy inference service ke Kubernetes.

FastAPI
   ↓
Docker
   ↓
Container Image
   ↓
kind
   ↓
Kubernetes

Deployment:

Deployment
Service
ConfigMap
Secret

Kita juga bisa menggunakan:

readinessProbe
livenessProbe
resources
replicas

untuk membuat deployment lebih realistic.

DoD
kubectl get pods

menunjukkan API running.

Kemudian:

POST /predict

berhasil melalui Kubernetes Service.

Sprint 6 — Monitoring & Model Observability

Ini menurutku sprint paling penting kedua setelah Sprint 1.

Kita pisahkan:

System monitoring
FastAPI
   ↓
Prometheus
   ↓
Grafana

Monitor:

request rate
error rate
latency
CPU
memory
prediction volume
ML monitoring
Prediction Logs
      ↓
Evidently
      ↓
Data Quality
Data Drift
Prediction Drift
Model Performance
DoD

Grafana dashboard menunjukkan:

Request rate
P95 latency
Error rate

Evidently menunjukkan:

Feature drift
Prediction drift
Data quality
Sprint 7 — Automated Retraining & Model Promotion
Goal

Menutup lifecycle ML.

Production
    ↓
Monitoring
    ↓
Drift detected
    ↓
Retraining
    ↓
Evaluation
    ↓
Quality Gate
    ↓
Model Registry
    ↓
Promotion
    ↓
Deployment
Quality Gate

Misalnya:

F1_new >= F1_current
AND
Recall_new >= 0.85
AND
Precision_new >= 0.70

Kalau gagal:

REJECT

Kalau lolos:

PROMOTE
DoD

Kita bisa menjalankan scenario:

Normal production data
        ↓
Drift introduced
        ↓
Drift detected
        ↓
Retraining triggered
        ↓
New model evaluated
        ↓
New model promoted

Ini akan menjadi demo utama project.

10. Final Architecture setelah PySpark

Jadi finalnya sekarang saya kunci seperti ini:

                         ┌────────────────────┐
                         │      GitHub        │
                         │ Source + CI/CD     │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │ GitHub Actions     │
                         │ Test / Lint / Build│
                         └─────────┬──────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────┐
│                  ML DATA PIPELINE                        │
│                                                         │
│  Raw Transaction Data                                   │
│          │                                              │
│          ▼                                              │
│      PySpark                                            │
│          │                                              │
│   ┌──────┼────────┐                                     │
│   │      │        │                                     │
│ Clean  Transform  Aggregate                             │
│   │      │        │                                     │
│   └──────┼────────┘                                     │
│          ▼                                              │
│    Feature Engineering                                  │
│          │                                              │
│          ▼                                              │
│       Parquet                                           │
└──────────┬──────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────┐
│ ML Training              │
│                          │
│ sklearn + XGBoost        │
│                          │
│ LR / RF / XGBoost        │
└────────────┬─────────────┘
             │
             ▼
       ┌────────────┐
       │   MLflow   │
       │ Experiment │
       │ + Registry │
       └─────┬──────┘
             │
             ▼
       Production Model
             │
             ▼
┌──────────────────────────────────┐
│       Kubernetes Cluster         │
│                                  │
│  ┌────────────────────────────┐  │
│  │       FastAPI              │  │
│  │       Inference            │  │
│  └─────────────┬──────────────┘  │
│                │                 │
│                ▼                 │
│             XGBoost              │
│                                  │
└────────────────┬─────────────────┘
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ Prometheus   │    │ Prediction   │
│              │    │ Logs         │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ Grafana      │    │ Evidently    │
│              │    │              │
│ API Metrics  │    │ Drift        │
│ Infra Metrics│    │ Performance  │
└──────────────┘    └──────┬───────┘
                           │
                           ▼
                    Drift Detected
                           │
                           ▼
                  Retraining Pipeline
                           │
                           ▼
                       MLflow
                           │
                           ▼
                    Model Promotion
                           │
                           └──────────→ Kubernetes
11. Yang menurutku paling menarik

Dengan PySpark, project ini sekarang punya tiga identitas sekaligus:

Data Engineering
PySpark
Parquet
Data validation
Feature engineering
Large-scale processing
Machine Learning
scikit-learn
XGBoost
Model evaluation
Experimentation
Model selection
Machine Learning Engineering
FastAPI
Docker
Kubernetes
MLflow
Prometheus
Grafana
Evidently
CI/CD
Retraining
Model promotion

Itu sangat cocok dengan positioning kamu sebagai AI & Data Engineer dengan Python Backend background, sekaligus menjadi bukti yang lebih kuat ketika melamar role MLE.

Dan saya justru suka bahwa project ini tidak bergantung pada cloud. Kita bisa fokus membuktikan fundamental MLE terlebih dahulu; cloud nantinya hanya menjadi deployment target, bukan inti kompetensinya.

Satu prinsip yang akan saya pegang untuk project ini

Kita tidak akan menambahkan komponen hanya untuk "centang requirement".

Kalau nanti ada reviewer bertanya:

"Why did you use PySpark?"

kita harus punya jawaban teknis.

"Why MLflow?"

Ada alasan.

"Why Kubernetes?"

Ada alasan.

"Why Evidently?"

Ada alasan.

"Why did the model get promoted?"

Ada measurable quality gate.

Dengan pendekatan itu, project ini akan jauh lebih bernilai saat masuk portfolio/interview daripada sekadar repository dengan 15 teknologi di README.
