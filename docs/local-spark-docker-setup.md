Local Spark Environment Strategy
Overview

Project Transaction Risk Scoring — End-to-End ML Platform menggunakan PySpark untuk data ingestion, validation, transformation, dan feature engineering.

Development environment saat ini menggunakan:

Component	Current
OS	Windows
Python	3.12.14
PySpark	4.2.0
Hadoop	3.5.0
Java	17.0.20
Container runtime	Not available on office laptop
Spark mode	Local

Target environment untuk laptop pribadi akan menggunakan Linux container agar Spark tidak bergantung pada Windows-specific Hadoop utilities seperti winutils.exe.

Apache Spark 4.2.0 adalah release resmi Spark dan dokumentasi versi tersebut tersedia dari Apache Spark.

1. Current Development Environment

Saat ini development dilakukan pada Windows laptop kantor.

Windows
   │
   ├── Conda
   │      └── trm-env
   │
   ├── Python 3.12
   │
   ├── PySpark 4.2.0
   │
   ├── Hadoop 3.5.0
   │
   └── Java 17

Spark dijalankan menggunakan:

SparkSession.builder \
    .appName("TransactionRiskML-DataPipeline") \
    .master("local[*]") \
    .getOrCreate()
Windows Hadoop compatibility

Native Spark execution pada Windows membutuhkan Hadoop filesystem support untuk operasi tertentu.

Tanpa konfigurasi Hadoop Windows, Spark menghasilkan:

Did not find winutils.exe
HADOOP_HOME and hadoop.home.dir are unset

Untuk sementara, development environment menggunakan:

winutils.exe
Hadoop 3.3.6

sementara Spark sebenarnya menggunakan:

Hadoop 3.5.0

Konfigurasi tersebut berhasil melewati filesystem initialization dan Spark sudah mulai melakukan Parquet write.

Namun, winutils 3.3.6 tidak sebaiknya dianggap sebagai final production-like environment untuk project ini karena versi Hadoop runtime berbeda.

2. Current Problem: JVM Heap Memory

Setelah winutils.exe tersedia, canonicalization pipeline sudah berhasil mencapai tahap Parquet writing.

Contoh warning:

MemoryManager:
Total allocation exceeds 95.00%
(1,020,054,720 bytes) of heap memory

Scaling row group sizes to 95.00% for 8 writers

Scaling row group sizes to 84.44% for 9 writers

Scaling row group sizes to 76.00% for 10 writers

Scaling row group sizes to 69.09% for 11 writers

Scaling row group sizes to 63.33% for 12 writers

Ini menunjukkan Spark memiliki heap yang relatif kecil dan Parquet writer berjalan secara paralel.

Dataset yang sedang diproses:

6,362,620 rows

Dengan canonical schema:

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
3. Immediate Solution for Office Laptop

Sebelum pindah ke Docker, development pada Windows masih bisa dilanjutkan dengan mengurangi parallelism dan meningkatkan JVM heap.

3.1 Configure Spark Memory

Update ml/data/spark.py.

Current:

from pyspark.sql import SparkSession


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("TransactionRiskML-DataPipeline")
        .master("local[*]")
        .getOrCreate()
    )

Recommended development configuration:

from pyspark.sql import SparkSession


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("TransactionRiskML-DataPipeline")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
Why local[4]?

local[*] menggunakan seluruh logical CPU yang tersedia.

Untuk laptop development, ini dapat menyebabkan terlalu banyak task/writer berjalan bersamaan.

Misalnya:

local[*]
    ↓
many concurrent tasks
    ↓
many Parquet writers
    ↓
higher memory pressure

Dengan:

local[4]

kita membatasi Spark menjadi empat worker threads.

4. Memory Configuration

Recommended starting point:

spark.driver.memory = 4g

Jika laptop memiliki RAM yang cukup:

8 GB RAM
→ jangan langsung gunakan 6–7 GB untuk Spark

16 GB RAM
→ 4g Spark biasanya reasonable

32 GB RAM
→ 6g–8g dapat dipertimbangkan

Jangan langsung menggunakan:

spark.driver.memory = 16g

hanya karena laptop memiliki 16 GB RAM.

Operating system, browser, IDE, Python, JVM, dan proses lainnya tetap membutuhkan memory.

5. Reduce Parquet Write Parallelism

Untuk dataset PaySim 6.3 juta rows, kita tidak perlu membuat terlalu banyak writer untuk local development.

Sebelum write:

canonical_df = canonical_df.coalesce(4)

kemudian:

canonical_df.write.mode("overwrite").parquet(
    f"{output_path}/canonical"
)

Sehingga:

canonical_df = canonical_df.coalesce(4)

canonical_df.write \
    .mode("overwrite") \
    .parquet(f"{output_path}/canonical")
Why coalesce()?

Tujuannya bukan meningkatkan distributed processing performance.

Tujuannya adalah:

mengurangi jumlah partition/output writer pada local development sehingga memory pressure lebih rendah.

Untuk production/distributed Spark, partition strategy harus dievaluasi berdasarkan workload. Jangan menganggap coalesce(4) sebagai konfigurasi universal.

6. Recommended Local Configuration

Untuk sekarang, gunakan:

def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("TransactionRiskML-DataPipeline")
        .master("local[4]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )

dan:

canonical_df = canonical_df.coalesce(4)

canonical_df.write \
    .mode("overwrite") \
    .parquet(f"{output_path}/canonical")

Kemudian jalankan:

python -m pipelines.canonicalize_paysim
7. Validate Output

Jika pipeline berhasil, periksa:

data/
└── processed/
    └── canonical/
        ├── part-*.parquet
        ├── _SUCCESS
        └── ...

Kemudian test kembali menggunakan PySpark:

df = spark.read.parquet("data/processed/canonical")

print(df.count())
df.printSchema()
df.show(5, truncate=False)

Expected row count:

6,362,620

Expected schema:

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
8. Important Data Leakage Consideration

The canonical layer intentionally retains:

origin_balance_before
origin_balance_after
destination_balance_before
destination_balance_after

because these fields are useful for:

data quality
transaction analysis
reconciliation
feature engineering experiments

However, they should not automatically become model features.

PaySim's fraud-generation mechanism means these balance fields can introduce target leakage / unrealistic predictive behavior for the fraud model.

Therefore:

Canonical Layer
       │
       ├── balance fields
       │       └── retained
       │
       └── ML Feature Layer
               └── explicitly excluded

This separation should remain in the project architecture.

9. Future Environment: Linux Container

When development is moved to the personal laptop, the recommended environment becomes:

Windows Host
│
└── Docker / Podman
      │
      └── Linux Container
            │
            ├── Python 3.12
            ├── PySpark 4.2.0
            ├── Hadoop libraries
            └── Java 17

The important difference is:

CURRENT

Windows
   ↓
PySpark
   ↓
Hadoop Windows filesystem
   ↓
winutils.exe

versus:

TARGET

Windows
   ↓
Docker
   ↓
Linux
   ↓
PySpark
   ↓
Hadoop
   ↓
Linux filesystem

The second architecture eliminates the Windows-specific winutils.exe dependency.

10. Target Docker Architecture

The final local development environment should look approximately like this:

                    Windows Laptop
                          │
                     Docker Engine
                          │
             ┌────────────┴────────────┐
             │                         │
       Spark / ML Container       Supporting Services
             │                         │
             │                    PostgreSQL
             │                    MLflow
             │
             ├── PySpark
             ├── Python
             ├── Java
             └── Hadoop libraries
                          │
                          ↓
                 Mounted Project Volume
                          │
             ┌────────────┴────────────┐
             │                         │
         data/raw                 data/processed

Initially, we do not need a Spark cluster.

The first Docker implementation can still use:

Spark local mode

inside a Linux container.

That keeps the development environment simple.

11. Docker Development Phases

Dockerization should be introduced gradually.

Phase 1 — Spark Container

Goal:

docker run
    ↓
Python
    ↓
PySpark
    ↓
Parquet

No MLflow yet.

Phase 2 — Data Pipeline

Run:

canonicalize_paysim.py

inside the container.

Expected:

PaySim CSV
    ↓
PySpark
    ↓
Validation
    ↓
Canonical Transformation
    ↓
Parquet
Phase 3 — MLflow

Add:

MLflow

for:

experiment tracking
parameters
metrics
artifacts
model registry
Phase 4 — FastAPI

Add:

FastAPI

for model inference.

Architecture:

Client
  ↓
FastAPI
  ↓
Model
  ↓
Prediction
Phase 5 — Kubernetes

After Docker works correctly:

Docker
   ↓
Kubernetes
   ↓
FastAPI
   ↓
ML model

For local Kubernetes:

kind

or:

minikube

can be used.

12. Why We Should Not Dockerize Everything Now

The office laptop has a restriction:

Virtualization / Containerization
        ↓
      BLOCKED

Therefore the current development strategy is deliberately split.

Office laptop
Windows
├── Conda
├── Python
├── PySpark
├── Java
└── temporary winutils setup

Focus on:

data pipeline
schema
validation
transformation
feature engineering
model development
testing
Personal laptop
Windows
└── Docker
      └── Linux
           ├── PySpark
           ├── MLflow
           ├── FastAPI
           ├── Prometheus
           ├── Grafana
           └── Evidently

Focus on:

reproducibility
containerization
service integration
monitoring
Kubernetes
production-like deployment
13. Environment Reproducibility

The project should eventually document two supported environments:

Development Environment
├── Windows native
│   └── intended for restricted corporate laptops
│
└── Linux container
    └── recommended / reproducible environment

The Linux container becomes the canonical environment for the portfolio project.

Windows native remains a convenience environment.

14. Recommended Repository Structure

Eventually:

transaction-risk-ml/
│
├── app/
│   └── ...
│
├── ml/
│   ├── data/
│   ├── features/
│   ├── training/
│   └── monitoring/
│
├── pipelines/
│   ├── validate_paysim.py
│   ├── canonicalize_paysim.py
│   └── ...
│
├── configs/
│   ├── data.yaml
│   └── ...
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── tests/
│
├── docker/
│   ├── spark/
│   │   └── Dockerfile
│   └── ...
│
├── docker-compose.yml
│
├── k8s/
│   ├── fastapi/
│   ├── mlflow/
│   ├── prometheus/
│   └── grafana/
│
├── .github/
│   └── workflows/
│
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
15. Environment Strategy Summary
Environment	Purpose	Spark	Container
Office laptop	Daily development	PySpark 4.2.0 native Windows	No
Personal laptop	Reproducible development	PySpark 4.2.0 Linux	Docker/Podman
Local Kubernetes	Production-like deployment	Containerized	Yes
CI	Automated validation	Containerized	Yes
16. Recommended Next Step

For the current office laptop, do not change the architecture yet.

First fix the memory issue with:

.master("local[4]")
.config("spark.driver.memory", "4g")
.config("spark.sql.shuffle.partitions", "8")

and:

canonical_df = canonical_df.coalesce(4)

Then rerun:

python -m pipelines.canonicalize_paysim

The goal is simply:

6,362,620 rows
        ↓
canonical transformation
        ↓
Parquet
        ↓
SUCCESS

Once this succeeds, Sprint 1 / canonicalization can be considered technically complete, and we can move on to the next data-engineering step rather than spending more time on Windows Hadoop.

For the personal laptop, Dockerization should be introduced after the native pipeline is stable, so we can verify that any future Docker issue is an environment issue rather than a pipeline issue.

Official references
Apache Spark 4.2.0 Release
Apache Spark 4.2.0 Documentation

The Spark 4.2.0 release and documentation are published by Apache Spark.

Satu catatan penting: dari log yang kamu kirim, saya belum akan menyimpulkan bahwa proses pasti akan OutOfMemoryError setelah warning tersebut. Warning itu menunjukkan memory pressure; error Java heap space baru bisa dipastikan kalau bagian akhir stack trace-nya memang menunjukkan OutOfMemoryError: Java heap space. Tetapi konfigurasi local[4] + 4g + coalesce(4) adalah langkah yang masuk akal untuk mengurangi tekanan memory pada laptop development.
