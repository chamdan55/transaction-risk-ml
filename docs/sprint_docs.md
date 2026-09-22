# Transaction Risk ML — Revised Sprint Plan

## 1. Planning Rules

Dokumen ini adalah landasan urutan implementasi. Arsitektur normatif berada di
`docs/arsitektur_dan_tech_stack.md`; unit kerja executable berada di `docs/tickets/`.

Aturan status:

- **Completed**: seluruh Definition of Done dan quality gate memiliki bukti aktual.
- **Conditionally completed**: capability utama ada, tetapi memiliki hardening item yang wajib
  diselesaikan sebelum production-like use.
- **Reopened**: pernah dianggap selesai, tetapi ditemukan defect yang membatalkan completion gate.
- Test yang terblokir environment adalah **not executed**, bukan passed.

## 2. Sprint and Ticket Map

| Sprint | Status | Objective | Tickets |
|---|---|---|---|
| 0 | Conditionally completed | Engineering foundation | TRM-004 |
| 1 | Conditionally completed | Data engineering pipeline | TRM-001, TRM-005 |
| 2 | Revalidation required | Leakage-free ML development | TRM-002, TRM-006 |
| 3 | Reopened | Reliable tracking and registry | TRM-003, TRM-004 |
| 3.5 | Required next | Architecture and model hardening | TRM-001–TRM-006, TRM-014, TRM-015 |
| 4 | In progress | Model serving | TRM-007 implemented; TRM-008 pending |
| 5 | In progress | Containerization and deployment | TRM-009 implemented; TRM-010 pending |
| 6 | Planned | System and ML observability | TRM-011, TRM-012 |
| 7 | Planned | Controlled retraining and promotion | TRM-013 |

## 3. Sprint 0 — Project Foundation

### Status

**Conditionally completed.** Repository structure, Python environment, configuration, logging,
pytest, Ruff, pre-commit, Makefile, and GitHub Actions exist.

### Remaining enhancement

- Package discovery currently must be hardened so `app`, `ml`, and pipeline code are present in a
  built distribution.
- Add reproducible dependency locking/constraints.
- CI must build and install the artifact in a clean environment.
- Split serving and training dependency profiles.

Tracked by: **TRM-004**.

### Exit criteria

- Wheel contains all runtime modules.
- Clean-install import and application smoke test pass.
- Supported Python and MLflow versions match implementation and CI.
- Ruff, format, unit, and non-optional integration gates are green.

## 4. Sprint 1 — Data Engineering Pipeline

### Status

**Conditionally completed.** Raw ingestion, explicit schema, canonical transformation, domain
validation, feature engineering, chronological split, and Parquet output exist.

### Strengths retained

- Production logic is outside notebooks.
- Schema and domain checks are testable.
- Behavioral windows exclude current/future rows.
- Output is deterministic for the tested sample.

### Required enhancement

- Declare and test the pre-transaction feature-availability contract.
- Prevent post-transaction fields from entering online model features.
- Add dataset/feature manifests.
- Reduce repeated Spark actions and global windows. **Implemented; benchmark pending.**
- Make Spark resources and output partitions configurable. **Implemented.**
- Clarify Parquet root versus train/validation/test layout. **Implemented** with an audit dataset
  separated from model-ready split projections.

Tracked by: **TRM-001** and **TRM-005**.

### Revised Definition of Done

- [x] Raw-to-canonical pipeline exists.
- [x] Structural/domain validation exists.
- [x] Chronological output splits exist.
- [x] Prediction-time feature contract is versioned and tested.
- [x] Forbidden post-event features cannot enter the online model set.
- [ ] Dataset and feature manifest is emitted.
- [ ] Spark plan avoids unnecessary repeated scans and unpartitioned global windows where practical
  — implementation exists; benchmark evidence is pending.
- [ ] Full integration pipeline passes in the supported runtime.

## 5. Sprint 2 — ML Development and Evaluation

### Status

**Revalidation required.** Model training and evaluation capabilities exist, but current model
metrics cannot be accepted for pre-transaction risk scoring while post-transaction balance fields
and derivatives remain in the model feature set.

### Existing capability

- Logistic Regression baseline.
- Random Forest and XGBoost comparison.
- Train-only preprocessing fit.
- Imbalance handling.
- Validation-based model selection and final test evaluation.
- Metrics, threshold analysis, Joblib artifacts, and JSON report.

### Required rework

- Rebuild features and retrain after TRM-001.
- Evaluate the complete test period by default.
- Review combined negative undersampling and class weighting. **Implemented:** configuration now
  selects one explicit imbalance strategy; the current candidate uses negative sampling without
  estimator-level balanced weights.
- Add configured business costs and minimum recall. **Implemented by TRM-006; configured values are
  explicitly labeled assumptions until business approval.**
- Add calibration metrics and plots. **Calibration metrics/data implemented by TRM-006; plotting is
  intentionally deferred until a reporting surface is selected.**
- Add temporal backtesting or clearly scoped single holdout limitations.
- Store dataset/schema fingerprints with the report. **Implemented by TRM-006.**

Tracked by: **TRM-002**, **TRM-006**, and **TRM-014**.

### Revised Definition of Done

- [ ] Every model feature is available at prediction time.
- [ ] Leakage tests cover direct and derived post-event features.
- [ ] Three models are compared on the rebuilt dataset.
- [ ] Selection uses validation only and isolates a rejected model from eligible candidates
  (TRM-014).
- [ ] Final test evaluation uses the selected model and frozen threshold.
- [x] PR-AUC, recall, precision, F1, calibration, alert volume, and business cost are reported.
- [ ] Sampling and class weighting strategy is justified and reproducible.
- [x] Final model artifact is loadable and tied to a dataset manifest in code; owner runtime
  validation remains pending.

## 6. Sprint 3 — Experiment Tracking and Model Registry

### Status

**Implemented with quality-gate pending.** MLflow client, nested runs, model registration, aliases,
metadata, and explicit promotion flow exist. TRM-003 fixed the MLflow 3.x `name` migration and the
registry unit/SQLite integration tests now pass. Full Spark training and remote CI remain pending.

### Required rework

- Complete the intentional MLflow 3.x `name` migration. **Completed by TRM-003.**
- Keep internal `LoggedModel.artifact_path` naming consistent or rename it everywhere in one change.
  **Completed by TRM-003.**
- Align the supported MLflow dependency floor with the API used. **Completed by TRM-003/TRM-004.**
- Restore unit and SQLite integration tests. **Completed locally.**
- Log model signature, input example, dataset manifest, and feature-contract version. **Lineage is
  implemented by TRM-006; runtime validation found the feature-only signature blocker tracked by
  TRM-015.**
- Do not claim CI green without evidence for the same commit.

Tracked by: **TRM-003** and **TRM-004**.

### Revised Definition of Done

- [x] Registry unit tests pass.
- [x] SQLite-backed MLflow integration tests pass.
- [x] Candidate model is registered and loadable.
- [x] Candidate/staging/production aliases require explicit workflow decisions.
- [x] Model version contains dataset, schema, config, code, and evaluation metadata in code; runtime
  verification remains pending.
- [ ] Full supported test suite passes outside sandbox restrictions.
- [ ] Remote CI is green for the completion commit.

## 7. Sprint 3.5 — Model and Platform Hardening

### Goal

Menutup correctness dan reproducibility gap sebelum membangun serving layer.

### Required ticket order

```text
TRM-001 Feature Contract
  -> TRM-002 Leakage-free Retraining
  -> TRM-006 Evaluation and Reproducibility
  -> TRM-014 Candidate Quality-Gate Isolation
  -> TRM-015 Feature-Only Inference and Signature Gate

TRM-003 MLflow Repair
TRM-004 Packaging and CI
TRM-005 Spark Optimization
```

TRM-003, TRM-004, dan sebagian TRM-005 dapat dikerjakan paralel setelah kontrak TRM-001 stabil.

Status eksekusi saat ini: **TRM-002 dan TRM-005 sudah diimplementasikan di codebase**. Validasi
Spark, retraining pada dataset penuh, dan benchmark before/after menunggu runtime Java/Spark yang
dapat membuka loopback connection.

### Definition of Done

- [ ] Sprint 1 feature dataset complies with pre-transaction contract.
- [ ] Sprint 2 model is retrained and re-evaluated without post-event leakage.
- [ ] Sprint 3 registry flow passes integration tests.
- [x] Dataset and model lineage is implemented in code; owner runtime verification remains pending.
- [ ] Wheel and dependency profiles are reproducible.
- [ ] Documentation and README show honest current status.

## 8. Sprint 4 — Model Serving

### Goal

Menyediakan approved model sebagai synchronous, versioned, production-oriented API tanpa
training-serving skew.

### Architecture

```text
Client
  -> authentication and request limits
  -> Pydantic request contract
  -> shared feature contract
  -> in-memory approved model
  -> calibrated score and decision
  -> non-blocking prediction event
```

### Endpoints

```text
POST /v1/predictions
GET  /health/live
GET  /health/ready
GET  /model/info
GET  /metrics
```

### Scope

- Startup/lifespan model loader using an approved registry alias or explicit local artifact.
- Strict input/output schemas and feature ordering.
- Separate liveness and readiness semantics.
- Risk score, decision, model version, feature-contract version, and request ID.
- Authentication hook, request-size limit, timeout, and safe structured logging.
- Unit, contract, integration, and concurrency tests.

### Non-goals

- Online historical feature store in the first API iteration.
- Per-request MLflow calls.
- Automatic model reload without an explicit rollout strategy.

Tracked by: **TRM-007** and **TRM-008**.

Implementation note: **TRM-007** provides a lifespan-managed local-artifact or MLflow-alias loader,
contract-derived pre-transaction features, prediction/readiness/model-info endpoints, and focused
API contract tests. **TRM-008** adds optional API-key authentication, bounded body/concurrency/time
limits, in-memory idempotency, correlation IDs, privacy-safe errors/logs, and security tests.

### Definition of Done

- [x] API code derives and scores the exact shared feature contract (TRM-007; owner runtime
  validation pending).
- [x] Unknown, missing, and invalid fields are rejected, not silently defaulted (TRM-007).
- [x] Readiness is false until model and schema are valid (TRM-007).
- [x] Model version and threshold are observable (TRM-007).
- [x] No secret, account ID, or raw sensitive payload is included in API error responses or
  application prediction logs (TRM-008; owner validation pending).
- [x] API security, idempotency, timeout, OpenAPI, and concurrency tests are implemented
  (TRM-008; owner validation pending).

## 9. Sprint 5 — Containerization and Deployment

### Goal

Menyediakan reproducible serving runtime, membuktikan performance envelope, lalu mendemonstrasikan
orchestration tanpa mengklaim HA yang tidak dimiliki local kind.

### Phase A — Podman and Compose

- Separate serving and training images.
- Multi-stage, non-root, minimal serving image.
- Healthcheck and read-only filesystem where possible.
- Podman Compose for API, MLflow profile, database/artifact services, and monitoring dependencies.
- Image scan and SBOM.

Tracked by: **TRM-009**.

### Phase B — Load test and kind

- Establish p50/p95/p99 latency, throughput, memory, and startup baseline.
- Add Kubernetes Deployment, Service, ConfigMap, Secret references, probes, resources, rolling update,
  and PodDisruptionBudget.
- Add HPA only when justified by measured behavior.
- Demonstrate deployment and rollback.

Tracked by: **TRM-010**.

### Definition of Done

- [ ] Clean image can start and become ready without source-tree mounts.
- [ ] Compose provides a one-command local demo.
- [ ] Load-test report documents capacity and bottleneck.
- [ ] kind deployment performs successful prediction and controlled rollout.
- [ ] Documentation states that single-node kind is not host/node HA.

## 10. Sprint 6 — Monitoring and Model Observability

### Goal

Mendeteksi service failure, data-quality degradation, prediction drift, dan delayed model-quality
regression without adding heavy work to the request path.

### System observability

- Prometheus request/error/latency metrics.
- Model load, readiness, validation rejection, and prediction metrics.
- Grafana dashboard and actionable alerts.
- No PII or unbounded-cardinality labels.

Tracked by: **TRM-011**.

### ML observability

- Versioned prediction event schema.
- Privacy-aware prediction and delayed label store.
- Scheduled Evidently data-quality, feature-drift, prediction-drift, and performance report.
- Calibration and metric segmentation by model version.

Tracked by: **TRM-012**.

### Definition of Done

- [ ] Dashboard shows request rate, p95 latency, error rate, readiness, and model version.
- [ ] Alert scenarios are testable and documented.
- [ ] Prediction events can be joined with delayed labels.
- [ ] Evidently job runs asynchronously and produces reproducible reports.
- [ ] Monitoring failure does not block synchronous prediction.

## 11. Sprint 7 — Controlled Retraining and Promotion

### Goal

Menutup lifecycle dengan retraining yang reproducible serta promotion/rollback yang terkendali.

### Flow

```text
Drift or performance alert
  -> reviewed dataset snapshot
  -> retraining
  -> temporal and leakage quality gates
  -> candidate alias
  -> explicit approval
  -> production alias and controlled rollout
  -> rollback when quality or service SLO regresses
```

Tracked by: **TRM-013**.

### Definition of Done

- [ ] Retraining consumes an immutable dataset manifest.
- [ ] New model must beat configured gates, not merely the previous F1 score.
- [ ] Failed candidates remain traceable but cannot be promoted.
- [ ] Approval identity and reason are recorded.
- [ ] Promotion and rollback scenarios are demonstrated end to end.
- [ ] Drift alone never performs unattended production promotion.

## 12. Final Project Completion Criteria

The project is complete when a reviewer can execute a documented scenario:

```text
Raw data
  -> leakage-safe feature snapshot
  -> reproducible training and evaluation
  -> MLflow candidate registration
  -> explicit production promotion
  -> containerized API prediction
  -> system and ML monitoring
  -> drift/performance scenario
  -> retraining candidate
  -> controlled promotion or rejection
  -> rollback demonstration
```

Required evidence:

- One-command local setup and teardown.
- Green CI for the final commit.
- Versioned architecture, model card, dataset manifest, and test/load reports.
- Explicit limitations for synthetic data, local storage, kind availability, and missing cloud HA.
- No known P0/P1 ticket remains open.
