# Transaction Risk ML — Engineering Backlog

## How to Use This Backlog

Setiap file adalah tiket implementasi yang dapat diberikan langsung kepada AI agent. Kerjakan tiket
sesuai dependency order dan jangan menganggap checkbox sprint selesai tanpa bukti test aktual.

Aturan umum untuk semua agent:

1. Baca `docs/arsitektur_dan_tech_stack.md`, `docs/sprint_docs.md`, dan tiket yang diberikan.
2. Inspect implementation dan tests sebelum mengubah kode.
3. Pertahankan perubahan user yang tidak terkait.
4. Gunakan configuration-driven design dan tambahkan test untuk behavior baru.
5. Jangan menandai test yang tidak dijalankan sebagai passed.
6. Update dokumentasi yang langsung berubah karena implementasi tiket.
7. Laporkan command validasi, hasil, limitation, dan file yang diubah.

## Backlog Order

| ID | Priority | Sprint | Title | Depends on | Status |
|---|---|---|---|---|---|
| [TRM-001](TRM-001-pre-transaction-feature-contract.md) | P0 | 3.5 | Define pre-transaction feature contract | — | Implemented — validation partial |
| [TRM-002](TRM-002-rebuild-leakage-free-models.md) | P0 | 3.5 | Rebuild leakage-free dataset and models | TRM-001 | Implemented — Spark validation pending |
| [TRM-003](TRM-003-complete-mlflow-name-migration.md) | P0 | 3.5 | Complete MLflow 3.x name migration | — | Implemented — tests pass |
| [TRM-004](TRM-004-packaging-dependencies-ci.md) | P1 | 3.5 | Harden packaging, dependencies, and CI | — | Implemented — CI verification pending |
| [TRM-005](TRM-005-optimize-spark-pipeline.md) | P1 | 3.5 | Optimize Spark split, sampling, and scans | TRM-001 | Implemented — benchmark pending |
| [TRM-006](TRM-006-business-evaluation-lineage.md) | P1 | 3.5 | Add business evaluation, calibration, and lineage | TRM-001, TRM-002 | Implemented — owner runtime validation pending |
| [TRM-014](TRM-014-isolate-candidate-quality-gates.md) | P0 | 3.5 | Isolate per-model gates and preserve rejected evaluations | TRM-006 | Completed — Pytest and full training passed |
| [TRM-015](TRM-015-fix-feature-only-inference-signature.md) | P0 | 3.5 | Fix feature-only inference and enforce MLflow signature validation | TRM-002, TRM-003, TRM-006, TRM-014 | Implemented — owner validation pending |
| [TRM-007](TRM-007-fastapi-serving.md) | P0 | 4 | Build shared inference contract and FastAPI serving | TRM-002, TRM-003, TRM-006, TRM-014, TRM-015 | Implemented — owner validation passed |
| [TRM-008](TRM-008-api-security-reliability.md) | P1 | 4 | Add API security, reliability, and contract tests | TRM-007 | Implemented — owner validation passed |
| [TRM-009](TRM-009-container-compose.md) | P1 | 5 | Build minimal images and Podman Compose runtime | TRM-004, TRM-008 | Implemented — owner Podman validation pending |
| [TRM-010](TRM-010-load-test-kind.md) | P2 | 5 | Establish load baseline and kind deployment | TRM-009 | Blocked |
| [TRM-011](TRM-011-runtime-observability.md) | P1 | 6 | Add runtime metrics, dashboards, and alerts | TRM-008, TRM-009 | Blocked |
| [TRM-012](TRM-012-ml-observability.md) | P1 | 6 | Add prediction events, label feedback, and drift jobs | TRM-007, TRM-011 | Blocked |
| [TRM-013](TRM-013-controlled-retraining.md) | P1 | 7 | Implement controlled retraining, promotion, and rollback | TRM-006, TRM-012 | Blocked |

## Recommended Execution Waves

```text
Wave 1: TRM-001, TRM-003, TRM-004
Wave 2: TRM-002, TRM-005
Wave 3: TRM-006
Wave 3.5: TRM-014
Wave 3.6: TRM-015
Wave 4: TRM-007
Wave 5: TRM-008
Wave 6: TRM-009
Wave 7: TRM-010, TRM-011
Wave 8: TRM-012
Wave 9: TRM-013
```

TRM-010 is optional for the minimum Docker Compose demo, but required for the Kubernetes portfolio
claim. Priority indicates architectural risk, not estimated effort.
