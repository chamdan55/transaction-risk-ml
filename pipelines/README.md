# Executable Pipelines

Folder ini berisi entry point yang menjalankan flow end-to-end. Reusable logic tetap berada di `ml/`.

## Main Commands

```powershell
# Full Sprint 1 data pipeline
python -m pipelines.run_pipeline

# Individual data stages
python -m pipelines.validate_paysim
python -m pipelines.canonicalize_paysim
python -m pipelines.validate_canonical
python -m pipelines.profile_canonical

# Sprint 2 model training and evaluation
python -m pipelines.train_models --config configs/model.yaml

# Create a reviewed, one-time dataset/config/code snapshot and run controlled retraining
python -m pipelines.prepare_retraining_manifest --approved-by "risk-reviewer" --reason "Reviewed snapshot"
python -m pipelines.retrain_model --approved-manifest artifacts/retraining/approved-manifest.json

# Roll back registry alias; or export and deploy an approved production release to kind
python -m pipelines.rollback_model --stage production --approved-by "on-call" --reason "Readiness regression"
python -m pipelines.export_model_bundle
python -m pipelines.rollout_model --bundle-path artifacts/model-bundles/random_forest-v7

# Compare exact and scalable Spark split plans/timings
python -m pipelines.benchmark_spark --config configs/data.yaml
```

## Output Policy

- Data outputs ditulis ke `data/processed/`.
- Full feature schema ditulis ke `data/processed/features/audit/`; training splits di bawah
  `data/processed/features/{train,validation,test}/` hanya berisi model-ready contract projection.
- Model artifacts, evaluation report, model card, dan lineage manifest ditulis ke `artifacts/`.
- Controlled retraining writes its local artifacts into unique `artifacts/retraining/runs/` folders;
  rejected or unpromoted candidates never replace the normal serving bundle.
- Pipeline harus dapat dijalankan ulang dari configuration tanpa parameter hard-coded.
