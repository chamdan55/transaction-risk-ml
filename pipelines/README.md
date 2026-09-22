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

# Compare exact and scalable Spark split plans/timings
python -m pipelines.benchmark_spark --config configs/data.yaml
```

## Output Policy

- Data outputs ditulis ke `data/processed/`.
- Full feature schema ditulis ke `data/processed/features/audit/`; training splits di bawah
  `data/processed/features/{train,validation,test}/` hanya berisi model-ready contract projection.
- Model artifacts, evaluation report, model card, dan lineage manifest ditulis ke `artifacts/`.
- Pipeline harus dapat dijalankan ulang dari configuration tanpa parameter hard-coded.
