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
```

## Output Policy

- Data outputs ditulis ke `data/processed/`.
- Model artifacts dan evaluation report ditulis ke `artifacts/`.
- Pipeline harus dapat dijalankan ulang dari configuration tanpa parameter hard-coded.
