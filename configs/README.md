# Configuration

Folder ini berisi konfigurasi version-controlled untuk pipeline data dan model.

## Files

- `data.yaml` — lokasi raw/processed/sample data, seed, timestamp dasar, rasio split, strategi
  chronological split, dan resource Spark/output partitions.
- `model.yaml` — dataset input training, feature contract, target, excluded columns, strategi
  imbalance/sampling, model parameters, business-cost/recall/threshold policy, calibration bins,
  dan output artifacts/model card.

## Usage

Konfigurasi model digunakan oleh training pipeline:

```powershell
python -m pipelines.train_models --config configs/model.yaml
```

Jangan menaruh credential, absolute path lokal, atau parameter rahasia di folder ini. Gunakan path relatif terhadap repository agar pipeline tetap reproducible.
