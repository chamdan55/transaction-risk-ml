# Configuration

Folder ini berisi konfigurasi version-controlled untuk pipeline data dan model.

## Files

- `data.yaml` — lokasi raw/processed/sample data, seed, timestamp dasar, dan rasio split.
- `model.yaml` — dataset input training, target, excluded columns, model parameters, evaluation thresholds, dan output artifacts.

## Usage

Konfigurasi model digunakan oleh training pipeline:

```powershell
python -m pipelines.train_models --config configs/model.yaml
```

Jangan menaruh credential, absolute path lokal, atau parameter rahasia di folder ini. Gunakan path relatif terhadap repository agar pipeline tetap reproducible.
