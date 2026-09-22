# Data Layer

Data layer menangani perjalanan data dari raw PaySim CSV sampai feature dataset yang siap digunakan model.

## Responsibilities

- Membuat Spark session.
- Membaca raw data dengan schema eksplisit.
- Menjalankan structural dan domain validation.
- Mengubah raw data menjadi canonical transaction schema.
- Membuat profiling dataset.
- Melakukan chronological train/validation/test split.

## Important Contracts

- Target model: `is_fraud`.
- Split berdasarkan urutan waktu, bukan random split.
- `transaction_id` harus unik dan tidak overlap antar split.
- Output downstream disimpan sebagai Parquet.
- Model input columns didefinisikan oleh versioned `MODEL_FEATURE_COLUMNS` di
  `ml/contracts/features.py`; `split.py` hanya mempertahankan re-export untuk compatibility.
- Contract saat ini adalah `pre-transaction-v1`: post-event, label, identifier, dan historical
  features yang belum memiliki online state source tidak boleh masuk model input.

## Related Outputs

```text
data/processed/canonical/
data/processed/features/audit/       # full feature schema for profiling/audit only
data/processed/features/train/
data/processed/features/validation/
data/processed/features/test/
```

Split Parquet di bawah `train`, `validation`, dan `test` adalah model-ready projection yang hanya
memuat `transaction_id`, fitur `pre-transaction-v1`, dan target `is_fraud`. Schema lengkap yang
memuat post-event fields disimpan terpisah di `features/audit` dan tidak boleh dibaca sebagai model
input.

`data/processed/features/lineage.json` menyimpan strategi split dan time range tiap split karena
kolom `timestamp` sengaja tidak dibawa ke dataset model-ready.
