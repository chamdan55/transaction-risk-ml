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
- Model input columns didefinisikan oleh `MODEL_FEATURE_COLUMNS` di `split.py`.

## Related Outputs

```text
data/processed/canonical/
data/processed/features/train/
data/processed/features/validation/
data/processed/features/test/
```
