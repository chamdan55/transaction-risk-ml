# Feature Engineering

Folder ini berisi transformasi feature yang dibangun dari canonical transaction data.

## Feature Groups

- `amount.py` — log amount dan amount ratios.
- `balance.py` — balance deltas, ratios, mismatch, zero/depleted, dan balance behavior flags.
- `transaction.py` — transaction-level features.
- `timestamp.py` — hour, day-of-week, dan time-derived features.
- `behavior.py` — historical transaction counts, amount aggregates, dan unique destinations.
- `pipeline.py` — komposisi seluruh feature builder.
- `schema.py` — feature schema dan feature value validation.

## Leakage Policy

Feature harus tersedia pada waktu prediksi. Identifier, timestamp mentah, target, dan `is_flagged_fraud` tidak boleh menjadi model input. Balance fields yang hanya tersedia setelah transaksi selesai harus ditinjau sebelum digunakan untuk model.

Feature output utama ditulis ke:

```text
data/processed/features/
```
