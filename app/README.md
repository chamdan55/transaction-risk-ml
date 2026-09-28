# Application Layer

Folder `app/` berisi application boundary untuk health, configuration, logging, metrics, dan synchronous
model inference API.

## Current Components

- `main.py` — FastAPI application entry point.
- `core/config.py` — application settings.
- `core/logging.py` — centralized logging configuration.
- `services/metrics.py` — Prometheus metrics with bounded, privacy-safe labels.
- `api/schemas.py` — strict pre-transaction request/response contract.
- `services/model_provider.py` — startup-only local/MLflow model loader.
- `services/feature_preparation.py` — shared prediction-time feature derivation.
- `services/reliability.py` — bounded concurrency, timeout, and idempotency primitives.
- `ml/monitoring/` — versioned event/label contracts and a bounded SQLite event writer.

Model training tidak dijalankan dari folder ini. Training dan evaluation menggunakan entry point di
`pipelines/`; inference memuat artifact yang sudah divalidasi saat startup.

`/health/live` dan `/health/ready` tetap public untuk probe. Untuk melindungi `/v1/*` dan
`/model/info` pada deployment internal, injeksikan `API_AUTH_ENABLED=true` dan `API_KEY` dari
secret store atau environment proses. TLS, network policy, dan enterprise identity berada pada
reverse proxy/ingress, bukan di aplikasi ini.

`POST /v1/predictions` mengembalikan `feedback_id` acak yang dapat disimpan sistem pemanggil untuk
mengirim label kemudian ke `POST /v1/feedback/labels`. Event prediksi masuk ke bounded in-process queue
dan ditulis ke SQLite oleh worker latar belakang. Penyimpanan telemetry tidak menjadi syarat readiness;
event yang gagal antre atau masih pending saat proses dihentikan dapat hilang dan dihitung/logged.
