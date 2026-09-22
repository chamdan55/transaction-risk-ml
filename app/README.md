# Application Layer

Folder `app/` berisi application boundary untuk health, configuration, logging, dan synchronous
model inference API.

## Current Components

- `main.py` — FastAPI application entry point.
- `core/config.py` — application settings.
- `core/logging.py` — centralized logging configuration.
- `api/schemas.py` — strict pre-transaction request/response contract.
- `services/model_provider.py` — startup-only local/MLflow model loader.
- `services/feature_preparation.py` — shared prediction-time feature derivation.
- `services/reliability.py` — bounded concurrency, timeout, and idempotency primitives.

Model training tidak dijalankan dari folder ini. Training dan evaluation menggunakan entry point di
`pipelines/`; inference memuat artifact yang sudah divalidasi saat startup.

`/health/live` dan `/health/ready` tetap public untuk probe. Untuk melindungi `/v1/*` dan
`/model/info` pada deployment internal, injeksikan `API_AUTH_ENABLED=true` dan `API_KEY` dari
secret store atau environment proses. TLS, network policy, dan enterprise identity berada pada
reverse proxy/ingress, bukan di aplikasi ini.
