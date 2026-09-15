# Application Layer

Folder `app/` berisi application boundary untuk health endpoint, configuration, logging, dan future model inference API.

## Current Components

- `main.py` — FastAPI application entry point.
- `core/config.py` — application settings.
- `core/logging.py` — centralized logging configuration.
- `api/routes/` — API route namespace.
- `inference/` — reserved untuk model loading dan inference service.

Model training tidak dijalankan dari folder ini. Training dan evaluation menggunakan entry point di `pipelines/`; inference API akan menggunakan artifacts yang dihasilkan training pipeline pada sprint lanjutan.
