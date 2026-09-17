# Experiment Tracking

Folder ini berisi integration boundary untuk MLflow pada Sprint 3.

## Components

- `config.py` — typed loader untuk `configs/tracking.yaml`.
- `client.py` — wrapper untuk tracking URI, experiment creation/reuse, dan run lifecycle.
- `logging.py` — flattening dan logging training parameters ke active MLflow run.
- `metadata.py` — dataset summary, config hash, git commit, dan training timestamp.
- `registry.py` — logging, registration reference, dan loading model artifact.

## Contract

- Experiment name: `transaction-risk-classification`.
- Registered model name: `transaction-risk-model`.
- Tracking URI dan artifact location berasal dari `configs/tracking.yaml`.
- Tracking storage lokal tidak di-commit ke Git.

Model artifact disimpan sebagai satu bundle yang mencakup preprocessing dan estimator. Model registry metadata tetap berasal dari MLflow run.
