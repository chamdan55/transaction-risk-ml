# Model Evaluation

Folder ini berisi evaluasi model dan pemilihan threshold untuk fraud classification.

## Components

- `metrics.py` — precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, Brier score, alert
  rate, dan calibration bins.
- `threshold.py` — threshold sweep, recall constraint, metric strategy, dan expected business cost
  strategy.
- `selection.py` — validation-based production candidate selection, per-model quality-gate
  isolation, dan final test evaluation.

## Evaluation Order

```text
Validation predictions
        ↓
Threshold analysis
        ↓
Production candidate selection
        ↓
Final test evaluation
```

Model yang tidak memiliki threshold memenuhi quality gate dicatat sebagai `rejected` beserta
threshold sweep dan alasan penolakannya. Rejection satu model tidak membatalkan evaluasi model lain;
pipeline hanya gagal jika tidak ada model yang eligible.

Model dan threshold tidak boleh dipilih berdasarkan test set. Final test menggunakan full period
secara default dan hanya dievaluasi setelah candidate serta threshold dibekukan. Evaluation report
menyimpan calibration data, business-cost assumptions, alert rate, lineage manifest, dan model card
di `artifacts/evaluation_report.json` serta `artifacts/model_card.md`.
