# Model Evaluation

Folder ini berisi evaluasi model dan pemilihan threshold untuk fraud classification.

## Components

- `metrics.py` — precision, recall, F1, ROC-AUC, PR-AUC, dan confusion matrix.
- `threshold.py` — threshold sweep dan expected business cost analysis.
- `selection.py` — validation-based production candidate selection dan final test evaluation.

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

Model dan threshold tidak boleh dipilih berdasarkan test set. Evaluation report disimpan sebagai JSON di `artifacts/evaluation_report.json`.
