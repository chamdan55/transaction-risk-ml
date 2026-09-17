from pathlib import Path

import pytest

from ml.tracking.client import TrackingClientError
from ml.tracking.metadata import build_run_metadata

PROJECT_ROOT = Path(__file__).parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


def _dataset_summary():
    return {
        "feature_columns": ["amount_log", "transactions_last_1h"],
        "split_summaries": {
            "train": {"row_count": 100},
            "validation": {"row_count": 20},
            "test": {"row_count": 20},
        },
    }


def test_run_metadata_contains_dataset_and_reproducibility_fields():
    metadata = build_run_metadata(
        dataset_summary=_dataset_summary(),
        target_column="is_fraud",
        config_path=MODEL_CONFIG_PATH,
        repository_root=PROJECT_ROOT,
    )

    assert metadata.dataset_name == "paysim"
    assert metadata.target_column == "is_fraud"
    assert metadata.feature_count == 2
    assert metadata.train_row_count == 100
    assert len(metadata.config_hash) == 64
    assert metadata.git_commit != ""
    assert metadata.training_timestamp.endswith("+00:00")


def test_run_metadata_rejects_incomplete_dataset_summary():
    with pytest.raises(TrackingClientError, match="split_summaries"):
        build_run_metadata(
            dataset_summary={},
            target_column="is_fraud",
            config_path=MODEL_CONFIG_PATH,
        )
