from pathlib import Path

from ml.tracking.lineage import build_dataset_manifest


def _summary(row_count: int = 3):
    return {
        "feature_columns": ["amount_log", "transaction_type"],
        "split_summaries": {
            "train": {"row_count": row_count, "min_timestamp": "2026-01-01"},
            "validation": {"row_count": 1, "min_timestamp": "2026-01-02"},
            "test": {"row_count": 1, "min_timestamp": "2026-01-03"},
        },
    }


def test_dataset_manifest_is_stable_for_identical_inputs(tmp_path):
    dataset_path = tmp_path / "features"
    (dataset_path / "train").mkdir(parents=True)
    (dataset_path / "train" / "part-000.parquet").write_bytes(b"dataset")
    config_path = tmp_path / "model.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")

    first = build_dataset_manifest(
        dataset_summary=_summary(),
        dataset_path=dataset_path,
        config_path=config_path,
        target_column="is_fraud",
        feature_contract_version="pre-transaction-v1",
        repository_root=Path(__file__).parents[2],
    )
    second = build_dataset_manifest(
        dataset_summary=_summary(),
        dataset_path=dataset_path,
        config_path=config_path,
        target_column="is_fraud",
        feature_contract_version="pre-transaction-v1",
        repository_root=Path(__file__).parents[2],
    )

    assert first.manifest_id == second.manifest_id
    assert first.stable_dict() == second.stable_dict()

    changed = build_dataset_manifest(
        dataset_summary=_summary(row_count=4),
        dataset_path=dataset_path,
        config_path=config_path,
        target_column="is_fraud",
        feature_contract_version="pre-transaction-v1",
        repository_root=Path(__file__).parents[2],
    )
    assert changed.manifest_id != first.manifest_id
