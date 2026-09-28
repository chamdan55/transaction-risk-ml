import json

import pytest

from ml.training.retraining import (
    RetrainingError,
    create_approved_manifest,
    evaluate_retraining_quality,
    expected_calibration_error,
    load_approved_manifest,
    load_quality_policy,
    validate_manifest_matches,
    write_approved_manifest,
)


def _manifest():
    return {
        "manifest_id": "a" * 64,
        "dataset_path": "data/processed/features",
        "content_fingerprint": "b" * 64,
        "config_fingerprint": "c" * 64,
        "code_fingerprint": "d" * 64,
        "feature_columns": ("amount", "transaction_type"),
        "generated_at": "2026-09-25T00:00:00+00:00",
    }


def _metrics(*, pr_auc=0.90, precision=0.80, recall=0.90, cost=0.01, brier=0.05, alert=0.01):
    return {
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "expected_cost_per_transaction": cost,
        "brier_score": brier,
        "alert_rate": alert,
    }


def test_approved_manifest_round_trip_and_content_digest(tmp_path):
    path = tmp_path / "approved.json"
    approved = create_approved_manifest(
        _manifest(),
        approved_by="risk-owner",
        reason="Reviewed dataset snapshot.",
    )
    write_approved_manifest(path, approved)

    loaded = load_approved_manifest(path)
    assert loaded["manifest"]["feature_columns"] == ["amount", "transaction_type"]
    validate_manifest_matches(loaded, _manifest())

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["approval"]["reason"] = "changed after approval"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RetrainingError, match="digest does not match"):
        load_approved_manifest(path)


def test_approved_manifest_is_never_overwritten(tmp_path):
    path = tmp_path / "approved.json"
    write_approved_manifest(
        path,
        create_approved_manifest(_manifest(), approved_by="owner", reason="approved"),
    )

    with pytest.raises(RetrainingError, match="Refusing to overwrite"):
        write_approved_manifest(
            path,
            create_approved_manifest(_manifest(), approved_by="owner", reason="second approval"),
        )


def test_manifest_match_ignores_generation_time_but_rejects_changed_fingerprints():
    approved = {"manifest": _manifest()}
    current = _manifest() | {"generated_at": "2026-09-26T00:00:00+00:00"}
    validate_manifest_matches(approved, current)

    current["content_fingerprint"] = "e" * 64
    with pytest.raises(RetrainingError, match="content_fingerprint"):
        validate_manifest_matches(approved, current)


def test_manifest_match_rejects_changed_feature_order():
    approved = create_approved_manifest(_manifest(), approved_by="owner", reason="reviewed")
    current = _manifest() | {"feature_columns": ("transaction_type", "amount")}

    with pytest.raises(RetrainingError, match="feature_columns"):
        validate_manifest_matches(approved, current)


def test_quality_policy_compares_candidate_against_incumbent_on_same_test_set():
    policy = load_quality_policy("configs/retraining.yaml")
    result = evaluate_retraining_quality(
        schema_contract_passed=True,
        duplicate_transaction_id_count=0,
        split_overlap_count=0,
        validation_metrics=_metrics(),
        test_metrics=_metrics(cost=0.015),
        test_calibration_error=0.03,
        test_row_count=10000,
        test_positive_count=100,
        incumbent_metrics=_metrics(cost=0.01) | {"calibration_error": 0.02},
        policy=policy,
    )

    assert result["status"] == "passed"
    assert result["failed_checks"] == []
    assert any(
        check["name"] == "comparison.expected_cost_per_transaction_increase"
        for check in result["checks"]
    )


def test_quality_gate_rejects_low_recall_and_cost_regression():
    policy = load_quality_policy("configs/retraining.yaml")
    result = evaluate_retraining_quality(
        schema_contract_passed=True,
        duplicate_transaction_id_count=0,
        split_overlap_count=0,
        validation_metrics=_metrics(recall=0.70),
        test_metrics=_metrics(recall=0.70, cost=0.03),
        test_calibration_error=0.03,
        test_row_count=10000,
        test_positive_count=100,
        incumbent_metrics=_metrics(cost=0.01) | {"calibration_error": 0.02},
        policy=policy,
    )

    assert result["status"] == "rejected"
    assert "validation.recall" in result["failed_checks"]
    assert "test.recall" in result["failed_checks"]
    assert "comparison.expected_cost_per_transaction_increase" in result["failed_checks"]


def test_expected_calibration_error_is_weighted_by_bin_volume():
    bins = [
        {"count": 9, "mean_predicted_probability": 0.1, "observed_positive_rate": 0.0},
        {"count": 1, "mean_predicted_probability": 0.9, "observed_positive_rate": 1.0},
    ]

    assert expected_calibration_error(bins) == pytest.approx(0.10)
