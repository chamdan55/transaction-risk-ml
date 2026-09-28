from types import SimpleNamespace

import pytest

from ml.tracking.client import TrackingClientError
from ml.tracking.config import TrackingConfig
from ml.tracking.registry import (
    LoggedModel,
    promote_registered_model,
    rollback_registered_model,
)


class FakeTrackingClient:
    def __init__(self, *, aliases=None, versions=None):
        self.alias_call = None
        self.tags_call = None
        self.aliases = aliases or {}
        self.versions = versions or {}
        self.config = TrackingConfig("http://mlflow", "experiment", "transaction-risk-model")

    def get_model_alias_version(self, *, alias):
        return self.aliases.get(alias)

    def set_model_alias(self, **kwargs):
        self.alias_call = kwargs
        self.aliases[kwargs["alias"]] = kwargs["version"]

    def delete_model_alias(self, *, alias):
        self.aliases.pop(alias, None)

    def get_model_version(self, *, version):
        if version not in self.versions:
            self.versions[version] = SimpleNamespace(
                tags={
                    "candidate_status": "candidate",
                    "signature_validation": "passed",
                    "serving_input_validation": "passed",
                },
                status="READY",
            )
        return self.versions[version]

    def set_model_version_tags(self, **kwargs):
        self.tags_call = kwargs
        self.versions.setdefault(kwargs["version"], SimpleNamespace(tags={}, status="READY"))
        self.versions[kwargs["version"]].tags.update(kwargs["tags"])


def test_promotion_requires_explicit_approval_and_sets_stage_alias():
    client = FakeTrackingClient()
    reference = LoggedModel(
        model_name="xgboost",
        artifact_path="model",
        model_uri="runs:/run/model",
        registered_model_name="transaction-risk-model",
        registered_model_version="7",
    )

    record = promote_registered_model(
        client,
        reference,
        stage="staging",
        approved_by="risk-reviewer",
        reason="PR-AUC and recall passed the documented quality gate.",
    )

    assert client.alias_call == {
        "registered_model_name": "transaction-risk-model",
        "alias": "staging",
        "version": "7",
    }
    assert client.tags_call["tags"]["candidate_status"] == "staging"
    assert client.tags_call["tags"]["promotion.approved_by"] == "risk-reviewer"
    assert client.tags_call["tags"]["promotion.previous_version"] == "none"
    assert client.tags_call["tags"]["promotion.new_version"] == "7"
    assert record.new_version == "7"
    assert record.previous_version is None


def test_promotion_records_prior_alias_and_rollback_restores_it():
    client = FakeTrackingClient(
        aliases={"production": "6"},
        versions={
            "6": SimpleNamespace(tags={}, status="READY"),
            "7": SimpleNamespace(
                tags={
                    "candidate_status": "candidate",
                    "signature_validation": "passed",
                    "serving_input_validation": "passed",
                },
                status="READY",
            ),
        },
    )
    reference = LoggedModel(
        model_name="random_forest",
        artifact_path="model",
        model_uri="runs:/run-7/model",
        registered_model_name="transaction-risk-model",
        registered_model_version="7",
    )

    promoted = promote_registered_model(
        client,
        reference,
        stage="production",
        approved_by="risk-reviewer",
        reason="Candidate passed all approved gates.",
    )
    rollback = rollback_registered_model(
        client,
        stage="production",
        approved_by="on-call",
        reason="Readiness failed after rollout.",
        expected_current_version="7",
    )

    assert promoted.previous_version == "6"
    assert client.aliases["production"] == "6"
    assert rollback.source_version == "7"
    assert rollback.restored_version == "6"
    assert client.versions["7"].tags["rollback.to_version"] == "6"


def test_rollback_refuses_a_concurrent_alias_change():
    client = FakeTrackingClient(aliases={"production": "8"})

    with pytest.raises(TrackingClientError, match="refusing to roll back a concurrent promotion"):
        rollback_registered_model(
            client,
            stage="production",
            approved_by="on-call",
            reason="Readiness failed.",
            expected_current_version="7",
        )
