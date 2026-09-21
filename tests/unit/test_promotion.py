from ml.tracking.registry import LoggedModel, promote_registered_model


class FakeTrackingClient:
    def __init__(self):
        self.alias_call = None
        self.tags_call = None

    def set_model_alias(self, **kwargs):
        self.alias_call = kwargs

    def set_model_version_tags(self, **kwargs):
        self.tags_call = kwargs


def test_promotion_requires_explicit_approval_and_sets_stage_alias():
    client = FakeTrackingClient()
    reference = LoggedModel(
        model_name="xgboost",
        artifact_path="model",
        model_uri="runs:/run/model",
        registered_model_name="transaction-risk-model",
        registered_model_version="7",
    )

    promote_registered_model(
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
