from ml.data.schema import PAYSIM_SCHEMA


def test_paysim_schema_columns():
    expected_columns = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]

    actual_columns = PAYSIM_SCHEMA.fieldNames()

    assert actual_columns == expected_columns
