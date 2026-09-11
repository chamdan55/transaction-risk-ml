from datetime import datetime

from ml.data.profiling import profile_balance_consistency_by_type
from ml.data.schema import CANONICAL_TRANSACTION_SCHEMA


def test_balance_profile_by_transaction_type(spark):
    rows = [
        (
            "a" * 64,
            datetime(2026, 1, 1, 1, 0, 0),
            "PAYMENT",
            "C123",
            "M456",
            100.0,
            1000.0,
            900.0,
            500.0,
            600.0,
            0,
            0,
        ),
        (
            "b" * 64,
            datetime(2026, 1, 1, 2, 0, 0),
            "PAYMENT",
            "C456",
            "M789",
            200.0,
            1000.0,
            900.0,
            500.0,
            700.0,
            0,
            0,
        ),
    ]

    df = spark.createDataFrame(
        rows,
        schema=CANONICAL_TRANSACTION_SCHEMA,
    )

    result = profile_balance_consistency_by_type(df)

    rows = result.collect()

    assert len(rows) == 1

    row = rows[0]

    assert row.transaction_type == "PAYMENT"
    assert row.transaction_count == 2
    assert row.inconsistent_origin_count == 1
    assert row.inconsistent_destination_count == 0
    assert row.origin_inconsistency_rate == 0.5
    assert row.destination_inconsistency_rate == 0.0
