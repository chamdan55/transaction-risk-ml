from datetime import datetime, timedelta

import pytest
from pyspark.sql import SparkSession

from ml.features.behavior import add_behavior_features


def test_behavior_features_use_only_strictly_prior_transactions(
    spark: SparkSession,
):
    base_time = datetime(2026, 1, 1, 10, 0, 0)
    data = [
        ("tx-001", base_time - timedelta(hours=2), "A", "D1", 50.0),
        ("tx-002", base_time - timedelta(minutes=30), "A", "D2", 100.0),
        ("tx-003", base_time, "A", "D1", 200.0),
        ("tx-004", base_time + timedelta(hours=1), "A", "D3", 300.0),
    ]
    df = spark.createDataFrame(
        data,
        [
            "transaction_id",
            "timestamp",
            "origin_account_id",
            "destination_account_id",
            "amount",
        ],
    )

    rows = {row["transaction_id"]: row for row in add_behavior_features(df).collect()}

    current = rows["tx-003"]
    assert current["transactions_last_1h"] == 1
    assert current["transactions_last_24h"] == 2
    assert current["amount_sum_last_24h"] == pytest.approx(150.0)
    assert current["unique_destinations_last_30d"] == 2

    future = rows["tx-004"]
    assert future["transactions_last_1h"] == 1
    assert future["transactions_last_24h"] == 3
    assert future["amount_sum_last_24h"] == pytest.approx(350.0)
    assert future["unique_destinations_last_30d"] == 2


def test_behavior_features_do_not_include_same_timestamp_rows(
    spark: SparkSession,
):
    timestamp = datetime(2026, 1, 1, 10, 0, 0)
    df = spark.createDataFrame(
        [
            ("tx-001", timestamp, "A", "D1", 50.0),
            ("tx-002", timestamp, "A", "D2", 100.0),
        ],
        [
            "transaction_id",
            "timestamp",
            "origin_account_id",
            "destination_account_id",
            "amount",
        ],
    )

    rows = add_behavior_features(df).collect()

    assert all(row["transactions_last_24h"] == 0 for row in rows)
    assert all(row["amount_sum_last_24h"] == 0.0 for row in rows)
    assert all(row["unique_destinations_last_30d"] == 0 for row in rows)
