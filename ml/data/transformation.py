from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def transform_to_canonical(
    df: DataFrame,
    base_timestamp: str,
) -> DataFrame:
    transaction_id = F.sha2(
        F.concat_ws(
            "||",
            F.col("step").cast("string"),
            F.col("type"),
            F.col("amount").cast("string"),
            F.col("nameOrig"),
            F.col("nameDest"),
        ),
        256,
    )

    timestamp = F.to_timestamp(F.lit(base_timestamp)) + F.make_interval(hours=F.col("step"))

    return df.select(
        transaction_id.alias("transaction_id"),
        timestamp.alias("timestamp"),
        F.col("type").alias("transaction_type"),
        F.col("nameOrig").alias("origin_account_id"),
        F.col("nameDest").alias("destination_account_id"),
        F.col("amount"),
        F.col("oldbalanceOrg").alias("origin_balance_before"),
        F.col("newbalanceOrig").alias("origin_balance_after"),
        F.col("oldbalanceDest").alias("destination_balance_before"),
        F.col("newbalanceDest").alias("destination_balance_after"),
        F.col("isFraud").alias("is_fraud"),
        F.col("isFlaggedFraud").alias("is_flagged_fraud"),
    )
