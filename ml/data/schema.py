from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

PAYSIM_SCHEMA = StructType(
    [
        StructField("step", IntegerType(), nullable=False),
        StructField("type", StringType(), nullable=False),
        StructField("amount", DoubleType(), nullable=False),
        StructField("nameOrig", StringType(), nullable=False),
        StructField("oldbalanceOrg", DoubleType(), nullable=False),
        StructField("newbalanceOrig", DoubleType(), nullable=False),
        StructField("nameDest", StringType(), nullable=False),
        StructField("oldbalanceDest", DoubleType(), nullable=False),
        StructField("newbalanceDest", DoubleType(), nullable=False),
        StructField("isFraud", IntegerType(), nullable=False),
        StructField("isFlaggedFraud", IntegerType(), nullable=False),
    ]
)
CANONICAL_TRANSACTION_SCHEMA = StructType(
    [
        StructField("transaction_id", StringType(), nullable=False),
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("transaction_type", StringType(), nullable=False),
        StructField("origin_account_id", StringType(), nullable=False),
        StructField("destination_account_id", StringType(), nullable=False),
        StructField("amount", DoubleType(), nullable=False),
        StructField("origin_balance_before", DoubleType(), nullable=False),
        StructField("origin_balance_after", DoubleType(), nullable=False),
        StructField("destination_balance_before", DoubleType(), nullable=False),
        StructField("destination_balance_after", DoubleType(), nullable=False),
        StructField("is_fraud", IntegerType(), nullable=False),
        StructField("is_flagged_fraud", IntegerType(), nullable=False),
    ]
)
