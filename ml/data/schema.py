from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
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
