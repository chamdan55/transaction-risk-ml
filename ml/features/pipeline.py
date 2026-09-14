from pyspark.sql import DataFrame

from ml.features.amount import add_amount_features
from ml.features.balance import add_balance_features
from ml.features.behavior import add_behavior_features
from ml.features.schema import FEATURE_SCHEMA
from ml.features.timestamp import add_timestamp_features
from ml.features.transaction import add_transaction_features


def build_features(df: DataFrame) -> DataFrame:
    """Build the Step 5 feature dataset from canonical transactions."""
    df = add_amount_features(df)
    df = add_balance_features(df)
    df = add_transaction_features(df)
    df = add_timestamp_features(df)
    df = add_behavior_features(df)
    return df.select(*FEATURE_SCHEMA.names)
