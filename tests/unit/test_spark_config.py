import pytest

from ml.data.spark import resolve_spark_config


def test_spark_config_merges_safe_defaults():
    config = resolve_spark_config({"master": "local[2]", "shuffle_partitions": 4})

    assert config["master"] == "local[2]"
    assert config["shuffle_partitions"] == 4
    assert config["driver_memory"] == "4g"
    assert config["output_partitions"] == 4


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("shuffle_partitions", 0),
        ("default_parallelism", -1),
        ("output_partitions", True),
    ],
)
def test_spark_config_rejects_invalid_parallelism(key, value):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_spark_config({key: value})


def test_spark_config_rejects_invalid_persistence_flag():
    with pytest.raises(ValueError, match="persist_intermediates"):
        resolve_spark_config({"persist_intermediates": "true"})
