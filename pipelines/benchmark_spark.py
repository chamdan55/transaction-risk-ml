"""Benchmark exact versus scalable Spark split strategies on a feature dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from app.core.logging import setup_logging
from ml.data.spark import create_spark_session
from ml.data.split import chronological_split


def benchmark_split_strategies(
    *,
    features_path: str | Path,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    quantile_relative_error: float,
    spark_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure split strategies and return JSON-serializable evidence."""

    spark = create_spark_session(spark_config)
    try:
        dataframe = spark.read.parquet(str(features_path))
        row_count = dataframe.count()
        results: dict[str, Any] = {
            "features_path": str(features_path),
            "row_count": row_count,
            "strategies": {},
        }
        for strategy in ("exact", "time_boundary"):
            started = time.perf_counter()
            splits = chronological_split(
                dataframe,
                train_ratio,
                validation_ratio,
                test_ratio,
                strategy=strategy,
                quantile_relative_error=quantile_relative_error,
            )
            split_counts = {name: split.count() for name, split in splits.items()}
            elapsed = time.perf_counter() - started
            results["strategies"][strategy] = {
                "elapsed_seconds": round(elapsed, 6),
                "split_counts": split_counts,
                "physical_plan": splits["train"]._jdf.queryExecution().simpleString(),
            }
        return results
    finally:
        spark.stop()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--features-path", default=None)
    parser.add_argument("--output", default="artifacts/spark_split_benchmark.json")
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = _parse_args()
    with Path(args.config).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    split_config = config["split"]
    features_path = args.features_path or f"{config['data']['processed_path']}/features/audit"
    result = benchmark_split_strategies(
        features_path=features_path,
        train_ratio=split_config["train_ratio"],
        validation_ratio=split_config["validation_ratio"],
        test_ratio=split_config["test_ratio"],
        quantile_relative_error=float(split_config.get("quantile_relative_error", 0.01)),
        spark_config=config.get("spark"),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
