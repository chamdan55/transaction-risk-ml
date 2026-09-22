import ast
from pathlib import Path


def test_model_deserialization_import_path_has_no_top_level_pyspark_dependency() -> None:
    """A Joblib model imports this module's type at serving startup."""

    source_path = Path("ml/training/imbalance.py")
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    top_level_imports = [
        node for node in module.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    assert all(
        not (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("pyspark")
        )
        for node in top_level_imports
    )
