from pathlib import Path
from typing import Any, Dict, List
from src.common.file_operations import FileOperations


def load_all_results(results_path: Path) -> List[Dict[str, Any]]:
    """Loads all JSONL files from a directory."""
    jsonl_files = list(results_path.rglob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {results_path}")

    all_results = []
    for jsonl_file in jsonl_files:
        data = FileOperations.load_jsonl(jsonl_file)
        if not data:
            raise ValueError(f"Empty JSONL file: {jsonl_file}")
        all_results.extend(data)

    return all_results


def extract_evaluated_model_name(results: List[Dict[str, Any]]) -> str:
    """Extracts and validates the model name from the results."""
    if not results:
        raise ValueError("No results provided")

    model_name = None
    for entry in results:
        entry_model_name = entry.get("model_name")
        if not entry_model_name:
            raise ValueError(f"No 'model_name' field in entry: {entry}")

        if model_name is None:
            model_name = entry_model_name
        elif model_name != entry_model_name:
            raise ValueError(
                f"Inconsistent model names: '{model_name}' vs '{entry_model_name}'"
            )

    return model_name
