import json
from pathlib import Path
from typing import Annotated, Any, Dict

import typer

from src.benchmark_framework.configs.model_config import ModelConfig
from src.benchmark_framework.getters.get_llm_model import get_llm_model
from src.benchmark_framework.stats.calculate_stats import calculate_stats_for_path
from src.common.file_operations import FileOperations

app = typer.Typer()


def load_all_results(results_path: Path) -> list[Dict[str, Any]]:
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


def extract_evaluated_model_name(results: list[Dict[str, Any]]) -> str:
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


def generate_model_card(
    generator_model_name: str,
    results_path: Path,
    output_dir: Path,
):
    print(f"Calculating aggregated stats for: {results_path}")
    stats = calculate_stats_for_path(results_path)

    # Load all results and extract the evaluated model name
    results = load_all_results(results_path)
    evaluated_model_name = extract_evaluated_model_name(results)
    print(f"Evaluated model: {evaluated_model_name}")

    # Initialize the generator model with empty config
    model_config = ModelConfig()
    model = get_llm_model(generator_model_name, model_config)
    print(f"Using generator model: {generator_model_name}")

    # Prepare stats summary for the prompt
    stats_json = json.dumps(stats, indent=2, ensure_ascii=False)

    # Generate model card (empty prompt for now as requested)
    prompt = f"{stats_json}"
    system_prompt = "Generaate example model card markdown based on the following model evaluation stats:\n"
    response = model.generate_response(system_prompt, prompt)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = evaluated_model_name.replace("/", "-").replace("\\", "-")
    output_path = output_dir / f"{safe_model_name}.md"
    output_path.write_text(response, encoding="utf-8")


@app.command()
def main(
    model_name: Annotated[
        str,
        typer.Argument(help="Name of the LLM model to use for generating the model card"),
    ],
    results_path: Annotated[
        Path,
        typer.Argument(help="Path to directory containing test results with calculated metrics"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to save generated model cards"),
    ] = Path("data/model_cards"),
):
    if not model_name:
        raise typer.BadParameter("Model name must be provided")

    if not results_path:
        raise typer.BadParameter("Results path must be provided")

    if not results_path.exists():
        raise typer.BadParameter(f"Results path does not exist: {results_path}")

    generate_model_card(
        generator_model_name=model_name,
        results_path=results_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    app()

