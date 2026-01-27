from pathlib import Path
from src.benchmark_framework.configs.model_config import ModelConfig
from src.benchmark_framework.getters.get_llm_model import get_llm_model
from src.benchmark_framework.stats.calculate_stats import calculate_stats_for_path

from src.model_cards.data_loader import load_all_results, extract_evaluated_model_name
from src.model_cards.prompts import prepare_analysis_prompt, SYSTEM_PROMPT


def generate_model_card(
        generator_model_name: str,
        results_path: Path,
        output_dir: Path,
):
    """
    Orchestrates the generation of a model card.

    1. Calculates statistics.
    2. Loads evaluation data.
    3. Initializes the generator LLM.
    4. Creates the prompt.
    5. Generates and saves the report.
    """
    print(f"Calculating aggregated stats for: {results_path}")
    stats = calculate_stats_for_path(results_path)

    # Load all results and extract the evaluated model name
    results = load_all_results(results_path)
    evaluated_model_name = extract_evaluated_model_name(results)
    print(f"Evaluated model: {evaluated_model_name}")

    # Initialize the generator model
    model_config = ModelConfig()
    model = get_llm_model(generator_model_name, model_config)
    print(f"Using generator model: {generator_model_name}")

    # Prepare prompt
    prompt = prepare_analysis_prompt(evaluated_model_name, results, stats)

    # Generate response
    response = model.generate_response(SYSTEM_PROMPT, prompt)

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = evaluated_model_name.replace("/", "-").replace("\\", "-")
    output_path = output_dir / f"{safe_model_name}.md"
    output_path.write_text(response, encoding="utf-8")

    print(f"Model card saved to: {output_path}")