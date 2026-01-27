from pathlib import Path
from typing import Annotated
import typer
from src.model_cards.generator import generate_model_card

app = typer.Typer()

@app.command()
def main(
    model_name: Annotated[
        str,
        typer.Argument(
            help="Name of the LLM model to use for generating the model card"
        ),
    ],
    results_path: Annotated[
        Path,
        typer.Argument(
            help="Path to directory containing test results with calculated metrics"
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to save generated model cards"),
    ] = Path("data/model_cards"),
):
    """
    Generates a Markdown model card based on benchmark results using an LLM.
    """
    if not model_name:
        raise typer.BadParameter("Model name must be provided")

    if not results_path:
        raise typer.BadParameter("Results path must be provided")

    if not results_path.exists():
        raise typer.BadParameter(f"Results path does not exist: {results_path}")

    try:
        generate_model_card(
            generator_model_name=model_name,
            results_path=results_path,
            output_dir=output_dir,
        )
    except Exception as e:
        typer.echo(f"Error generating model card: {e}", err=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()