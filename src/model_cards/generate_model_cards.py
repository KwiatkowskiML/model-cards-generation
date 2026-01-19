import json
from pathlib import Path
from typing import Annotated, Any, Dict, List

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


def prepare_analysis_prompt(
    model_name: str, entries: List[Dict[str, Any]], results_path: Path
) -> str:
    successes = [
        e
        for e in entries
        if e.get("accuracy_metrics").get("answer") == 1.0
        and e.get("accuracy_metrics").get("legal_basis") == 1.0
    ][:10]

    wrong_answers = [
        e for e in entries if e.get("accuracy_metrics").get("answer") == 0.0
    ][:10]

    hallucinations_legal = [
        e
        for e in entries
        if e.get("accuracy_metrics").get("answer") == 1.0
        and e.get("accuracy_metrics").get("legal_basis") == 0.0
    ][:10]

    hallucinations_content = [
        e
        for e in entries
        if e.get("accuracy_metrics").get("legal_basis") == 1.0
        and e.get("text_metrics").get("rouge_n_f1") < 0.5
    ][:10]

    malformed_responses = [
        e for e in entries if len(e.get("model_legal_basis_content")) == 0
    ]

    stats = calculate_stats_for_path(results_path)

    prompt = f"""
    Przeanalizuj wyniki modelu **{model_name}** w teście znajomości prawa polskiego (egzaminy wstępne na aplikacje prawnicze).

    ### WYNIKI LICZBOWE (Do umieszczenia w tabeli):
    Poniższe dane musisz przedstawić w czytelnej tabeli Markdown w sekcji "Wyniki Benchmarku":

    1. **Dokładność Odpowiedzi (DO)**: {stats['accuracy_metrics']['answer']:.2%}
       *Zdolność wyboru poprawnej odpowiedzi (A/B/C).*
    2. **Dokładność Oznaczenia Przepisu (DOP)**: {stats['accuracy_metrics']['legal_basis']:.2%}
       *Zdolność wskazania poprawnego artykułu (np. "art. 148 k.k."). Wysoka wartość oznacza dobrą znajomość struktury kodeksów.*
    3. **Dokładność Treści Przepisu (DTP - Exact Match)**: {stats['text_metrics']['exact_match']:.2%}
       *Odsetek przypadków, gdzie model zacytował przepis idealnie (znak w znak).*
    4. **ROUGE-N F1 (Średnia)**: {stats['text_metrics']['rouge_n_f1']}
       *Ogólna jakość odtworzenia treści przepisu (n-gramy 1-3).*
    5. **ROUGE-N TF-IDF Recall**: {stats['text_metrics']['rouge_n_tfidf']}
       *Czułość ważona TF-IDF. Mierzy, czy model uwzględnił słowa kluczowe (rzadkie/ważne) dla danego przepisu.*
    6. **ROUGE-W F1**: {stats['text_metrics']['rouge_w']}
       *Miara uwzględniająca ciągłość sekwencji (Weighted LCS). Ważna przy cytowaniu długich fragmentów.*
    7. **Odsetek przypadków niemożliwych do sparsowania**: {stats['malformed_response_rate']:.2%}

    ### DANE JAKOŚCIOWE DO ANALIZY:

    **PRZYKŁADY SUKCESÓW** (Idealne działanie modelu):
    {json.dumps(successes, ensure_ascii=False, indent=2)}

    **PRZYKŁADY BŁĘDNYCH ODPOWIEDZI** (Zły wybór odpowiedzi A/B/C):
    {json.dumps(wrong_answers, ensure_ascii=False, indent=2)}

    **HALUCYNACJE PODSTAWY PRAWNEJ** (Poprawna odpowiedź, ale błędny przepis):
    *Analiza tego punktu pozwala ocenić, czy model "zgaduje", czy faktycznie zna źródło prawa.*
    {json.dumps(hallucinations_legal, ensure_ascii=False, indent=2)}

    **ZMYŚLANIE TREŚCI** (Poprawny numer artykułu, ale treść niezgodna z oryginałem):
    *Sprawdź, czy model parafrazuje, czy całkowicie zmyśla treść przepisu.*
    {json.dumps(hallucinations_content, ensure_ascii=False, indent=2)}

    **PROBLEMY Z FORMATEM** (Odpowiedzi niesparsowane/puste):
    {json.dumps(malformed_responses, ensure_ascii=False, indent=2)}

    Stwórz raport zgodnie z instrukcjami w System Prompcie.
    """
    return prompt


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

    prompt = prepare_analysis_prompt(evaluated_model_name, results, results_path)
    system_prompt = """
    Jesteś ekspertem ds. ewaluacji modeli językowych (LLM) w domenie prawa polskiego.
    Twoim zadaniem jest wygenerowanie profesjonalnego **Model Card** (Karty Modelu) w formacie Markdown.
    Analiza dotyczy benchmarku opartego na **pytaniach z egzaminów wstępnych na aplikacje prawnicze** (adwokacką, radcowską, notarialną, komorniczą) z lat 2016-2025.

    ### METODOLOGIA I METRYKI (Kontekst):
    Benchmark sprawdza trzy kluczowe aspekty:
    1.  **DO (Dokładność Odpowiedzi)**: Czy model wybrał poprawną odpowiedź (A/B/C)?
    2.  **DOP (Dokładność Oznaczenia Przepisu)**: Czy model wskazał poprawną podstawę prawną (np. właściwy artykuł i paragraf Kodeksu Karnego)? Jest to kluczowe dla oceny, czy model nie "zgaduje".
    3.  **Jakość cytowania treści (Metryki Tekstowe)**:
        * **ROUGE-W F1**: Nagradza ciągłe sekwencje słów (ważne przy dosłownym cytowaniu prawa).
        * **ROUGE-N TF-IDF**: Ważona metryka, która karze model mocniej za pominięcie słów kluczowych (rzadkich w korpusie), takich jak konkretne liczby, daty czy nazwy własne, a mniej za pominięcie słów pospolitych.
        * **ROUGE-N F1**: średnia miar F1 dla n-gramów o róznych długościach, od 1 do 3.

    ### STRUKTURA RAPORTU (Output):
    Wygeneruj raport w Markdown zawierający dokładnie te sekcje:

    1.  **Nagłówek**: Nazwa modelu.
    2.  **Podsumowanie Wykonawcze**: Krótki, jednoakapitowy opis, czy model nadaje się do zastosowań prawniczych.
    3.  **Wyniki Testóws (Tabela)**: Tabela zawierająca kolumny: Metryka, Wartość, Interpretacja. Uwzględnij DO, DOP, DTP oraz metryki ROUGE.
    4.  **Analiza Mocnych Stron**:
        * Czy model ma wysoką korelację między poprawną odpowiedzią a znajomością przepisu?
        * Czy dobrze radzi sobie z cytowaniem (wysokie ROUGE)?
    5.  **Analiza Słabych Stron i Halucynacji**:
        * **Rozdźwięk DO vs DOP**: Jeśli DO jest wysokie, a DOP niskie, oznacza to, że model posiada wiedzę ogólną, ale nie zna konkretnych źródeł prawa (halucynuje podstawy).
        * **Jakość cytowania**: Czy model zmyśla treść przepisów (niskie TF-IDF Recall)? Czy parafrazuje zamiast cytować?
        * Zwróć uwagę na typowe błędy: mylenie numeracji artykułów, wymyślanie nieistniejących paragrafów.
    6.  **Analiza Techniczna i Stabilność**:
        * **Skala problemu**: Musisz podać procent odpowiedzi, których nie udało się sparsować.
        * **Typologia błędów**: Na podstawie dostarczonych przykładów niesparsowanych odpowiedzi ( **PROBLEMY Z FORMATEM**), zidentyfikuj i opisz konkretne problemy, które występują w dostarczonych przykładach.
    7.  **Wnioski i Rekomendacja**: Czy model może służyć jako asystent prawny?

    Pamiętaj o profesjonalnym, analitycznym tonie. Używaj pogrubień dla kluczowych wniosków.
    """

    response = model.generate_response(system_prompt, prompt)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = evaluated_model_name.replace("/", "-").replace("\\", "-")
    output_path = output_dir / f"{safe_model_name}.md"
    output_path.write_text(response, encoding="utf-8")


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
