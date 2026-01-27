import json
from typing import Any, Dict, List

SYSTEM_PROMPT = """
Jesteś ekspertem ds. ewaluacji modeli językowych (LLM) w domenie prawa polskiego.
Twoim zadaniem jest wygenerowanie profesjonalnego **Model Card** (Karty Modelu) w formacie Markdown.
Analiza dotyczy benchmarku opartego na **pytaniach z egzaminów wstępnych na aplikacje prawnicze** (adwokacką, radcowską, notarialną, komorniczą) z lat 2016-2025.

### METODOLOGIA I METRYKI (Kontekst):
Benchmark sprawdza trzy kluczowe aspekty:
1.  **DO (Dokładność Odpowiedzi)**: Czy model wybrał poprawną odpowiedź (A/B/C)?
2.  **DOP (Dokładność Oznaczenia Przepisu)**: Czy model wskazał poprawną podstawę prawną (np. właściwy artykuł i paragraf Kodeksu Karnego)? Jest to kluczowe dla oceny, czy model nie "zgaduje".
3.  **Jakość cytowania treści (Metryki Tekstowe)**:
    * **ROUGE-W F1**: Nagradza ciągłe sekwencje słów (ważne przy dosłownym cytowaniu prawa).
    * **ROUGE-N TF-IDF**: Ważona metryka, która karze model mocniej za pominięcie słów kluczowych (rzadkich w korpusie), takich jak konkretne liczby, daty czy nazwy własne.
    * **ROUGE-N F1**: Średnia miar F1 dla n-gramów o różnych długościach, od 1 do 3.

### STRUKTURA RAPORTU (Output):
Wygeneruj raport w Markdown zawierający dokładnie te sekcje:

1.  **Nagłówek**: Nazwa modelu.
2.  **Podsumowanie Wykonawcze**: Krótki, jednoakapitowy opis, czy model nadaje się do zastosowań prawniczych.
3.  **Wyniki Testów (Tabela)**: Tabela zawierająca kolumny: Metryka, Wartość, Interpretacja. Uwzględnij DO, DOP, DTP oraz metryki ROUGE.
4.  **Analiza Mocnych Stron**:
    * Czy model ma wysoką korelację między poprawną odpowiedzią a znajomością przepisu?
    * Czy dobrze radzi sobie z cytowaniem (wysokie ROUGE)?
5.  **Analiza Słabych Stron i Halucynacji**:
    * **Rozdźwięk DO vs DOP**: Jeśli DO jest wysokie, a DOP niskie, oznacza to, że model posiada wiedzę ogólną, ale nie zna konkretnych źródeł prawa.
    * **Jakość cytowania**: Czy model zmyśla treść przepisów?
6.  **Analiza Techniczna i Stabilność**:
    * **Skala problemu**: Musisz podać procent odpowiedzi, których nie udało się sparsować.
    * **Typologia błędów**: Na podstawie dostarczonych przykładów niesparsowanych odpowiedzi zidentyfikuj konkretne problemy.
7.  **Wnioski i Rekomendacja**: Czy model może służyć jako asystent prawny?

Pamiętaj o profesjonalnym, analitycznym tonie. Używaj pogrubień dla kluczowych wniosków.
"""


def prepare_analysis_prompt(
        model_name: str,
        entries: List[Dict[str, Any]],
        stats: Dict[str, Any]
) -> str:
    """Prepares the user prompt with statistics and specific examples."""

    # Filter examples for the prompt
    successes = [
        e for e in entries
        if e.get("accuracy_metrics").get("answer") == 1.0
           and e.get("accuracy_metrics").get("legal_basis") == 1.0
    ][:10]

    wrong_answers = [
        e for e in entries if e.get("accuracy_metrics").get("answer") == 0.0
    ][:10]

    hallucinations_legal = [
        e for e in entries
        if e.get("accuracy_metrics").get("answer") == 1.0
           and e.get("accuracy_metrics").get("legal_basis") == 0.0
    ][:10]

    hallucinations_content = [
        e for e in entries
        if e.get("accuracy_metrics").get("legal_basis") == 1.0
           and e.get("text_metrics").get("rouge_n_f1") < 0.5
    ][:10]

    malformed_responses = [
        e for e in entries if len(e.get("model_legal_basis_content")) == 0
    ]

    prompt = f"""
    Przeanalizuj wyniki modelu **{model_name}** w teście znajomości prawa polskiego.

    ### WYNIKI LICZBOWE (Do umieszczenia w tabeli):
    1. **Dokładność Odpowiedzi (DO)**: {stats['accuracy_metrics']['answer']:.2%}
    2. **Dokładność Oznaczenia Przepisu (DOP)**: {stats['accuracy_metrics']['legal_basis']:.2%}
    3. **Dokładność Treści Przepisu (DTP - Exact Match)**: {stats['text_metrics']['exact_match']:.2%}
    4. **ROUGE-N F1 (Średnia)**: {stats['text_metrics']['rouge_n_f1']}
    5. **ROUGE-N TF-IDF Recall**: {stats['text_metrics']['rouge_n_tfidf']}
    6. **ROUGE-W F1**: {stats['text_metrics']['rouge_w']}
    7. **Odsetek przypadków niemożliwych do sparsowania**: {stats['malformed_response_rate']:.2%}

    ### DANE JAKOŚCIOWE DO ANALIZY:

    **PRZYKŁADY SUKCESÓW** (Idealne działanie):
    {json.dumps(successes, ensure_ascii=False, indent=2)}

    **PRZYKŁADY BŁĘDNYCH ODPOWIEDZI**:
    {json.dumps(wrong_answers, ensure_ascii=False, indent=2)}

    **HALUCYNACJE PODSTAWY PRAWNEJ** (Poprawna odpowiedź, ale błędny przepis):
    {json.dumps(hallucinations_legal, ensure_ascii=False, indent=2)}

    **ZMYŚLANIE TREŚCI** (Poprawny numer artykułu, ale treść niezgodna):
    {json.dumps(hallucinations_content, ensure_ascii=False, indent=2)}

    **PROBLEMY Z FORMATEM**:
    {json.dumps(malformed_responses, ensure_ascii=False, indent=2)}

    Stwórz raport zgodnie z instrukcjami w System Prompcie.
    """
    return prompt