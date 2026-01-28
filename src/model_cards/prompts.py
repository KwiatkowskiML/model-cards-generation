import json
from typing import Any, Dict, List

SYSTEM_PROMPT = """
Jesteś analitykiem systemów AI dedykowanych dla branży prawniczej (LegalTech).
Twój cel: Stworzyć **Model Card (Kartę Modelu)**, która pomoże prawnikowi podjąć decyzję, czy może zaufać temu modelowi w swojej pracy. Analiza dotyczy wyników modeli w benchmarku opartym na **pytaniach z egzaminów wstępnych na aplikacje prawnicze** (adwokacką_radcowską, notarialną, komorniczą) z lat 2016-2025.

Odbiorca: Prawnik, który nie zna się na ML, ale musi wiedzieć, czy model nie wprowadzi go w błąd (nie "zhalucynuje" nieistniejącego przepisu).

METODOLOGIA I METRYKI (Kontekst):
Benchmark sprawdza trzy kluczowe aspekty:
1.  **DO (Dokładność Odpowiedzi)**: Czy model wybrał poprawną odpowiedź (A/B/C)?
2.  **DOP (Dokładność Oznaczenia Przepisu)**: Czy model wskazał poprawną podstawę prawną (np. właściwy artykuł i paragraf Kodeksu Karnego)? Jest to kluczowe dla oceny, czy model nie "zgaduje".
3.  **Jakość cytowania treści (Metryki Tekstowe)**:
    * **ROUGE-W F1**: Nagradza ciągłe sekwencje słów (ważne przy dosłownym cytowaniu prawa).
    * **ROUGE-N TF-IDF**: Ważona metryka, która karze model mocniej za pominięcie słów kluczowych (rzadkich w korpusie), takich jak konkretne liczby, daty czy nazwy własne.
    * **ROUGE-N F1**: Średnia miar F1 dla n-gramów o różnych długościach, od 1 do 3.

STRUKTURA RAPORTU:
 **Nazwa modelu**.
1.  **Ocena Przydatności Prawniczej (Werdykt)**:
    * Jasna deklaracja: ZALECANY / OGRANICZONE ZAUFANIE / NIEBEZPIECZNY.
    * Główne ryzyka (np. "Model często cytuje nieistniejące artykuły").
2.  **Wyniki Testów (Tabela)**: Tabela zawierająca kolumny: Metryka, Wartość, Interpretacja. Uwzględnij DO, DOP, DTP oraz metryki ROUGE.
4.  **Szczegółowa Analiza per Egzamin**:
    * Jak model radzi sobie z różnymi typami egzaminów (Komorniczy vs Adwokacki_Radcowy vs notarialny)?
    * Czy widać specjalizację w konkretnej dziedzinie?
5.  **Analiza Mocnych Stron**:
    * Czy model ma wysoką korelację między poprawną odpowiedzią a znajomością przepisu?
    * Czy dobrze radzi sobie z cytowaniem (wysokie ROUGE)?
6.  **Analiza Słabych Stron i Halucynacji**:
    * Rozdźwięk DO vs DOP: Jeśli DO jest wysokie, a DOP niskie, oznacza to, że model posiada wiedzę ogólną, ale nie zna konkretnych źródeł prawa.
    * Jakość cytowania: Czy model zmyśla treść przepisów?
7.  **Analiza Błędów Krytycznych (Edge Cases)**:
    * Omów przypadki niespójności (poprawny przepis, błędna odpowiedź).
    * Omów jakość języka (pętle, wtrącenia obcojęzyczne).
6.  **Analiza Techniczna i Stabilność**:
    * Skala problemu: Musisz podać procent odpowiedzi, których nie udało się sparsować.
    * Typologia błędów: Na podstawie dostarczonych przykładów niesparsowanych odpowiedzi zidentyfikuj konkretne problemy.
8.  **Rekomendacja użycia**:
    * Do czego używać (np. "Wyszukiwanie ogólnych idei"), a do czego absolutnie nie (np. "Cytowanie konkretnych paragrafów").

STYL:
Konkretny, ostrzegawczy w przypadku wykrycia halucynacji, profesjonalny. Pamiętaj o profesjonalnym, analitycznym tonie. Używaj pogrubień dla kluczowych wniosków.
"""


def prepare_analysis_prompt(
    model_name: str, entries: List[Dict[str, Any]], stats: Dict[str, Any]
) -> str:
    exam_performance = {}
    for entry in entries:
        e_type = entry.get("exam_type", "unknown")
        if e_type not in exam_performance:
            exam_performance[e_type] = {"correct": 0, "total": 0}
        exam_performance[e_type]["total"] += 1
        if entry.get("accuracy_metrics", {}).get("answer") == 1.0:
            exam_performance[e_type]["correct"] += 1

    exam_summary_str = "\n".join(
        [
            f"- **{etype.capitalize()}**: {data['correct'] / data['total']:.2%} (próba: {data['total']})"
            for etype, data in exam_performance.items()
        ]
    )

    print(exam_summary_str)

    # Model zna przepis (DOP=1, ROUGE wysokie), ale zaznacza złą odpowiedź (DO=0).
    inconsistent_logic = [
        e
        for e in entries
        if e.get("accuracy_metrics", {}).get("answer") == 0.0
        and e.get("accuracy_metrics", {}).get("legal_basis") == 1.0
        and e.get("text_metrics", {}).get("rouge_n_f1", 0) > 0.9
    ][:3]

    anomalies = [
        e
        for e in entries
        if "json"
        in e.get("model_legal_basis_content", "").lower()  # Wyciek formatowania
    ][:3]

    # Filter examples for the prompt
    successes = [
        e
        for e in entries
        if e.get("accuracy_metrics").get("answer") == 1.0
        and e.get("accuracy_metrics").get("legal_basis") == 1.0
    ][:3]

    wrong_answers = [
        e for e in entries if e.get("accuracy_metrics").get("answer") == 0.0
    ][:3]

    hallucinations_legal = [
        e
        for e in entries
        if e.get("accuracy_metrics").get("answer") == 1.0
        and e.get("accuracy_metrics").get("legal_basis") == 0.0
    ][:3]

    hallucinations_content = [
        e
        for e in entries
        if e.get("accuracy_metrics").get("legal_basis") == 1.0
        and e.get("text_metrics").get("rouge_n_f1") < 0.5
    ][:3]

    malformed_responses = [
        e for e in entries if len(e.get("model_legal_basis_content")) == 0
    ]

    prompt = f"""
Przeanalizuj wyniki modelu **{model_name}**.

### 1. WYNIKI LICZBOWE (Do umieszczenia w tabeli):
1. **Dokładność Odpowiedzi (DO)**: {stats['accuracy_metrics']['answer']:.2%}
2. **Dokładność Oznaczenia Przepisu (DOP)**: {stats['accuracy_metrics']['legal_basis']:.2%}
3. **Dokładność Treści Przepisu (DTP - Exact Match)**: {stats['text_metrics']['exact_match']:.2%}
4. **ROUGE-N F1 (Średnia)**: {stats['text_metrics']['rouge_n_f1']}
5. **ROUGE-N TF-IDF Recall**: {stats['text_metrics']['rouge_n_tfidf']}
6. **ROUGE-W F1**: {stats['text_metrics']['rouge_w']}
7. **Odsetek przypadków niemożliwych do sparsowania**: {stats['malformed_response_rate']:.2%}

### 2. WYNIKI WG TYPU EGZAMINU:
{exam_summary_str}

### 3. PRZYKŁADY POPRAWNYCH I BŁĘDNYCH ODPOWIEDZI:
**PRZYKŁADY SUKCESÓW** (Idealne działanie):
{json.dumps(successes, ensure_ascii=False, indent=2)}

**PRZYKŁADY BŁĘDNYCH ODPOWIEDZI**:
{json.dumps(wrong_answers, ensure_ascii=False, indent=2)}

### 4. ANALIZA RYZYKA (Szczególne przypadki):

**NIESPÓJNOŚĆ LOGICZNA (Wysokie Ryzyko):**
Poniższe przykłady pokazują sytuacje, gdzie model znał przepis, ale udzielił BŁĘDNEJ porady.
Dla prawnika to sygnał, że model nie potrafi wnioskować na podstawie wiedzy.
{json.dumps(inconsistent_logic, ensure_ascii=False, indent=2)}

**ANOMALIE TECHNICZNE (Pętle, Artefakty):**
Przykłady, gdzie model "zgubił się" technicznie.
{json.dumps(anomalies, ensure_ascii=False, indent=2)}

**HALUCYNACJE PODSTAWY PRAWNEJ** (Poprawna odpowiedź, ale błędny przepis):
{json.dumps(hallucinations_legal, ensure_ascii=False, indent=2)}

**PROBLEMY Z FORMATEM**:
{json.dumps(malformed_responses, ensure_ascii=False, indent=2)}

**ZMYŚLANIE TREŚCI** (Poprawny numer artykułu, ale treść niezgodna):
{json.dumps(hallucinations_content, ensure_ascii=False, indent=2)}

Stwórz raport skupiając się na bezpieczeństwie użycia tego modelu w kancelarii prawnej.
    """
    return prompt
