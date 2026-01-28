# Automated model cards generation for law exams

Functionality for generating model cards based on the results of the tests.

## Project Structure

```
├── src/
│   ├── benchmark_framework/    # LLM models inference functionalities
│   ├── model_cards             # Functionality for generating model cards
│   └── common/                 # Shared utilities and domain models
└── data/
    ├── model_cards/            # Generated model cards (Markdown)
    ├── corpuses/               # Extracted legal code articles (JSON)
    ├── tasks/                  # Benchmark tasks (JSONL)
    └── results/                # Benchmark results
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
export GOOGLE_API_KEY="..."      # For Gemini models
export OPENAI_API_KEY="..."      # For GPT models
export ANTHROPIC_API_KEY="..."   # For Claude models
export OPENROUTER_API_KEY="..."  # For OpenRouter-hosted models
export HF_TOKEN="..."            # For models hosted using Hugging Face Inference Endpoints
export HF_ENDPOINT_URL="..."     # Custom endpoint URL for Hugging Face Inference Endpoints
```
---

### Generate model cards

Generate model cards based on the results of the tests.

```bash
# Run the model card generator
python -m src.model_cards.generate_model_cards <model-name> <results-path>

# Generate Bielik's model card
python -m src.model_cards.generate_model_cards gemini-3-flash-preview data/results_with_metrics/speakleash-bielik-11b-v2.6-instruct/exams/
```

#### Arguments

| Argument                                                                                                               | Description                                                                                                                       |
|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| `model-name`                                                                                                           | Model identifier of the model which will generate the cards (e.g., `gemini-3-flash-preview`, `gpt-5.2`)                           |
| `results-path` | Path to the directory with results of the model based on which the model cards will be generated. The metrics must be calculated. |
---


### Run the tests

Run LLM evaluations and calculate metrics.

```bash
# Run benchmark
python -m src.benchmark_framework.cli <model-name> [output-path] [input-path] [options]
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `model-name` | Model identifier (e.g., `gemini-3-flash-preview`, `gpt-5.2`) |
| `output-path` | Output directory for results (default: `data/results`) |
| `input-path` | Input directory with task files (default: `data/tasks`) |

#### Options

| Option | Description |
|--------|-------------|
| `--google-search` | Enable Google Search grounding (Gemini only) |
| `--year`, `-y` | Filter tasks to a specific year (e.g., `2024`) |


```bash
# Calculate metrics on results
python -m src.benchmark_framework.calculate_metrics <input-dir> <output-dir>
```


#### Arguments

| Argument | Description |
|----------|-------------|
| `input-dir` | Directory containing result JSONL files |
| `output-dir` | Directory for processed files with metrics |
| `corpuses-dir` | Corpuses directory for TF-IDF (default: `data/corpuses`) |

---

## Development

```bash
# Run tests
python -m pytest src/

# Code formatting
black src/
```
