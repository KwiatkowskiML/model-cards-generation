# Polish Law LLM Benchmark

A benchmark framework for evaluating Large Language Models on Polish legal tasks, including bar exam questions and court judgment analysis.

## Project Structure

```
PolishLawLLM-Benchmark/
├── src/
│   ├── benchmark_framework/    # LLM benchmarking framework
│   ├── parsers/                # PDF parsing for exam data extraction
│   └── common/                 # Shared utilities and domain models
└── data/
    ├── pdfs/                   # Source PDF files (exams, legal codes)
    ├── corpuses/               # Extracted legal code articles (JSON)
    ├── tasks/                  # Benchmark tasks (JSONL)
    └── results/                # Benchmark results
```

## Quick Start

### Installation

```bash
git clone <repository-url>
cd PolishLawLLM-Benchmark
pip install -r requirements.txt
```

### Environment Variables

```bash
export GOOGLE_API_KEY="..."      # For Gemini models
export OPENAI_API_KEY="..."      # For GPT models
export ANTHROPIC_API_KEY="..."   # For Claude models
export NVIDIA_API_KEY="..."      # For NVIDIA-hosted models
export OPENROUTER_API_KEY="..."  # For OpenRouter-hosted models
export HF_TOKEN="..."            # For models hosted using Hugging Face Inference Endpoints
export HF_ENDPOINT_URL="..."     # Custom endpoint URL for Hugging Face Inference Endpoints
```

### Run the tests

```bash
python -m src.benchmark_framework.cli gpt-5.2 exams
```

---

## Modules

### Benchmark Framework

Run LLM evaluations, calculate metrics, and aggregate statistics.

```bash
# Run benchmark
python -m src.benchmark_framework.cli <model-name> <task-type>

# Calculate metrics on results
python -m src.benchmark_framework.calculate_metrics <input-dir> <output-dir>

# Get aggregate statistics
python -m src.benchmark_framework.stats.cli stats <file-path>
```

📖 **[Detailed documentation →](src/benchmark_framework/README.md)**

---
### Model Cards

Generate model cards based on the results of the tests.

```bash
# Run the model card generator
python -m src.model_cards.generate_model_cards <model-generating-the-card> <path-to-results>

# Generate Bielik's model card
python -m src.model_cards.generate_model_cards gemini-3-flash-preview data/results/speakleash-bielik-11b-v2.6-instruct/exams/
```

---

## Development

```bash
# Run tests
python -m pytest src/

# Code formatting
black src/
```
