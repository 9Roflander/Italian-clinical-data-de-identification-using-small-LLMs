# Italian Clinical Note De-identification with LLMs

This repository contains a comprehensive toolkit for de-identifying Italian clinical notes using Large Language Models (LLMs) and evaluating the de-identification performance through both deterministic methods and LLM-based evaluation.

## Overview

The project implements a GDPR-compliant de-identification system for Italian clinical notes using various LLMs. It includes:

1. **De-identification System**: Uses LLMs to identify and mask sensitive information in clinical notes
2. **Evaluation Framework**: Two evaluation approaches:
   - Deterministic evaluation based on exact matching
   - LLM-as-a-judge evaluation for more nuanced assessment
3. **Analysis Tools**: Comprehensive tools for analyzing and visualizing de-identification performance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Italian-Clinical-Note-Deidentification.git
cd Italian-Clinical-Note-Deidentification
```

2. Install dependencies (recommended inside a virtual environment):
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Set up Ollama (for local LLM inference):
```bash
# Follow instructions at https://ollama.ai to install Ollama
# Pull required models:
ollama pull gemma3:27b
ollama pull mistral-small:24b
ollama pull deepseek-r1:32b
```

## Usage

### 1. De-identification

The main de-identification script (`deid.py`) processes clinical notes using LLMs:

```bash
python deid.py --input input_notes.jsonl --output deidentified_notes.jsonl --model gemma3:27b --backend ollama
```

Key features:
- Supports multiple LLM backends (Ollama, VLLM)
- Configurable de-identification categories
- Handles various sensitive information types (names, dates, locations, etc.)
- Preserves medical information while removing PII

### 2. Evaluation

#### Deterministic Evaluation
```bash
python eval.py --input deidentified_notes.jsonl --gold_standard annotated_data.jsonl
```

#### LLM-as-a-Judge Evaluation
```bash
python llm_as_a_judge.py --deidentified_data_path deidentified_notes.jsonl --model gemma3:27b --category NOME
```

The LLM-as-a-judge evaluation:
- Evaluates de-identification quality using another LLM
- Categorizes results as True Positives (TP), False Positives (FP), or False Negatives (FN)
- Supports evaluation of specific categories (NOME, ETÀ, LUOGO/INDIRIZZO, DATA)

### 3. Analysis and Visualization

Generate performance metrics and visualizations:
```bash
python compute_metrics.py --input evaluation_results.jsonl
python create_boxplot.py --input metrics/
```

## Data Format

### Input Format (JSONL)
```json
{
    "input": "Original clinical note text",
    "output": "De-identified clinical note text"
}
```

### Evaluation Results Format
```json
{
    "report_id": "1",
    "annotations_gold": [
        {"text": "Mario Rossi", "type": "NOME"}
    ],
    "annotations_deidentified": [
        {"text": "[NOME]", "type": "NOME", "counted_as": "TP"}
    ]
}
```

## Project Structure

- `deid.py`: Main de-identification script
- `llm_as_a_judge.py`: LLM-based evaluation system
- `eval.py`: Deterministic evaluation script
- `compute_metrics.py`: Performance metrics computation
- `create_boxplot.py`: Visualization generation
- `plots.py`: Additional visualization utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your License Here]

## Citation

If you use this code in your research, please cite:
[Your Citation Information]