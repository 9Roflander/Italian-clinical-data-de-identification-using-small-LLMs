# Named Entity Recognition and Deidentification Analysis for Italian Clinical Notes

This repository contains tools for performing Named Entity Recognition (NER) on Italian clinical notes and analyzing the deidentification process. The tools work with JSONL files containing both "input" (original) and "output" (deidentified) text fields.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Tools and Scripts](#tools-and-scripts)
  - [Basic NER](#1-basic-ner-nerpy)
  - [Medical NER](#2-medical-ner-medical_nerpy)
  - [Combined NER Analysis](#3-combined-ner-analysis-run_ner_analysispy)
  - [Deidentification Report Generator](#4-deidentification-report-generator-generate_deidentification_reportpy)
  - [Visualization Tools](#5-visualization-tools-visualize_deidentificationpy)
- [Complete Workflow Example](#complete-workflow-example)
- [Input Format](#input-format)
- [Output Format](#output-format)
- [Testing](#testing)
- [Extending the Tools](#extending-the-tools)

## Overview

This toolkit provides a comprehensive solution for analyzing the effectiveness of deidentification in Italian clinical notes. It helps identify named entities in both original and deidentified text, determine which entities have been removed, and generate detailed reports and visualizations to assess the deidentification process.

Key capabilities:
- Identify standard named entities (people, locations, organizations) in Italian text
- Recognize medical-specific terminology and measurements
- Compare original and deidentified text to detect removed entities
- Generate statistics on deidentification effectiveness
- Visualize entity removal patterns

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

The tools require a spaCy Italian language model:

```bash
python -m spacy download it_core_news_lg
```

For smaller/faster processing, you can use the smaller model instead:

```bash
python -m spacy download it_core_news_sm
```

## Tools and Scripts

### 1. Basic NER (`ner.py`)

This script performs standard Named Entity Recognition using spaCy's Italian language model and compares entities between original and deidentified text.

**Features:**
- Uses spaCy's Italian language model for general entity recognition
- Identifies entities like people, organizations, locations, dates, etc.
- Compares input and output entities to analyze deidentification patterns
- Calculates deidentification statistics

**Usage:**
```bash
python ner.py input.jsonl [--output_file output_ner.jsonl] [--model it_core_news_lg]
```

### 2. Medical NER (`medical_ner.py`)

This script enhances spaCy's general NER with specialized pattern matching for medical terminology in Italian.

**Features:**
- Specialized for medical terminology in Italian
- Uses pattern matching and dictionaries to identify medical entities
- Recognizes 8 types of medical entities:
  - CONDITION: Diseases, disorders, symptoms (e.g., "diabete", "infarto")
  - MEDICATION: Drugs, medications (e.g., "antibiotico", "insulina") 
  - PROCEDURE: Medical procedures, surgeries (e.g., "intervento", "biopsia")
  - TEST: Laboratory tests, diagnostics (e.g., "analisi", "radiografia")
  - ANATOMY: Body parts, organs (e.g., "cuore", "fegato")
  - MEASUREMENT: Measurements, test results (e.g., "120/80 mmHg", "5.4 mg/dL")
  - DEVICE: Medical devices, equipment (e.g., "pacemaker", "catetere")
  - TIME: Time expressions related to medical events (e.g., "cronico", "acuto")

**Usage:**
```bash
python medical_ner.py input.jsonl [--output_file output_med_ner.jsonl] [--model it_core_news_lg]
```

### 3. Combined NER Analysis (`run_ner_analysis.py`)

This script runs both general and medical NER and combines the results into a single output file.

**Features:**
- Runs both `ner.py` and `medical_ner.py` scripts sequentially
- Combines entities from both scripts with intelligent merging
- Handles entity overlaps with priority for longer matches and medical entities
- Preserves original entity source information
- Cleans up temporary files automatically

**Usage:**
```bash
python run_ner_analysis.py input.jsonl [--output_file output_combined_ner.jsonl] [--model it_core_news_lg] [--keep_temp]
```

### 4. Deidentification Report Generator (`generate_deidentification_report.py`)

This script analyzes the NER results and generates a detailed Markdown report on the deidentification process.

**Features:**
- Generates a comprehensive Markdown report of the deidentification process
- Highlights removed entities and newly added placeholders
- Provides statistics on deidentification rate
- Identifies the most common removed entities
- Groups removed entities by type (PER, LOC, ORG, etc.)
- Includes sample records with their deidentification details

**Usage:**
```bash
python generate_deidentification_report.py input_ner.jsonl [--output_file input_ner_report.md]
```

**Report Structure:**
1. **Overall Statistics** - Total records, entities, and deidentification rate
2. **Removed Entities by Label** - Summary of removed entities by type
3. **Most Common Removed Entities** - Frequently removed entities
4. **Detailed Removed Entities by Label** - Breakdown by entity type
5. **Most Common New Entities (Placeholders)** - Added placeholder entities
6. **Sample Records** - Example records with their changes

### 5. Visualization Tools (`visualize_deidentification.py`)

This script generates charts and graphs to visualize the deidentification patterns from the NER analysis.

**Features:**
- Generates charts and graphs to visualize deidentification patterns
- Shows distribution of entity types removed during deidentification
- Creates histograms of deidentification rates across documents
- Generates four types of charts:
  1. Entity Counts - Bar chart of input, output, removed, and new entities
  2. Removed by Label - Distribution of removed entities by type
  3. Top Entities - Most frequently removed entities and added placeholders
  4. Deidentification Rates - Histogram of deidentification rates

**Usage:**
```bash
python visualize_deidentification.py input_ner.jsonl [--output_dir deidentification_charts]
```

## Complete Workflow Example

Here's a complete workflow to analyze a deidentification process:

```bash
# 1. Perform NER and entity comparison analysis
python ner.py output.jsonl

# 2. Generate a detailed deidentification report
python generate_deidentification_report.py output_ner.jsonl

# 3. Create visualizations of the deidentification patterns
python visualize_deidentification.py output_ner.jsonl

# 4. For more comprehensive analysis, include medical entities
python run_ner_analysis.py output.jsonl
python generate_deidentification_report.py output_combined_ner.jsonl
```

## Input Format

The scripts expect a JSONL file with each line containing a JSON object with "input" and "output" fields. For example:

```json
{"input": "Paziente di 65 anni con diabete mellito di tipo 2.", "output": "Paziente di [ETÀ] anni con diabete mellito di tipo 2."}
```

## Output Format

### NER Output

The NER script adds these fields to each record:

- `input_entities`: Entities found in the input field
- `output_entities`: Entities found in the output field
- `entity_comparison`: Analysis of differences between input and output entities
  - `removed_entities`: Entities that were removed during deidentification
  - `new_entities`: New entities (likely placeholders) added during deidentification
  - `stats`: Statistics about the deidentification process

### Medical NER Output

The medical NER script adds:

- `input_medical_entities`: Medical entities from the "input" field
- `output_medical_entities`: Medical entities from the "output" field

### Combined NER Output

The combined analysis adds:

- All fields from the NER and Medical NER outputs
- `combined_input_entities`: Merged entities from both NER systems for "input"
- `combined_output_entities`: Merged entities from both NER systems for "output"

### Entity Format

Each entity is represented as a JSON object with:

```json
{
  "text": "diabete mellito",
  "start": 20,
  "end": 35,
  "label": "CONDITION",
  "source": "medical"  // Only in combined output
}
```

## Testing

To test the scripts on sample data:

```bash
python test_ner.py
```

This script creates a small test JSONL file and runs all the NER scripts on it to verify functionality.

## Extending the Tools

The tools can be extended in several ways:

1. **Add medical terms**: Extend the `MEDICAL_TERMS` dictionary in `medical_ner.py` with additional Italian medical terminology.

2. **Add measurement patterns**: Enhance the `MEASUREMENT_PATTERNS` list in `medical_ner.py` with additional regex patterns for medical measurements.

3. **Create custom models**: Train custom spaCy models on domain-specific data for better recognition of clinical terms.

4. **Add visualizations**: Create additional visualization types in `visualize_deidentification.py` to represent deidentification patterns.

5. **Extend entity types**: Add new entity categories to capture additional clinical information.