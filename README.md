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
  - [Dataset Annotation with Gemma 3 27B](#6-dataset-annotation-with-gemma-3-27b-annotatepy)
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
- Preserves o