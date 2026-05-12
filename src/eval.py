#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for comparing annotated dataset with deidentified outputs.
Processes all model outputs in a directory and generates a comparative CSV report
with precision, recall, and F1 scores for each model.
"""

import json
import argparse
import os
import sys
import csv
from typing import Dict, List, Tuple
from collections import defaultdict
import re

def load_annotated_data(file_path: str) -> List[Dict]:
    """
    Load annotated data from CSV file
    """
    if not os.path.exists(file_path):
        print(f"Error: Annotated data file not found: {file_path}")
        sys.exit(1)
        
    annotated_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use annotations_redacted field which contains the gold standard annotations
                annotated_data.append({
                    'text': row['input'],
                    'annotations': json.loads(row['annotations_redacted'])
                })
    except Exception as e:
        print(f"Error loading annotated data: {e}")
        sys.exit(1)
        
    return annotated_data

def load_deidentified_data(file_path: str) -> List[Dict]:
    """
    Load deidentified data from JSONL file
    """
    if not os.path.exists(file_path):
        print(f"Error: Deidentified data file not found: {file_path}")
        sys.exit(1)
        
    deidentified_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    deidentified_data.append({
                        'input': data['input'],
                        'output': data['output']
                    })
    except Exception as e:
        print(f"Error loading deidentified data: {e}")
        sys.exit(1)
        
    return deidentified_data

def find_entity_in_text(entity: Dict, text: str) -> bool:
    """
    Check if an entity exists in the text
    """
    return entity['text'] in text

def evaluate_deidentification(original_text: str, 
                            deidentified_text: str, 
                            gold_annotations: List[Dict]) -> Dict[str, Tuple[int, int, int]]:
    """
    Evaluate deidentification by comparing original and deidentified text with gold annotations
    Returns: Dictionary mapping entity types to (true_positives, false_positives, false_negatives)
    """
    # Initialize counters for each entity type
    metrics = {
        'NOME': {'tp': 0, 'fp': 0, 'fn': 0},
        'ETÀ': {'tp': 0, 'fp': 0, 'fn': 0},
        'LUOGO/INDIRIZZO': {'tp': 0, 'fp': 0, 'fn': 0},
        'DATA': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    
    # Group annotations by text and type
    annotation_groups = defaultdict(list)
    for annotation in gold_annotations:
        key = (annotation['text'], annotation['type'])
        annotation_groups[key].append(annotation)
    
    #breakpoint()

    # Check each unique entity
    for (entity_text, entity_type), annotations in annotation_groups.items():
        #breakpoint()
        if entity_type not in metrics:
            continue  # Skip unknown entity types
            
        # Count occurrences in original and deidentified text
        original_count = len(re.findall(re.escape(entity_text), original_text))
        left_entities = len(re.findall(re.escape(entity_text), deidentified_text))
        
        # If we find more instances than annotated, use the annotated count
        gold_count = len(annotations)
        #breakpoint()
        
        # Calculate true positives and false negatives for this entity type
        removed_count = max(gold_count - left_entities, 0)
        metrics[entity_type]['tp'] += removed_count
        metrics[entity_type]['fn'] += left_entities
            
    # Count potential false positives by checking for placeholder patterns
    placeholder_patterns = {
        'NOME': r'\[NOME\]',
        'ETÀ': r'\[ETÀ\]',
        'LUOGO/INDIRIZZO': r'\[LUOGO/INDIRIZZO\]',
        'DATA': r'\[DATA\]'
    }
    
    # For each entity type, count placeholders that don't correspond to gold annotations
    for entity_type, pattern in placeholder_patterns.items():
        placeholders = len(re.findall(pattern, deidentified_text))
        gold_count = len([a for a in gold_annotations if a['type'] == entity_type])
        if placeholders > gold_count:
            metrics[entity_type]['fp'] += placeholders - gold_count
    
    # Convert metrics dict to tuple format for each type
    return {
        entity_type: (metrics[entity_type]['tp'], 
                     metrics[entity_type]['fp'], 
                     metrics[entity_type]['fn'])
        for entity_type in metrics
    }

def calculate_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculate precision, recall and F1 score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate_model(model_file: str, annotated_data: List[Dict]) -> Dict:
    """
    Evaluate a single model's output and return metrics
    """
    print(f"\nEvaluating model: {os.path.basename(model_file)}")
    
    deidentified_data = load_deidentified_data(model_file)
    
    if len(annotated_data) != len(deidentified_data):
        print("Warning: Number of annotated and deidentified documents doesn't match")
        print(f"Annotated: {len(annotated_data)}, Deidentified: {len(deidentified_data)}")
    
    # Initialize metrics for each category
    category_metrics = {
        'NOME': {'tp': 0, 'fp': 0, 'fn': 0},
        'ETÀ': {'tp': 0, 'fp': 0, 'fn': 0},
        'LUOGO/INDIRIZZO': {'tp': 0, 'fp': 0, 'fn': 0},
        'DATA': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    
    for ann, deid in zip(annotated_data, deidentified_data):
        if ann['text'] != deid['input']:
            print("Warning: Mismatch between annotated and deidentified documents")
            continue
            
        # Get metrics per category for this document
        doc_metrics = evaluate_deidentification(
            deid['input'],
            deid['output'],
            ann['annotations']
        )
        
        # Aggregate metrics by category
        for category, (tp, fp, fn) in doc_metrics.items():
            category_metrics[category]['tp'] += tp
            category_metrics[category]['fp'] += fp
            category_metrics[category]['fn'] += fn
    
    # Calculate overall metrics
    total_tp = sum(m['tp'] for m in category_metrics.values())
    total_fp = sum(m['fp'] for m in category_metrics.values())
    total_fn = sum(m['fn'] for m in category_metrics.values())
    overall_precision, overall_recall, overall_f1 = calculate_metrics(total_tp, total_fp, total_fn)
    
    # Calculate per-category metrics
    category_results = {}
    for cat, metrics in category_metrics.items():
        cat_precision, cat_recall, cat_f1 = calculate_metrics(
            metrics['tp'], metrics['fp'], metrics['fn']
        )
        category_results[cat] = {
            'precision': cat_precision,
            'recall': cat_recall,
            'f1': cat_f1,
            'true_positives': metrics['tp'],
            'false_positives': metrics['fp'],
            'false_negatives': metrics['fn']
        }
    
    return {
        'model_name': os.path.basename(model_file).replace('_clean_output=False.jsonl', ''),
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        },
        'category_metrics': category_results
    }

def write_csv_report(results: List[Dict], output_file: str):
    """
    Write evaluation results to CSV file in long format with one row per category per model
    """
    # Prepare CSV headers
    headers = [
        'Model',
        'Category',
        'Precision',
        'Recall',
        'F1',
        'True_Positives',
        'False_Positives',
        'False_Negatives'
    ]
    
    # Write results
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for result in results:
            model_name = result['model_name']
            
            # Write overall metrics
            writer.writerow([
                model_name,
                'Overall',
                result['overall_metrics']['precision'],
                result['overall_metrics']['recall'],
                result['overall_metrics']['f1'],
                result['overall_metrics']['true_positives'],
                result['overall_metrics']['false_positives'],
                result['overall_metrics']['false_negatives']
            ])
            
            # Write per-category metrics
            for category, metrics in result['category_metrics'].items():
                writer.writerow([
                    model_name,
                    category,
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1'],
                    metrics['true_positives'],
                    metrics['false_positives'],
                    metrics['false_negatives']
                ])

def main():
    parser = argparse.ArgumentParser(description='Evaluate deidentification results for multiple models')
    parser.add_argument('--annotated', required=True, help='Path to annotated dataset CSV')
    parser.add_argument('--input-dir', required=True, help='Directory containing model outputs')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Load annotated data once
    print("Loading annotated data...")
    annotated_data = load_annotated_data(args.annotated)
    print(f"Loaded {len(annotated_data)} annotated documents")
    

    #print([f for f in os.listdir(args.input_dir) if f.endswith('clean_output=False.jsonl')])
    # Find all model output files
    model_files = [f for f in os.listdir(args.input_dir) 
                  if f.endswith('_clean_output=False.jsonl') and 
                  not f.endswith('_evaluatedBy')]
    
    if not model_files:
        print(f"Error: No model output files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(model_files)} model output files")
    
    # Evaluate each model
    results = []
    for model_file in model_files:
        try:
            result = evaluate_model(
                os.path.join(args.input_dir, model_file),
                annotated_data
            )
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {model_file}: {e}")
            continue
    
    # Write comparative report
    try:
        write_csv_report(results, args.output)
        print(f"\nResults written to {args.output}")
    except Exception as e:
        print(f"Error writing results: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
