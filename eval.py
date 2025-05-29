#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for comparing annotated dataset with deidentified output.
Calculates precision, recall, and F1 scores based on true positives, false positives,
and false negatives.
"""

import json
import argparse
import os
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
import csv

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
                            gold_annotations: List[Dict]) -> Tuple[int, int, int]:
    """
    Evaluate deidentification by comparing original and deidentified text with gold annotations
    Returns: (true_positives, false_positives, false_negatives)
    """
    true_positives = 0
    false_negatives = 0
    
    # Check each gold annotation
    for annotation in gold_annotations:
        entity_text = annotation['text']
        entity_type = annotation['type']
        
        # If entity is in original but not in deidentified -> true positive
        if entity_text in original_text and entity_text not in deidentified_text:
            true_positives += 1
        # If entity is in both -> false negative
        elif entity_text in original_text and entity_text in deidentified_text:
            false_negatives += 1
            
    # Count potential false positives by checking for placeholder patterns
    false_positives = 0
    placeholder_patterns = {
        'NOME': r'\[NOME\]',
        'ETÀ': r'\[ETÀ\]',
        'LUOGO/INDIRIZZO': r'\[LUOGO/INDIRIZZO\]',
        'DATA': r'\[DATA\]'
    }
    
    # For each entity type, count placeholders that don't correspond to gold annotations
    for entity_type, pattern in placeholder_patterns.items():
        import re
        placeholders = len(re.findall(pattern, deidentified_text))
        gold_count = len([a for a in gold_annotations if a['type'] == entity_type])
        if placeholders > gold_count:
            false_positives += placeholders - gold_count
            
    return true_positives, false_positives, false_negatives

def calculate_metrics(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """
    Calculate precision, recall and F1 score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description='Evaluate deidentification results')
    parser.add_argument('--annotated', required=True, help='Path to annotated dataset CSV')
    parser.add_argument('--deidentified', required=True, help='Path to deidentified output JSONL')
    parser.add_argument('--output', required=True, help='Path to output evaluation results')
    args = parser.parse_args()
    
    # Load data
    print("Loading annotated data...")
    annotated_data = load_annotated_data(args.annotated)
    print(f"Loaded {len(annotated_data)} annotated documents")
    
    print("Loading deidentified data...")
    deidentified_data = load_deidentified_data(args.deidentified)
    print(f"Loaded {len(deidentified_data)} deidentified documents")
    
    if len(annotated_data) != len(deidentified_data):
        print("Warning: Number of annotated and deidentified documents doesn't match")
        print(f"Annotated: {len(annotated_data)}, Deidentified: {len(deidentified_data)}")
    
    # Aggregate metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Metrics per category
    category_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    # Evaluate each document
    results = []
    for ann, deid in zip(annotated_data, deidentified_data):
        # Verify matching documents
        if ann['text'] != deid['input']:
            print("Warning: Mismatch between annotated and deidentified documents")
            print("Annotated:", ann['text'][:100], "...")
            print("Deidentified:", deid['input'][:100], "...")
            continue
            
        # Get metrics for this document
        tp, fp, fn = evaluate_deidentification(
            deid['input'],
            deid['output'],
            ann['annotations']
        )
        
        # Update totals
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Update per-category metrics
        for annotation in ann['annotations']:
            cat = annotation['type']
            if annotation['text'] in deid['input'] and annotation['text'] not in deid['output']:
                category_metrics[cat]['tp'] += 1
            elif annotation['text'] in deid['input'] and annotation['text'] in deid['output']:
                category_metrics[cat]['fn'] += 1
                
        # Count category false positives
        for cat, pattern in {
            'NOME': r'\[NOME\]',
            'ETÀ': r'\[ETÀ\]', 
            'LUOGO/INDIRIZZO': r'\[LUOGO/INDIRIZZO\]',
            'DATA': r'\[DATA\]'
        }.items():
            import re
            placeholders = len(re.findall(pattern, deid['output']))
            gold_count = len([a for a in ann['annotations'] if a['type'] == cat])
            if placeholders > gold_count:
                category_metrics[cat]['fp'] += placeholders - gold_count
        
        # Calculate document-level metrics
        precision, recall, f1 = calculate_metrics(tp, fp, fn)
        
        results.append({
            'document_id': len(results) + 1,
            'true_positives': tp,
            'false_positives': fp, 
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Calculate overall metrics
    overall_precision, overall_recall, overall_f1 = calculate_metrics(
        total_tp, total_fp, total_fn
    )
    
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
    
    # Save results
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                'overall_metrics': {
                    'precision': overall_precision,
                    'recall': overall_recall,
                    'f1': overall_f1,
                    'true_positives': total_tp,
                    'false_positives': total_fp,
                    'false_negatives': total_fn
                },
                'category_metrics': category_results,
                'document_results': results
            }, f, indent=2)
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)
        
    # Print summary
    print(f"\nOverall Metrics:")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall: {overall_recall:.3f}")
    print(f"F1 Score: {overall_f1:.3f}")
    print(f"\nTrue Positives: {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    
    print("\nMetrics by Category:")
    for cat, metrics in category_results.items():
        print(f"\n{cat}:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1']:.3f}")
        print(f"  TP: {metrics['true_positives']}, FP: {metrics['false_positives']}, FN: {metrics['false_negatives']}")

if __name__ == '__main__':
    main()
