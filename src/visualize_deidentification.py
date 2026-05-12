#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize deidentification results.
This script generates charts from the NER analysis to visualize
entity removal patterns and deidentification rates.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any
from collections import Counter

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("This script requires matplotlib. Install it with: pip install matplotlib")
    sys.exit(1)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries from the JSONL file
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def collect_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collect statistics from the data.
    
    Args:
        data: List of dictionaries from the JSONL file
        
    Returns:
        Dictionary with collected statistics
    """
    # Check if we have overall stats
    if not data or 'overall_stats' not in data[0]:
        return {}
    
    stats = {
        'overall': data[0]['overall_stats'],
        'top_removed': Counter(),
        'top_new': Counter(),
        'removed_by_label': {},
        'record_stats': []
    }
    
    # Collect entity statistics
    for item in data:
        if 'entity_comparison' in item:
            comparison = item['entity_comparison']
            
            # Top removed entities
            for entity in comparison['removed_entities']:
                stats['top_removed'][entity['text']] += 1
            
            # Top new entities
            for entity in comparison['new_entities']:
                stats['top_new'][entity['text']] += 1
            
            # Record-level statistics
            stats['record_stats'].append({
                'input_count': comparison['stats']['total_input_entities'],
                'output_count': comparison['stats']['total_output_entities'],
                'removed_count': comparison['stats']['total_removed_entities'],
                'new_count': comparison['stats']['total_new_entities'],
                'deidentification_rate': comparison['stats']['deidentification_rate']
            })
    
    # Collect removed by label from overall stats
    if 'removed_entities_by_label' in stats['overall']:
        stats['removed_by_label'] = stats['overall']['removed_entities_by_label']
    
    return stats


def plot_entity_counts(stats: Dict[str, Any], output_dir: str) -> None:
    """
    Plot entity counts.
    
    Args:
        stats: Dictionary with collected statistics
        output_dir: Directory to save the plots
    """
    if not stats or not stats.get('overall'):
        return
    
    # Create bar chart of input vs output entities
    plt.figure(figsize=(10, 6))
    
    labels = ['Input', 'Output', 'Removed', 'New (Placeholders)']
    values = [
        stats['overall']['total_input_entities'],
        stats['overall']['total_output_entities'],
        stats['overall']['total_removed_entities'],
        stats['overall']['total_new_entities']
    ]
    
    ax = plt.bar(labels, values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    
    plt.title('Entity Counts in Deidentification Process')
    plt.ylabel('Number of Entities')
    plt.ylim(bottom=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entity_counts.png'), dpi=300)
    plt.close()


def plot_removed_by_label(stats: Dict[str, Any], output_dir: str) -> None:
    """
    Plot removed entities by label.
    
    Args:
        stats: Dictionary with collected statistics
        output_dir: Directory to save the plots
    """
    if not stats or not stats.get('removed_by_label'):
        return
    
    # Create bar chart of removed entities by label
    plt.figure(figsize=(12, 8))
    
    labels = list(stats['removed_by_label'].keys())
    values = list(stats['removed_by_label'].values())
    
    # Sort by value in descending order
    sorted_indices = np.argsort(values)[::-1]
    labels = [labels[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    ax = plt.bar(labels, values, color='#e74c3c')
    
    plt.title('Removed Entities by Label')
    plt.ylabel('Number of Entities')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(bottom=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'removed_by_label.png'), dpi=300)
    plt.close()


def plot_top_entities(stats: Dict[str, Any], output_dir: str) -> None:
    """
    Plot top removed and new entities.
    
    Args:
        stats: Dictionary with collected statistics
        output_dir: Directory to save the plots
    """
    if not stats:
        return
    
    # Plot top removed entities
    if stats.get('top_removed'):
        plt.figure(figsize=(12, 8))
        
        # Get top 15 entities
        top_entities = stats['top_removed'].most_common(15)
        labels = [entity[0] for entity in top_entities]
        values = [entity[1] for entity in top_entities]
        
        # Create horizontal bar chart
        ax = plt.barh(labels, values, color='#3498db')
        
        plt.title('Top 15 Most Frequently Removed Entities')
        plt.xlabel('Count')
        plt.gca().invert_yaxis()  # Invert y-axis to have most frequent at top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(v + 0.2, i, str(v), va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_removed_entities.png'), dpi=300)
        plt.close()
    
    # Plot top new entities (placeholders)
    if stats.get('top_new'):
        plt.figure(figsize=(12, 8))
        
        # Get top 15 entities
        top_entities = stats['top_new'].most_common(15)
        labels = [entity[0] for entity in top_entities]
        values = [entity[1] for entity in top_entities]
        
        # Create horizontal bar chart
        ax = plt.barh(labels, values, color='#2ecc71')
        
        plt.title('Top 15 Most Frequently Added Placeholders')
        plt.xlabel('Count')
        plt.gca().invert_yaxis()  # Invert y-axis to have most frequent at top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(v + 0.2, i, str(v), va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_new_entities.png'), dpi=300)
        plt.close()


def plot_deidentification_rates(stats: Dict[str, Any], output_dir: str) -> None:
    """
    Plot deidentification rates.
    
    Args:
        stats: Dictionary with collected statistics
        output_dir: Directory to save the plots
    """
    if not stats or not stats.get('record_stats'):
        return
    
    # Extract deidentification rates
    rates = [record['deidentification_rate'] for record in stats['record_stats'] 
             if record['input_count'] > 0]  # Skip records with no input entities
    
    if not rates:
        return
    
    # Create histogram of deidentification rates
    plt.figure(figsize=(10, 6))
    
    bins = np.linspace(0, 100, 21)  # 5% bins from 0% to 100%
    plt.hist(rates, bins=bins, color='#3498db', alpha=0.7, edgecolor='black')
    
    plt.title('Distribution of Deidentification Rates')
    plt.xlabel('Deidentification Rate (%)')
    plt.ylabel('Number of Records')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add overall rate line
    if 'overall' in stats and 'overall_deidentification_rate' in stats['overall']:
        overall_rate = stats['overall']['overall_deidentification_rate']
        plt.axvline(x=overall_rate, color='#e74c3c', linestyle='--', 
                   label=f'Overall Rate: {overall_rate:.2f}%')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deidentification_rates.png'), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize deidentification results")
    parser.add_argument("input_file", help="Path to input JSONL file (output from ner.py)")
    parser.add_argument("--output_dir", help="Directory to save the charts (default: 'deidentification_charts')", default="./charts")
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if not args.output_dir:
        args.output_dir = "deidentification_charts"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} records")
    
    # Collect statistics
    print("Collecting statistics...")
    stats = collect_statistics(data)
    
    if not stats:
        print("Error: No deidentification data found in the input file.")
        return
    
    # Generate plots
    print("Generating plots...")
    plot_entity_counts(stats, args.output_dir)
    plot_removed_by_label(stats, args.output_dir)
    plot_top_entities(stats, args.output_dir)
    plot_deidentification_rates(stats, args.output_dir)
    
    print(f"Charts saved to: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main() 