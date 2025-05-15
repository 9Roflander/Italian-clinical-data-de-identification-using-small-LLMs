#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a deidentification report from NER results.
This script analyzes the output of ner.py and generates a report
highlighting what entities were removed during the deidentification process.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any
from collections import Counter


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


def save_report(report: str, file_path: str) -> None:
    """
    Save report to a file.
    
    Args:
        report: Report content
        file_path: Path where to save the report
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report)


def format_entity_list(entities: List[Dict[str, Any]], limit: int = 20) -> str:
    """
    Format a list of entities for the report.
    
    Args:
        entities: List of entities
        limit: Maximum number of entities to include
        
    Returns:
        Formatted entity list as string
    """
    if not entities:
        return "None"
    
    # Group entities by text
    entity_counter = Counter([entity['text'] for entity in entities])
    most_common = entity_counter.most_common(limit)
    
    # Format as list
    formatted = []
    for text, count in most_common:
        formatted.append(f"- {text} ({count}x)")
    
    if len(entity_counter) > limit:
        formatted.append(f"... and {len(entity_counter) - limit} more unique entities")
    
    return "\n".join(formatted)


def format_entity_by_label(entities_by_label: Dict[str, List[Dict[str, Any]]], limit: int = 10) -> str:
    """
    Format entities grouped by label.
    
    Args:
        entities_by_label: Dictionary of entities grouped by label
        limit: Maximum number of entities per label to include
        
    Returns:
        Formatted entities by label as string
    """
    if not entities_by_label:
        return "None"
    
    lines = []
    for label, entities in sorted(entities_by_label.items()):
        # Group entities by text
        entity_counter = Counter([entity['text'] for entity in entities])
        most_common = entity_counter.most_common(limit)
        
        lines.append(f"\n### {label} ({len(entities)})")
        for text, count in most_common:
            lines.append(f"- {text} ({count}x)")
        
        if len(entity_counter) > limit:
            lines.append(f"... and {len(entity_counter) - limit} more unique {label} entities")
    
    return "\n".join(lines)


def generate_report(data: List[Dict[str, Any]]) -> str:
    """
    Generate a deidentification report.
    
    Args:
        data: List of dictionaries from the JSONL file
        
    Returns:
        Report as string
    """
    # Check if we have overall stats
    if not data or 'overall_stats' not in data[0]:
        return "No deidentification data found in the input file."
    
    overall_stats = data[0]['overall_stats']
    
    # Initialize report
    report = [
        "# Deidentification Analysis Report",
        "",
        "## Overall Statistics",
        "",
        f"- Total records analyzed: {len(data)}",
        f"- Total input entities: {overall_stats['total_input_entities']}",
        f"- Total output entities: {overall_stats['total_output_entities']}",
        f"- Total removed entities: {overall_stats['total_removed_entities']}",
        f"- Total new entities (placeholders): {overall_stats['total_new_entities']}",
        f"- Overall deidentification rate: {overall_stats['overall_deidentification_rate']:.2f}%",
        "",
        "## Removed Entities by Label",
        ""
    ]
    
    # Add removed entities by label
    removed_by_label = overall_stats.get('removed_entities_by_label', {})
    if removed_by_label:
        for label, count in sorted(removed_by_label.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {label}: {count}")
    else:
        report.append("No entities removed.")
    
    # Collect all removed entities for detailed analysis
    all_removed_entities = []
    all_removed_by_label = {}
    all_new_entities = []
    
    for item in data:
        if 'entity_comparison' in item:
            comparison = item['entity_comparison']
            all_removed_entities.extend(comparison['removed_entities'])
            all_new_entities.extend(comparison['new_entities'])
            
            # Group by label
            for entity in comparison['removed_entities']:
                label = entity['label']
                if label not in all_removed_by_label:
                    all_removed_by_label[label] = []
                all_removed_by_label[label].append(entity)
    
    # Add most common removed entities
    report.extend([
        "",
        "## Most Common Removed Entities",
        ""
    ])
    report.append(format_entity_list(all_removed_entities))
    
    # Add detailed entities by label
    report.extend([
        "",
        "## Detailed Removed Entities by Label",
        ""
    ])
    report.append(format_entity_by_label(all_removed_by_label))
    
    # Add most common new entities (likely placeholders)
    report.extend([
        "",
        "## Most Common New Entities (Placeholders)",
        ""
    ])
    report.append(format_entity_list(all_new_entities))
    
    # Add sample records
    report.extend([
        "",
        "## Sample Records",
        ""
    ])
    
    # Add up to 5 sample records
    samples_added = 0
    for i, item in enumerate(data):
        if 'entity_comparison' in item and samples_added < 5:
            comparison = item['entity_comparison']
            if comparison['removed_entities']:
                samples_added += 1
                report.extend([
                    f"### Record {i+1}",
                    "",
                    "#### Input Text",
                    f"```",
                    item.get('input', '')[:500] + ('...' if len(item.get('input', '')) > 500 else ''),
                    f"```",
                    "",
                    "#### Output Text",
                    f"```",
                    item.get('output', '')[:500] + ('...' if len(item.get('output', '')) > 500 else ''),
                    f"```",
                    "",
                    "#### Removed Entities",
                    format_entity_list(comparison['removed_entities']),
                    "",
                    "#### New Entities (Placeholders)",
                    format_entity_list(comparison['new_entities']),
                    ""
                ])
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Generate a deidentification report from NER results")
    parser.add_argument("input_file", help="Path to input JSONL file (output from ner.py)")
    parser.add_argument("--output_file", help="Path to output report file (default: input_file with '_report.md' suffix)", default="output_report.md")
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output_file:
        base, ext = os.path.splitext(args.input_file)
        args.output_file = f"{base}_report.md"
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} records")
    
    # Generate report
    print("Generating deidentification report...")
    report = generate_report(data)
    
    # Save report
    print(f"Saving report to: {args.output_file}")
    save_report(report, args.output_file)
    print("Done!")


if __name__ == "__main__":
    main() 