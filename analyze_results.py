#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze JSONL results from the clinical notes de-identification system.

This script provides utilities to analyze the JSONL output files produced by deid.py,
including statistics, extraction of specific fields, and comparison between original
and de-identified texts.

Usage:
    python analyze_results.py --input results.jsonl [--output report.txt] [--extract-field output]
"""

import argparse
import json
import sys
import os
import re
from typing import Dict, List, Any, Tuple, Optional

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line}")
                print(f"Error details: {e}")
    return data

def extract_field(data: List[Dict[str, Any]], field: str) -> List[str]:
    """Extract a specific field from each entry"""
    return [entry[field] for entry in data if field in entry]

def count_entities(text: str) -> Dict[str, int]:
    """Count the number of de-identified entities (text in [BRACKETS])"""
    entities = re.findall(r'\[(.*?)\]', text)
    entity_counts = {}
    for entity in entities:
        entity_counts[entity] = entity_counts.get(entity, 0) + 1
    return entity_counts

def analyze_entities(data: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """Analyze the de-identified entities across all entries"""
    all_entity_counts = {}
    entity_examples = {}
    
    for entry in data:
        if 'output' in entry:
            entity_counts = count_entities(entry['output'])
            for entity, count in entity_counts.items():
                all_entity_counts[entity] = all_entity_counts.get(entity, 0) + count
                
                # Store an example of this entity type if we don't have one already
                if entity not in entity_examples:
                    match = re.search(r'\[' + re.escape(entity) + r'\]', entry['output'])
                    if match:
                        # Get some context around the entity
                        start = max(0, match.start() - 30)
                        end = min(len(entry['output']), match.end() + 30)
                        context = entry['output'][start:end]
                        entity_examples[entity] = context
    
    return all_entity_counts, entity_examples

def generate_report(data: List[Dict[str, Any]]) -> str:
    """Generate a comprehensive analysis report"""
    report = []
    
    # Basic statistics
    successful = len([e for e in data if 'output' in e])
    errors = len([e for e in data if 'error' in e])
    
    report.append("=== DE-IDENTIFICATION REPORT ===")
    report.append(f"Total entries: {len(data)}")
    report.append(f"Successfully processed: {successful}")
    report.append(f"Errors: {errors}")
    report.append("")
    
    # Entity analysis
    if successful > 0:
        entity_counts, entity_examples = analyze_entities(data)
        
        report.append("=== ENTITY STATISTICS ===")
        if entity_counts:
            # Sort entities by frequency
            sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
            report.append("Entity types found (sorted by frequency):")
            for entity, count in sorted_entities:
                report.append(f"  [{entity}]: {count} occurrences")
            
            report.append("\n=== ENTITY EXAMPLES ===")
            for entity, example in entity_examples.items():
                report.append(f"  [{entity}] example: \"...{example}...\"")
        else:
            report.append("No entities ([TAG]) found in the de-identified text.")
        
        report.append("")
    
    # Error analysis
    if errors > 0:
        report.append("=== ERROR ANALYSIS ===")
        error_types = {}
        for entry in data:
            if 'error' in entry:
                error_msg = entry['error']
                # Get the first part of the error message for categorization
                error_category = error_msg.split(':')[0] if ':' in error_msg else error_msg
                error_types[error_category] = error_types.get(error_category, 0) + 1
        
        report.append("Error types encountered:")
        for error_type, count in error_types.items():
            report.append(f"  {error_type}: {count} occurrences")
    
    return "\n".join(report)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Analyze JSONL results from the clinical notes de-identification system")
    parser.add_argument("--input", required=True, help="Input JSONL file to analyze")
    parser.add_argument("--output", help="Output file for analysis report (default: stdout)")
    parser.add_argument("--extract-field", choices=["input", "prompt", "output"], 
                      help="Extract a specific field to a separate file")
    parser.add_argument("--extract-output", help="Output file for extracted field")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    # Load JSONL data
    data = load_jsonl(args.input)
    if not data:
        print("Error: No valid data found in the input file")
        return 1
    
    # Generate report
    report = generate_report(data)
    
    # Output report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Analysis report written to {args.output}")
    else:
        print(report)
    
    # Extract field if requested
    if args.extract_field:
        field_data = extract_field(data, args.extract_field)
        
        if field_data:
            output_file = args.extract_output or f"{os.path.splitext(args.input)[0]}_{args.extract_field}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n\n---\n\n".join(field_data))
            print(f"Extracted {args.extract_field} field to {output_file}")
        else:
            print(f"No {args.extract_field} field found in the data")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 