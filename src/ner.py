#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Named Entity Recognition for Italian clinical notes.
This script reads a JSONL file with 'input' and 'output' fields and
performs NER on both values using a spaCy Italian language model.
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any, Tuple, Set

import spacy
from spacy.tokens import Doc, Span
from tqdm import tqdm


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


def save_jsonl(data: List[Dict[str, Any]], output_dir: str, input_file: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where to save the JSONL file
    """
    output_file = f"{input_file}_ner.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def extract_entities(doc: Doc) -> List[Dict[str, Any]]:
    """
    Extract named entities from a spaCy Doc object.
    
    Args:
        doc: spaCy Doc object
        
    Returns:
        List of dictionaries with entity information
    """
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'start': ent.start_char,
            'end': ent.end_char,
            'label': ent.label_,
        })
    return entities


def process_text(nlp, text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Process text with spaCy NER model.
    
    Args:
        nlp: spaCy model
        text: Text to process
        
    Returns:
        Tuple of (processed text, list of entities)
    """
    doc = nlp(text)
    entities = extract_entities(doc)
    return doc.text, entities


def compare_entities(input_entities: List[Dict[str, Any]], 
                     output_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare input and output entities to identify removed or modified entities.
    
    Args:
        input_entities: List of entities from the input text
        output_entities: List of entities from the output text
        
    Returns:
        Dictionary with comparison results
    """
    # Create sets for easier comparison
    input_texts = {entity['text'].lower() for entity in input_entities}
    output_texts = {entity['text'].lower() for entity in output_entities}
    
    # Find removed entities
    removed_entities = []
    for entity in input_entities:
        if entity['text'].lower() not in output_texts:
            removed_entities.append(entity)
    
    # Find new entities (placeholders)
    new_entities = []
    for entity in output_entities:
        if entity['text'].lower() not in input_texts:
            new_entities.append(entity)
    
    # Count entities by label
    input_label_counts = {}
    for entity in input_entities:
        label = entity['label']
        input_label_counts[label] = input_label_counts.get(label, 0) + 1
    
    output_label_counts = {}
    for entity in output_entities:
        label = entity['label']
        output_label_counts[label] = output_label_counts.get(label, 0) + 1
    
    # Calculate statistics
    total_input = len(input_entities)
    total_output = len(output_entities)
    total_removed = len(removed_entities)
    total_new = len(new_entities)
    
    deidentification_rate = 0
    if total_input > 0:
        deidentification_rate = (total_removed / total_input) * 100
    
    return {
        'removed_entities': removed_entities,
        'new_entities': new_entities,
        'input_label_counts': input_label_counts,
        'output_label_counts': output_label_counts,
        'stats': {
            'total_input_entities': total_input,
            'total_output_entities': total_output,
            'total_removed_entities': total_removed,
            'total_new_entities': total_new,
            'deidentification_rate': deidentification_rate
        }
    }


def process_jsonl(nlp, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all records in the data with spaCy NER.
    
    Args:
        nlp: spaCy model
        data: List of dictionaries with 'input' and 'output' fields
        
    Returns:
        List of dictionaries with added NER information
    """
    results = []
    
    # Track overall statistics
    total_input_entities = 0
    total_output_entities = 0
    total_removed_entities = 0
    total_new_entities = 0
    all_removed_entities_by_label = {}
    
    for item in tqdm(data, desc="Processing records"):
        result = item.copy()
        
        input_entities = []
        output_entities = []
        
        # Process 'input' field
        if 'input' in item:
            input_text, input_entities = process_text(nlp, item['input'])
            result['input_entities'] = input_entities
            total_input_entities += len(input_entities)
        
        # Process 'output' field
        if 'output' in item:
            output_text, output_entities = process_text(nlp, item['output'])
            result['output_entities'] = output_entities
            total_output_entities += len(output_entities)
        
        # Compare entities if both input and output are present
        if 'input' in item and 'output' in item:
            comparison = compare_entities(input_entities, output_entities)
            result['entity_comparison'] = comparison
            
            # Update overall statistics
            total_removed_entities += comparison['stats']['total_removed_entities']
            total_new_entities += comparison['stats']['total_new_entities']
            
            # Update removed entities by label
            for entity in comparison['removed_entities']:
                label = entity['label']
                if label not in all_removed_entities_by_label:
                    all_removed_entities_by_label[label] = []
                all_removed_entities_by_label[label].append(entity)
        
        results.append(result)
    
    # Calculate overall deidentification rate
    overall_deidentification_rate = 0
    if total_input_entities > 0:
        overall_deidentification_rate = (total_removed_entities / total_input_entities) * 100
    
    # Add overall statistics to the first record (if any results exist)
    if results:
        results[0]['overall_stats'] = {
            'total_input_entities': total_input_entities,
            'total_output_entities': total_output_entities,
            'total_removed_entities': total_removed_entities,
            'total_new_entities': total_new_entities,
            'overall_deidentification_rate': overall_deidentification_rate,
            'removed_entities_by_label': {
                label: len(entities) for label, entities in all_removed_entities_by_label.items()
            }
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Perform NER on Italian clinical notes in JSONL format")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("--output_dir", help="Path to output directory (default: outputs)", default="./outputs")
    parser.add_argument("--model", default="it_core_news_lg", help="spaCy Italian model to use (default: it_core_news_lg)")
    args = parser.parse_args()
    
    # Set default output file if not provided
    output_file = os.path.join(args.output_dir, f"{args.input_file}_ner.jsonl")
    
    # Load spaCy model
    try:
        print(f"Loading spaCy model: {args.model}")
        nlp = spacy.load(args.model)
    except OSError:
        print(f"Model '{args.model}' not found. Attempting to download it...")
        os.system(f"python -m spacy download {args.model}")
        try:
            nlp = spacy.load(args.model)
        except OSError:
            print(f"Failed to download model '{args.model}'. Please install it manually:")
            print(f"python -m spacy download {args.model}")
            sys.exit(1)
    
    # Load data
    print(f"Loading data from: {args.input_file}")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} records")
    
    # Process data
    print("Processing data with NER...")
    results = process_jsonl(nlp, data)
    
    # Save results
    print(f"Saving results to: {args.output_dir}")
    save_jsonl(results, args.output_dir, args.input_file)
    print("Done!")


if __name__ == "__main__":
    main()
