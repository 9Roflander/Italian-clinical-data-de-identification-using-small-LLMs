#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Italian Clinical Note Annotation Tool with Gemma 3 27B

This script processes clinical notes written in Italian and uses Gemma 3 27B
to identify personally identifiable information (PII) according to GDPR requirements.
The model annotates the text with the type of personal information and its position.

Usage:
    python annotate.py --input input_file.jsonl --output annotated_file.jsonl [--model gemma-3-27b-it] [--backend ollama]
"""

import argparse
import json
import os
import sys
import time
import logging
from typing import List, Dict, Any, Tuple, Optional
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("annotate")

# Try to import backends, gracefully handle missing dependencies
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available. Install with: pip install ollama")

# GDPR sensitive data categories for annotation
GDPR_CATEGORIES = [
    "Nome e cognome",
    "Data di nascita",
    "Età",
    "Codice fiscale",
    "Numeri di tessera sanitaria",
    "Numeri di cartella clinica",
    "Numeri di telefono",
    "Indirizzi email",
    "Indirizzi di residenza/domicilio",
    "Nomi propri",
    "Numeri di identificazione di dispositivi medici",
    #"Nomi di medici curanti",
    #"Date esatte di ricovero/dimissione",
    "Numeri di previdenza sociale",
    "Nome dell'ospedale o struttura sanitaria specifica",
    "Località geografiche specifiche",
    "Qualsiasi altro dato che potrebbe identificare il paziente in modo univoco, da soli o in combinazione con altri dati"
]

class BackendNotAvailableError(Exception):
    """Raised when the requested backend is not available"""
    pass

class ModelBackend:
    """Base class for model backends"""
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate(self, prompt: str) -> str:
        raise NotImplementedError("Subclasses must implement generate()")

class OllamaBackend(ModelBackend):
    """Ollama backend for local model inference"""
    def __init__(self, model_name: str, max_tokens: int = 8192, temperature: float = 0.1):
        if not OLLAMA_AVAILABLE:
            raise BackendNotAvailableError("Ollama is not installed")
        super().__init__(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info(f"Using Ollama backend with model: {model_name} (max tokens: {max_tokens}, temperature: {temperature})")
    
    def generate(self, prompt: str) -> str:
        try:
            response = ollama.generate(
                model=self.model_name, 
                prompt=prompt,
                options={
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                }
            )
            return response['response']
        except Exception as e:
            logger.error(f"Error generating with Ollama: {e}")
            raise

def create_annotation_prompt(text: str) -> str:
    """
    Create the prompt for the LLM to annotate personal information in clinical notes
    
    Args:
        text: The clinical note text to be annotated
        
    Returns:
        str: The complete prompt for the LLM
    """
    categories_str = "\n".join([f"- {cat}" for cat in GDPR_CATEGORIES])
    
    prompt = f"""Sei un assistente specializzato nell'annotazione di informazioni personali in note cliniche in italiano, in conformità con il GDPR.

Ti fornirò una nota clinica e tu dovrai identificare tutte le informazioni personali che rientrano nelle seguenti categorie:

{categories_str}

ISTRUZIONI IMPORTANTI:
1. Per ogni informazione personale identificata, indica:
   a) Il tipo di informazione (es. "NOME", "DATA_NASCITA", "CODICE_FISCALE", "INDIRIZZO", ecc.)
   b) Il testo esatto dell'informazione 
   c) La posizione di inizio e fine dell'informazione nel testo (indice di carattere)

2. Restituisci le annotazioni in formato JSON come segue:
   {{
     "annotations": [
       {{ "type": "NOME", "text": "Mario Rossi", "start": 12, "end": 23 }},
       {{ "type": "DATA_NASCITA", "text": "15/06/1965", "start": 34, "end": 44 }},
       ...
     ]
   }}

3. Sii estremamente preciso nell'identificare tutte le informazioni personali.
4. Non annotare dati relativi a misurazioni di valori clinici o dosaggi di farmaci.
5. Non annotare date ed esiti di ricoveri e procedure mediche.
6. La posizione (start, end) deve riferirsi esattamente all'indice di carattere nel testo originale.
7. Classifica correttamente il tipo di informazione personale utilizzando etichette brevi e descrittive.
8. Tieni presente che queste annotazioni sono utilizzate per la deidentificazione delle note cliniche, quindi assicurati di non annotare dati che sono necessari per delineare il quadro clinico, altrimenti verranno rimossi.
 
NOTA CLINICA:
{text}

ANNOTAZIONI:
"""
    
    return prompt

def parse_annotations(model_output: str) -> Dict:
    """
    Parse the model's output to extract annotations
    
    Args:
        model_output: The raw output from the model
        
    Returns:
        Dict: Parsed annotations
    """
    # Try to find JSON structure in the output
    json_pattern = r'({[\s\S]*})'
    match = re.search(json_pattern, model_output)
    
    if match:
        try:
            json_str = match.group(1)
            annotations = json.loads(json_str)
            return annotations
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from model output")
    
    # Fallback: try to extract annotations using regex if JSON parsing fails
    annotations = {"annotations": []}
    pattern = r'type":\s*"([^"]+)",\s*"text":\s*"([^"]+)",\s*"start":\s*(\d+),\s*"end":\s*(\d+)'
    
    matches = re.findall(pattern, model_output)
    for match in matches:
        annotation_type, text, start, end = match
        annotations["annotations"].append({
            "type": annotation_type,
            "text": text,
            "start": int(start),
            "end": int(end)
        })
    
    if not annotations["annotations"]:
        logger.warning("Could not extract any annotations from model output")
    
    return annotations

def annotate_clinical_note(note: str, backend: ModelBackend) -> Dict:
    """
    Annotate personal information in a clinical note using an LLM
    
    Args:
        note: The clinical note to annotate
        backend: The model backend to use
        
    Returns:
        Dict: Annotations of personal information
    """
    prompt = create_annotation_prompt(note)
    logger.debug(f"Generated prompt: {prompt}")
    
    model_output = backend.generate(prompt)
    logger.debug(f"Model output: {model_output}")
    
    annotations = parse_annotations(model_output)
    logger.debug(f"Parsed annotations: {annotations}")
    
    return annotations

def process_clinical_notes(
    input_file: str,
    output_file: str,
    backend: ModelBackend,
    max_failures: int = 3
) -> Tuple[int, int]:
    """
    Process a file containing clinical notes and annotate personal information
    
    Args:
        input_file: Path to the input file (JSONL format)
        output_file: Path to the output file (JSONL format)
        backend: The model backend to use
        max_failures: Maximum number of consecutive failures before aborting
        
    Returns:
        Tuple[int, int]: Number of processed notes and number of failed notes
    """
    processed_count = 0
    failed_count = 0
    consecutive_failures = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                record = json.loads(line.strip())
                
                if 'input' not in record:
                    logger.warning(f"Line {line_num}: Missing 'input' field, skipping")
                    continue
                
                logger.info(f"Processing note {line_num}")
                start_time = time.time()
                
                # Annotate the original note
                annotations = annotate_clinical_note(record['input'], backend)
                
                # Add annotations to the record
                record['annotations'] = annotations.get('annotations', [])
                
                # Write the annotated record to the output file
                f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                
                processed_count += 1
                consecutive_failures = 0
                
                logger.info(f"Completed note {line_num} in {time.time() - start_time:.2f} seconds")
                logger.info(f"Found {len(record['annotations'])} personal information entities")
                
            except Exception as e:
                logger.error(f"Error processing line {line_num}: {e}")
                failed_count += 1
                consecutive_failures += 1
                
                if consecutive_failures >= max_failures:
                    logger.error(f"Aborting after {consecutive_failures} consecutive failures")
                    break
    
    return processed_count, failed_count

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Annotate personal information in Italian clinical notes using Gemma 3 27B')
    parser.add_argument('--input', required=True, help='Input file in JSONL format')
    parser.add_argument('--output', required=True, help='Output file in JSONL format')
    parser.add_argument('--model', default='gemma3:27b', help='Model name to use')
    parser.add_argument('--backend', default='ollama', choices=['ollama'], help='Backend to use for model inference')
    parser.add_argument('--max-tokens', type=int, default=8192, help='Maximum tokens for model inference')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for model inference')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    if not os.path.exists(args.input):
        logger.error(f"Input file does not exist: {args.input}")
        return 1
    
    try:
        # Create model backend
        if args.backend == 'ollama':
            backend = OllamaBackend(args.model, args.max_tokens, args.temperature)
        else:
            logger.error(f"Unsupported backend: {args.backend}")
            return 1
        
        logger.info(f"Starting annotation process with model {args.model} on {args.backend} backend")
        
        # Process the clinical notes
        processed_count, failed_count = process_clinical_notes(
            args.input,
            args.output,
            backend
        )
        
        logger.info(f"Annotation completed. Processed {processed_count} notes, failed {failed_count} notes.")
        
        return 0 if failed_count == 0 else 1
    
    except Exception as e:
        logger.error(f"Error during annotation process: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
