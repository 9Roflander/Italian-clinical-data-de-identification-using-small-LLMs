#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Italian Clinical Note De-identification System

This script processes clinical notes written in Italian and redacts personally identifiable 
information (PII) and sensitive health data in accordance with GDPR requirements.
It uses a Large Language Model to identify and mask sensitive information.

Usage:
    python deid.py --input input_notes.csv --output redacted_notes.csv [--model model_name] [--backend backend_type]
"""

import argparse
import csv
import os
import sys
import time
import json
from typing import List, Dict, Optional, Tuple, Any
import logging
from datetime import datetime
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("deid")

# Try to import backends, gracefully handle missing dependencies
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama not available. Install with: pip install ollama")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("VLLM not available. Install with: pip install vllm")


'''
GDPR sensitive info according to usercentrics.com/resources/gdpr-checlist

names, addresses, phone numbers, and email addresses
identification numbers like Social Security, passport, or driver’s license numbers
location data such as GPS coordinates or IP addresses
biometric data like fingerprints, facial recognition, or DNA
genetic data
health-related or healthcare information
political opinions, religious beliefs, or membership in trade unions
'''

# GDPR sensitive data categories to be de-identified
GDPR_CATEGORIES = [
    "Nome e cognome del paziente",
    "Data di nascita completa",
    "Codice fiscale",
    "Numeri di tessera sanitaria",
    "Numeri di cartella clinica",
    "Numeri di telefono",
    "Indirizzi email",
    "Indirizzi di residenza/domicilio",
    "Nomi di familiari/caregiver",
    "Numeri di identificazione di dispositivi medici",
    "Nomi di medici curanti",
    "Date esatte di ricovero/dimissione",
    "Numeri di previdenza sociale",
    "Nome dell'ospedale o struttura sanitaria specifica",
    "Località geografiche specifiche",
    "Qualsiasi altro dato che potrebbe identificare il paziente in modo univoco"
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

class VLLMBackend(ModelBackend):
    """VLLM backend for efficient model inference"""
    def __init__(self, model_name: str, max_tokens: int = 8192, temperature: float = 0.1):
        if not VLLM_AVAILABLE:
            raise BackendNotAvailableError("VLLM is not installed")
        super().__init__(model_name)
        self.max_tokens = max_tokens
        self.temperature = temperature
        logger.info(f"Loading model with VLLM backend: {model_name} (max tokens: {max_tokens}, temperature: {temperature})")
        self.llm = LLM(model=model_name)
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        logger.info("Model loaded successfully")
    
    def generate(self, prompt: str) -> str:
        try:
            outputs = self.llm.generate([prompt], self.sampling_params)
            return outputs[0].outputs[0].text
        except Exception as e:
            logger.error(f"Error generating with VLLM: {e}")
            raise

def create_deidentification_prompt(text: str) -> str:
    """
    Create the prompt for the LLM to de-identify clinical notes
    
    Args:
        text: The clinical note text to be de-identified
        
    Returns:
        str: The complete prompt for the LLM
    """
    categories_str = "\n".join([f"- {cat}" for cat in GDPR_CATEGORIES])
    
    prompt = f"""Sei un assistente specializzato nella de-identificazione di note cliniche in italiano, in conformità con il GDPR.
    
Ti fornirò una nota clinica e tu dovrai identificare e sostituire tutte le seguenti informazioni sensibili:

{categories_str}

ISTRUZIONI IMPORTANTI:
1. Sostituisci tutte le informazioni sensibili con i tag appropriati come [NOME], [COGNOME], [DATA], [INDIRIZZO], [ID], ecc.
2. Non modificare nulla all'infuori delle informazioni sensibili.
3. Non rimuovere o modificare informazioni mediche rilevanti come diagnosi, trattamenti, dosaggi, ecc.
4. Se un'informazione potrebbe essere identificativa ma non sei sicuro, mascherala comunque.
5. Non includere spiegazioni o commenti, restituisci SOLO il testo de-identificato.
6. Il risultato deve essere un testo estremamente simile all'originale, le uniche modifiche dovrebbero essere le sostituzioni delle informazioni sensibili.
7. Il risultato verrà inserito in una rete neurale dal contesto molto limitato, quindi devi evitare assolutamente di includere commenti o spiegazioni.

NOTA CLINICA:
{text}

TESTO DE-IDENTIFICATO:
"""
    
    return prompt

def clean_model_output(raw_output: str) -> str:
    """
    Clean and extract only the de-identified text from the model's output
    
    Args:
        raw_output: The raw output from the model
        
    Returns:
        str: The cleaned, de-identified text only
    """
    pre_text_markers = [
        "TESTO DE-IDENTIFICATO:",
        "testo de-identificato:",
        "Ecco il testo",
        "ecco il testo",
    ]
    
    # Remove the breakpoint() call
    # Remove thinking if present
    #breakpoint()
    if "<think>" in raw_output:
        raw_output = raw_output.split("</think>")[1].strip()

    # Remove any explanations or comments before the actual de-identified text with any possible capitalization
    for marker in pre_text_markers:
        if marker in raw_output:
            # Extract text after the marker
            cleaned_output = raw_output.split(marker)[1].strip()
            break
    else:
        cleaned_output = raw_output.strip()
    
    # Remove any explanations after the de-identified text (often starts with phrases like "Ho sostituito")
    explanations_markers = [
        "Ho sostituito",
        "è stato sostituito", 
        "Ecco il testo", 
        "Come richiesto", 
        "Questo è il", 
        "Nota de-identificata:",
        "Ho de-identificato",
        "Note",
        "note",
        "Nota",
        "nota"
    ]
    
    for marker in explanations_markers:
        if marker in cleaned_output:
            # Keep only text before the explanation starts
            marker_pos = cleaned_output.find(marker)
            if marker_pos > 0:  # Ensure there's actual content before the marker
                cleaned_output = cleaned_output[:marker_pos].strip()
    
    # Remove any wrapping quotes that might be present
    if cleaned_output.startswith('"') and cleaned_output.endswith('"'):
        cleaned_output = cleaned_output[1:-1]
    
    # Remove any "```" markdown code blocks that might be present
    if cleaned_output.startswith("```") and "```" in cleaned_output[3:]:
        first_marker = cleaned_output.find("```")
        second_marker = cleaned_output.find("```", first_marker + 3)
        if second_marker > first_marker:
            # Extract text between the markers
            cleaned_output = cleaned_output[first_marker+3:second_marker].strip()
    
    return cleaned_output

def estimate_tokens(text: str) -> int:
    """
    Roughly estimate the number of tokens in a text.
    This is a very basic estimator using word count as a proxy.
    For more accurate estimation, one would need to use the model's tokenizer.
    
    Args:
        text: Input text to estimate token count for
        
    Returns:
        int: Estimated number of tokens
    """
    # A very rough estimate: approximately 1.3 tokens per word for most languages
    words = re.findall(r'\b\w+\b', text)
    return int(len(words) * 1.3) + 20  # Add a small buffer

def process_clinical_notes(
    notes: List[str], 
    output_dir: str, 
    backend: ModelBackend,
    max_failures: int = 3,
    clean_output: bool = True,
    include_prompt: bool = False,
    model_name: str = "llama3",
    backend_name: str = "ollama"
) -> Tuple[int, int]:
    """
    Process the clinical notes from the input file and write de-identified notes to the output file
    
    Args:
        notes: List of clinical notes to be de-identified
        output_file: Path to the output file where de-identified notes will be written
        backend: The model backend to use for inference
        max_failures: Maximum number of consecutive failures before aborting
        clean_output: Whether to clean and extract only the de-identified text
        include_prompt: Whether to include the prompt in the output JSONL
    
    Returns:
        Tuple[int, int]: (number of processed notes, number of failed notes)
    """
    processed_count = 0
    failed_count = 0
    consecutive_failures = 0
    output_file = os.path.join(output_dir, f"{model_name}_clean_output={clean_output}.jsonl")
    start_time = time.time()
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, note in enumerate(notes, 1):
            
            if not note.strip():
                continue
            
            # Estimate token count and warn if it might exceed limits
            input_token_estimate = estimate_tokens(note)
            max_tokens = getattr(backend, 'max_tokens', 8192)
            if input_token_estimate > (max_tokens // 2):
                logger.warning(f"Note {i} is estimated to be {input_token_estimate} tokens, which might be too large for the limit of {max_tokens} output tokens. Output might be truncated.")
            
            logger.info(f"Processing note {i}/{len(notes)}... (est. {input_token_estimate} tokens)")
            
            # Create prompt and generate de-identified text
            prompt = create_deidentification_prompt(note)
            raw_output = backend.generate(prompt)
            
            # Check if the output might be truncated
            if len(raw_output) > 25000:  # Roughly 5,000 tokens
                logger.warning(f"Note {i} generated a very large output ({len(raw_output)} characters). It might be truncated.")
            
            # Clean and extract only the de-identified text
            if clean_output:
                deidentified_note = clean_model_output(raw_output)
            else:
                deidentified_note = raw_output
            
            # Check if output is significantly shorter than input (might indicate truncation)
            if len(deidentified_note) < len(note) * 0.7:
                logger.warning(f"Note {i} de-identified output is much shorter than input. Possible truncation or incomplete processing.")
            
            # Log a sample of original and de-identified text for comparison if it's the first note
            if i == 1:
                logger.info(f"Sample original (first 100 chars): {note[:100]}...")
                logger.info(f"Sample de-identified (first 100 chars): {deidentified_note[:100]}...")
            
            # Create a dictionary and write it as a JSON line
            result = {
                "input": note,
                "output": deidentified_note
            }
            
            # Include prompt if requested
            if include_prompt:
                result["prompt"] = prompt
                
            json.dump(result, outfile, ensure_ascii=False)
            outfile.write('\n')
            
            processed_count += 1
            consecutive_failures = 0
            
            # Log progress
            if processed_count % 5 == 0 or processed_count == len(notes):
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                logger.info(f"Processed {processed_count}/{len(notes)} notes ({rate:.2f} notes/sec)")
    
    return processed_count, failed_count

def preprocess_clinical_notes_from_txt(
    input_file: str
) -> List[str]:
    """
    Process the clinical notes from the input file
    
    Args:
        input_file: Path to the input txt file containing clinical notes.
        The file format assumes that each paragraph represents a clinical note,
        with empty lines separating the notes.
        
    Returns:
        List[str]: A list of clinical notes
    """
    logger.info(f"Reading clinical notes from {input_file}")
    
    notes = []
    current_note = []
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            
            if line:  # Non-empty line
                current_note.append(line)
            elif current_note:  # Empty line and we have content in current_note
                notes.append(" ".join(current_note))
                current_note = []
    
    # Don't forget to add the last note if the file doesn't end with an empty line
    if current_note:
        notes.append(" ".join(current_note))
    
    logger.info(f"Found {len(notes)} clinical notes")
    return notes

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="De-identify Italian clinical notes using LLMs")
    parser.add_argument("--input", required=True, help="Input file with clinical notes")
    parser.add_argument("--output_dir", required=True, help="Output directory for de-identified notes")
    parser.add_argument("--model", default="llama3", help="Model name to use for de-identification")
    parser.add_argument("--backend", choices=["ollama", "vllm"], default="ollama", 
                        help="Backend to use for model inference")
    parser.add_argument("--format", choices=["txt", "csv"], default="txt",
                        help="Format of the input file (txt = one note per paragraph, csv = one note per row)")
    parser.add_argument("--clean_output", action="store_true", default=False,
                        help="Clean the output of the model")
    parser.add_argument("--max_tokens", type=int, default=8192,
                       help="Maximum number of tokens to generate for each note (default: 8192)")
    parser.add_argument("--include_prompt", action="store_true", default=False,
                       help="Include the prompt in the output JSONL file (useful for debugging)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for text generation (default: 0.1, higher = more creative, lower = more deterministic)")
    
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return 1
    
    # Validate max_tokens parameter
    if args.max_tokens < 1000:
        logger.warning(f"Specified max_tokens ({args.max_tokens}) is very low. This might lead to truncated outputs.")
    elif args.max_tokens > 16384:
        logger.warning(f"Specified max_tokens ({args.max_tokens}) is very high. Some models may not support this.")
    
    # Validate temperature parameter
    if args.temperature < 0.0:
        logger.error(f"Temperature cannot be negative. Using default of 0.7 instead.")
        args.temperature = 0.7
    elif args.temperature > 1.5:
        logger.warning(f"Very high temperature ({args.temperature}) may lead to unreliable de-identification.")
    elif args.temperature == 0.0:
        logger.info("Temperature set to 0.0 (fully deterministic mode)")
    
    
    # Create backend
    try:
        if args.backend == "ollama":
            backend = OllamaBackend(args.model, max_tokens=args.max_tokens, temperature=args.temperature)
        elif args.backend == "vllm":
            backend = VLLMBackend(args.model, max_tokens=args.max_tokens, temperature=args.temperature)
        else:
            logger.error(f"Unknown backend: {args.backend}")
            return 1
    except BackendNotAvailableError as e:
        logger.error(f"Backend error: {e}")
        return 1
    
    logger.info(f"Starting de-identification process")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Output format: JSONL (JSON Lines) with input, prompt, and output")
    logger.info(f"Max tokens per generation: {args.max_tokens}")
    logger.info(f"Temperature: {args.temperature}")
    
    start_time = time.time()
    
    # Process notes based on input format
    if args.format == "txt":
        notes = preprocess_clinical_notes_from_txt(args.input)
    else:  # csv format
        # Open CSV and read notes from the first column
        notes = []
        with open(args.input, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            # Skip header
            next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    notes.append(row[0])
    
    # Process the notes
    processed, failed = process_clinical_notes(
        notes, 
        args.output_dir, 
        backend, 
        clean_output=args.clean_output,
        include_prompt=args.include_prompt,
        model_name=args.model,
        backend_name=args.backend
    )
    elapsed = time.time() - start_time
    
    logger.info(f"De-identification complete")
    logger.info(f"Processed notes: {processed}")
    logger.info(f"Failed notes: {failed}")
    logger.info(f"Total time: {elapsed:.2f} seconds")
    
    if processed > 0:
        logger.info(f"Average processing time: {elapsed/processed:.2f} seconds per note")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
