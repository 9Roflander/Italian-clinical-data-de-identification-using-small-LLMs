import json
import argparse
from collections import defaultdict
import pandas as pd
import csv
import os
import glob
from collections import Counter


#choose GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def get_entity_key(entity):
    """
    Create a unique key for an entity based on its text and type.
    """
    return f"{entity['text']}_{entity['type']}"

def get_majority_vote(votes, threshold=0.5):
    """
    Determine if there is a majority vote for an entity's classification.
    
    Args:
        votes (list): List of classifications (TP, FP, FN)
        threshold (float): Minimum proportion required for majority
        
    Returns:
        str or None: The majority classification or None if no majority
    """
    if not votes:
        return None
    #breakpoint()
    vote_counts = Counter(votes)
    total_votes = len(votes)
    
    # Find the most common vote and its count
    most_common = vote_counts.most_common(1)[0]
    classification, count = most_common
    
    # Check if the proportion meets the threshold
    if count / total_votes > threshold:
        return classification
    return None

def aggregate_llm_judgments(eval_files):
    """
    Aggregate judgments from multiple LLM evaluations.
    
    Args:
        eval_files (list): List of evaluation file paths
        
    Returns:
        dict: Dictionary containing aggregated judgments per report
    """
    aggregated_judgments = defaultdict(lambda: defaultdict(list))
    
    for eval_file in eval_files:
        jsonObj = pd.read_json(path_or_buf=eval_file, lines=True)
        data = []
        #breakpoint()
        for sample in jsonObj[0]:
            try:
                data.append(json.loads(sample))
            except Exception as e:
                print(f"Error loading JSON: {e}")
                continue
            
        for record in data:
            report_id = record['report_id']
            
            # Store gold annotations for reference
            if 'annotations_gold' in record:
                aggregated_judgments[report_id]['gold_annotations'] = record['annotations_gold']
            
            # Aggregate deidentified annotations
            for annotation in record.get('annotations_deidentified', []):
                print(annotation)
                entity_key = get_entity_key(annotation)
                if entity_key not in aggregated_judgments[report_id]:
                    aggregated_judgments[report_id][entity_key] = []
                
                aggregated_judgments[report_id][entity_key].append(annotation['counted_as'])
    
    return aggregated_judgments

def compute_metrics_with_majority_voting(eval_files, majority_threshold=0.5):
    """
    Compute evaluation metrics using majority voting across multiple LLM judgments.
    
    Args:
        eval_files (list): List of evaluation file paths
        majority_threshold (float): Minimum proportion required for majority
        
    Returns:
        dict: Dictionary containing precision, recall, F1 score, and other metrics
    """
    # Initialize counters
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_discarded = 0
    
    # Aggregate judgments from all LLM evaluations
    aggregated_judgments = aggregate_llm_judgments(eval_files)
    
    # Process each report's judgments
    for report_id, report_data in aggregated_judgments.items():
        
        
        # Process each entity's judgments
        for entity_key, votes in report_data.items():
            #breakpoint()
            if entity_key != "gold_annotations":
                majority = get_majority_vote(votes, majority_threshold)
                
                if majority is None:
                    total_discarded += 1
                    continue
                    
                if majority == "TP":
                    total_tp += 1
                elif majority == "FP":
                    total_fp += 1
                elif majority == "FN":
                    total_fn += 1
    
    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "discarded": total_discarded
    }

def extract_model_info(filename):
    """
    Extract deidentifier and evaluator model information from filename
    
    Args:
        filename (str): Name of the evaluation file
        
    Returns:
        tuple: (deidentifier_model, evaluator_model, entity_type)
    """
    # Example filename: gemma3:12b_clean_output=True.jsonl_evaluatedBygemma3:27b_NOME.jsonl
    deidentifier = filename.split("_")[0]
    evaluator = filename.split("_")[3].split("evaluatedBy")[1].split("_")[0]
    entity_type = filename.split("_")[4].split(".")[0]
    
    return deidentifier, evaluator, entity_type

def group_eval_files(eval_files):
    """
    Group evaluation files by deidentifier and entity type.
    
    Args:
        eval_files (list): List of evaluation file paths
        
    Returns:
        dict: Dictionary of grouped files
    """
    grouped_files = defaultdict(list)
    
    for eval_file in eval_files:
        filename = os.path.basename(eval_file)
        deidentifier, _, entity_type = extract_model_info(filename)
        key = (deidentifier, entity_type)
        grouped_files[key].append(eval_file)
    
    return grouped_files

def save_combined_metrics_to_csv(all_metrics, output_file):
    """
    Save combined metrics from all files to a CSV file
    
    Args:
        all_metrics (list): List of dictionaries containing metrics and model info
        output_file (str): Path to the output CSV file
    """
    # Create metrics directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['deidentifier_model', 'entity_type', 
                     'precision', 'recall', 'f1', 'accuracy', 
                     'tp', 'fp', 'fn', 'discarded']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for metric_data in all_metrics:
            writer.writerow(metric_data)

def main():
    parser = argparse.ArgumentParser(description='Compute NER evaluation metrics from evaluation files')
    parser.add_argument('--outputs-dir', default='outputs', help='Directory containing evaluation files')
    parser.add_argument('--output-csv', default='metrics/combined_metrics.csv', help='Path to save combined metrics in CSV format')
    parser.add_argument('--majority-threshold', type=float, default=0.3, help='Minimum proportion required for majority voting')
    args = parser.parse_args()
    
    # Find all evaluation files
    eval_files = glob.glob(os.path.join(args.outputs_dir, '*evaluatedBy*.jsonl'))
    if not eval_files:
        print(f"No evaluation files found in {args.outputs_dir}")
        return
    
    # Group files by deidentifier and entity type
    grouped_files = group_eval_files(eval_files)
    
    all_metrics = []
    
    # Process each group of files
    for (deidentifier, entity_type), files in grouped_files.items():
        print(f"\nProcessing files for {deidentifier} - {entity_type}")
        print(f"Number of LLM judges: {len(files)}")
        
        metrics = compute_metrics_with_majority_voting(files, args.majority_threshold)
        
        # Combine metrics with model info
        metrics.update({
            'deidentifier_model': deidentifier,
            'entity_type': entity_type
        })
        
        all_metrics.append(metrics)
        
        # Print individual metrics
        print("\n===== METRICS =====")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"True Positives: {metrics['tp']}")
        print(f"False Positives: {metrics['fp']}")
        print(f"False Negatives: {metrics['fn']}")
        print(f"Discarded (no majority): {metrics['discarded']}")
    
    # Save combined metrics to CSV
    save_combined_metrics_to_csv(all_metrics, args.output_csv)
    print(f"\nCombined metrics saved to {args.output_csv}")

if __name__ == "__main__":
    main()
