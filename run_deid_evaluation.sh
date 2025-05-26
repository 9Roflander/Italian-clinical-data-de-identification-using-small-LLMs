#!/bin/bash

# Exit on error
set -e

# Create output directory if it doesn't exist
mkdir -p outputs
mkdir -p metrics

#choose the gpu
export CUDA_VISIBLE_DEVICES=0
# Define input file and output directory
INPUT_FILE="samples.csv"
ANNOTATION_FILE="Annotated clinical notes - samples.csv"
OUTPUT_DIR="outputs"
METRICS_DIR="metrics"

# Define list of models to use for de-identification
#DEID_MODELS=("llama3.2:1b" "llama3.2:3b" "gemma3:1b" "gemma3:4b" "gemma3:12b" "gemma3:27b" "mistral:7b" "phi4:14b")
DEID_MODELS=("gemma3:4b" "gemma3:12b")
# Define the judge models to use for evaluation
JUDGE_MODELS=("gemma3:27b" "mistral-small:24b" "deepseek-r1:32b")

# Define the categories for evaluation
#CATEGORIES=("NOME" "ETÀ" "LUOGO/INDIRIZZO" "DATA")
CATEGORIES=("ETÀ")
# Step 1: Run de-identification with different models
echo "=== Running de-identification with multiple models ==="
for MODEL in "${DEID_MODELS[@]}"; do
  echo "Processing with model: $MODEL"
  python deid.py --input "$INPUT_FILE" --output_dir "$OUTPUT_DIR" --model "$MODEL" --backend ollama --format csv --test_length 1  # --clean_output
done

# Step 2: Evaluate the de-identified output with different judge models
echo "=== Evaluating de-identification results ==="
for DEID_MODEL in "${DEID_MODELS[@]}"; do
  # Determine the output file name pattern from deid.py
  DEIDENTIFIED_FILE="${OUTPUT_DIR}/${DEID_MODEL}_clean_output=False.jsonl"
  
  # Check if the file exists
  if [ ! -f "$DEIDENTIFIED_FILE" ]; then
    echo "Warning: Deidentified file $DEIDENTIFIED_FILE not found. Skipping evaluation."
    continue
  fi
  
  echo "Evaluating results from model: $DEID_MODEL"
  
  for JUDGE_MODEL in "${JUDGE_MODELS[@]}"; do
    echo "  Using judge model: $JUDGE_MODEL"
    
    for CATEGORY in "${CATEGORIES[@]}"; do
      echo "    Evaluating category: $CATEGORY"
      echo "    Deidentified file: $DEIDENTIFIED_FILE"
      python llm_as_a_judge.py \
        --model "$JUDGE_MODEL" \
        --category "$CATEGORY" \
        --deidentified_data_path "$DEIDENTIFIED_FILE"
    done
  done
done

# Step 3: Compute metrics for all evaluation results
echo "=== Computing metrics for all evaluation results ==="
for DEID_MODEL in "${DEID_MODELS[@]}"; do
  for JUDGE_MODEL in "${JUDGE_MODELS[@]}"; do
    for CATEGORY in "${CATEGORIES[@]}"; do
      # Sanitize category name for filename
      SAFE_CATEGORY=$(echo "$CATEGORY" | tr '/' '_')
      #EVAL_FILE="${DEID_MODEL}_clean_output=False_evaluatedBy${JUDGE_MODEL}_${SAFE_CATEGORY}.jsonl"
      #EVAL_PATH="${OUTPUT_DIR}/${EVAL_FILE}"
      
      #if [ -f "$EVAL_PATH" ]; then
      echo "Computing metrics"
      python compute_metrics.py --outputs-dir "$OUTPUT_DIR" > "${OUTPUT_DIR}/metrics_${DEID_MODEL}_${JUDGE_MODEL}_${SAFE_CATEGORY}.txt"
      #else
      #  echo "Warning: Evaluation file $EVAL_PATH not found. Skipping metrics computation."
      #fi
    done
  done
done

# Step 4: Generate combined metrics CSV
echo "=== Generating combined metrics CSV ==="
# Generate timestamp for the metrics file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
METRICS_FILE="${METRICS_DIR}/combined_metrics_${TIMESTAMP}.csv"

# Run the updated compute_metrics.py script to generate combined CSV
python compute_metrics.py --outputs-dir "$OUTPUT_DIR" --output-csv "$METRICS_FILE"

echo "=== All processing complete ==="
echo "Individual results are saved in the $OUTPUT_DIR directory"
echo "Combined metrics are saved in $METRICS_FILE" 