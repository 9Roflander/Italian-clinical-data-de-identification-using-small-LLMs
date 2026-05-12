import json
import pprint
import sys

def explore():
    print("="*50)
    print("1. SYNTHETIC RECORDS (1,000 generated)")
    print("="*50)
    try:
        with open('../data/../data/synthetic_clinical_1000.json', 'r', encoding='utf-8') as f:
            synthetic_data = json.load(f)
        print("Keys/Columns:", list(synthetic_data[0].keys()))
        print("Sample Row:")
        pprint.pprint(synthetic_data[0])
    except Exception as e:
        print(f"Error reading synthetic data: {e}")
    print("\n")

    print("="*50)
    print("2. GOLD STANDARD RECORDS (80 hand-annotated)")
    print("="*50)
    try:
        with open('../data/../data/gold_standard_80.json', 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        print("Keys/Columns:", list(gold_data[0].keys()))
        print("Sample Row:")
        pprint.pprint(gold_data[0])
    except Exception as e:
        print(f"Error reading gold standard data: {e}")
    print("\n")

    print("="*50)
    print("3. CRF DATASET (HF: NLP-FBK/synthetic-crf-train)")
    print("="*50)
    try:
        from datasets import load_dataset
        crf_dataset = load_dataset('NLP-FBK/synthetic-crf-train', split='train')
        print("Keys/Columns:", crf_dataset.column_names)
        print("Sample Row:")
        pprint.pprint(crf_dataset[0])
    except ImportError:
        print("Error: 'datasets' library is not installed. Please install it using 'pip install datasets'.")
    except Exception as e:
        print(f"Error reading CRF dataset: {e}")
    print("\n")

if __name__ == "__main__":
    explore()