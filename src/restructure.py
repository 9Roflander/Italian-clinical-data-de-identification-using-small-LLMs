import os
import glob

def main():
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('src', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Move data files
    data_files = ['synthetic_clinical_1000.json', 'gold_standard_80.json']
    for f in data_files:
        if os.path.exists(f):
            os.rename(f, os.path.join('data', f))

    # Move source files
    src_files = ['explore_data.py', 'train_cpt.py', 'train_phase2_sft.py', 'train_evaluate_cv.py', 'test_inference.py']
    for f in src_files:
        if os.path.exists(f):
            os.rename(f, os.path.join('src', f))

    # Update paths in all python scripts inside src/
    replacements = {
        'gold_standard_80.json': '../data/gold_standard_80.json',
        'synthetic_clinical_1000.json': '../data/synthetic_clinical_1000.json',
        './llama-3.2-3b-dart-sft': '../models/llama-3.2-3b-dart-sft',
        './temp_merged_phase1': '../models/temp_merged_phase1',
        './cv_fold_outputs': '../models/cv_fold_outputs',
        './llama-3.2-3b-deid-sft-v2': '../models/llama-3.2-3b-deid-sft-v2',
        './llama-3.2-3b-deid-sft': '../models/llama-3.2-3b-deid-sft'
    }

    for script in glob.glob('src/*.py'):
        with open(script, 'r', encoding='utf-8') as file:
            content = file.read()
        
        for old_str, new_str in replacements.items():
            content = content.replace(old_str, new_str)
            # Also handle without ./ just in case
            content = content.replace(old_str.replace('./', ''), new_str.replace('../', '../'))
            
        with open(script, 'w', encoding='utf-8') as file:
            file.write(content)

if __name__ == "__main__":
    main()
