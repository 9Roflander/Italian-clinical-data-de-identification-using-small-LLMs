import os
import json
import re
import torch
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import KFold
from tqdm import tqdm

# Constants
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B"
PHASE1_ADAPTER_DIR = "./llama-3.2-3b-dart-sft"
TEMP_MERGED_DIR = "./temp_merged_phase1"
OUTPUT_DIR_BASE = "./cv_fold_outputs"

SYSTEM_PROMPT = (
    "Sei un assistente medico esperto specializzato in de-identificazione. "
    "Estrai tutte le informazioni sensibili dal seguente referto clinico. "
    "Restituisci ESCLUSIVAMENTE un array JSON di oggetti, dove ogni oggetto ha due chiavi: "
    "'text' (l'entità sensibile con 2-3 parole di contesto circostante per disambiguare) e "
    "'type' (rigorosamente uno tra NOME, ETÀ, DATA, LUOGO/INDIRIZZO). Se non ci sono entità, restituisci []."
)

VALID_TYPES = {"NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"}

# --- Data Utilities ---
def sanitize_type(etype):
    etype = etype.upper()
    if "LUOGO" in etype or "INDIRIZZO" in etype:
        return "LUOGO/INDIRIZZO"
    if etype in VALID_TYPES:
        return etype
    return None

def extract_with_context_by_index(text, start, end, padding=20):
    start_idx = max(0, start - padding)
    end_idx = min(len(text), end + padding)
    return text[start_idx:end_idx].strip()

def extract_with_context_by_search(text, ent_text, padding=20):
    idx = text.find(ent_text)
    if idx != -1:
        return extract_with_context_by_index(text, idx, idx + len(ent_text), padding)
    return ent_text

def format_records(data_list, has_indices=False):
    formatted = []
    for row in data_list:
        text = row.get("original_text", row.get("text", ""))
        entities = row.get("entities", [])
        
        extracted = []
        for ent in entities:
            etype = sanitize_type(ent.get("type", ""))
            if not etype: continue
            
            if has_indices and "start" in ent and "end" in ent:
                c_str = extract_with_context_by_index(text, ent["start"], ent["end"])
            else:
                c_str = extract_with_context_by_search(text, ent.get("text", ""))
            
            extracted.append({"text": c_str, "type": etype})
            
        formatted.append({
            "system": SYSTEM_PROMPT,
            "input": text,
            "output": json.dumps(extracted, ensure_ascii=False)
        })
    return formatted

def load_all_data():
    with open('./data/gold_standard_80.json', 'r', encoding='utf-8') as f:
        gold = json.load(f)

    with open('./data/synthetic_clinical_1000.json', 'r', encoding='utf-8') as f:
        synth = json.load(f)
        
    crf_ds = load_dataset('NLP-FBK/synthetic-crf-train', split='it')
    crf = [{"system": SYSTEM_PROMPT, "input": x["clinical_note"], "output": "[]"} for x in crf_ds]
    
    return gold, synth, crf

def get_train_dataset(train_gold, synth, crf):
    gold_fmt = format_records(train_gold, has_indices=False)
    synth_fmt = format_records(synth, has_indices=True)

    messages_data = [
        {"messages": [
            {"role": "system", "content": ex["system"]},
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]}
        ]}
        for ex in gold_fmt + synth_fmt + crf
    ]
    return Dataset.from_list(messages_data)

# --- Model setup ---
def prepare_base_merged_model():
    if not os.path.exists(TEMP_MERGED_DIR):
        print("Merging phase 1 adapter into base model...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cpu")
        model = PeftModel.from_pretrained(model, PHASE1_ADAPTER_DIR).merge_and_unload()
        model.save_pretrained(TEMP_MERGED_DIR)
        tokenizer.save_pretrained(TEMP_MERGED_DIR)
    
LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)

def get_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TEMP_MERGED_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(TEMP_MERGED_DIR, quantization_config=quant_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM", bias="none"
    )
    return get_peft_model(model, lora_cfg), tokenizer

# --- Evaluator ---
def calculate_metrics(y_true, y_pred):
    # Simplified string-level strict entity match metric per note
    tp = sum(1 for p in y_pred if p in y_true)
    fp = sum(1 for p in y_pred if p not in y_true)
    fn = sum(1 for t in y_true if t not in y_pred)
    return tp, fp, fn

def evaluate_fold(model, tokenizer, test_data):
    model.eval()
    tp_total, fp_total, fn_total = 0, 0, 0
    
    for row in tqdm(test_data, desc="Evaluating Fold"):
        text = row.get("text", "")
        true_ents = [sanitize_type(e.get("type", "")) + ":" + e.get("text", "") for e in row.get("entities", []) if sanitize_type(e.get("type", ""))]
        
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0)
            
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON
        pred_ents = []
        try:
            # find JSON array
            match = re.search(r'\[.*\]', gen_text.replace('\n', ' '))
            if match:
                parsed = json.loads(match.group(0))
                for item in parsed:
                    etype = sanitize_type(item.get("type", ""))
                    snippet = item.get("text", "")
                    if etype and snippet:
                        # Extract exact entity from snippet inside logic (simplified to snippet matching here)
                        pred_ents.append(etype + ":" + snippet)
        except Exception:
            pass # Failed to parse JSON
            
        tp, fp, fn = calculate_metrics(true_ents, pred_ents)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return prec, rec, f1

def run_cv():
    prepare_base_merged_model()
    gold, synth, crf = load_all_data()
    gold_arr = np.array(gold)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(gold_arr)):
        print(f"\n{'='*30}\nStarting Fold {fold + 1}/5\n{'='*30}")
        
        train_gold = gold_arr[train_idx].tolist()
        test_gold = gold_arr[test_idx].tolist()
        
        model, tokenizer = get_model_and_tokenizer()
        train_ds = get_train_dataset(train_gold, synth, crf)

        out_dir = f"{OUTPUT_DIR_BASE}/fold_{fold+1}"
        args = SFTConfig(
            output_dir=out_dir, per_device_train_batch_size=2, gradient_accumulation_steps=8,
            learning_rate=5e-5, lr_scheduler_type="cosine", warmup_ratio=0.05, num_train_epochs=3,
            bf16=True, logging_steps=10, save_strategy="no",
            dataset_text_field="messages", max_length=2048, assistant_only_loss=True
        )

        trainer = SFTTrainer(
            model=model, train_dataset=train_ds, args=args,
            processing_class=tokenizer
        )
        trainer.train()
        
        # Evaluate
        p, r, f1 = evaluate_fold(model, tokenizer, test_gold)
        print(f"Fold {fold+1} Metrics: P={p:.4f}, R={r:.4f}, F1={f1:.4f}")
        fold_metrics.append((p, r, f1))
        
        # Cleanup memory heavily between folds
        del model
        del trainer
        del tokenizer
        torch.cuda.empty_cache()
        
    # Global metrics
    avg_p = np.mean([m[0] for m in fold_metrics])
    avg_r = np.mean([m[1] for m in fold_metrics])
    avg_f1 = np.mean([m[2] for m in fold_metrics])
    print("\n" + "="*30)
    print(f"Final Cross-Validation Results (k=5):")
    print(f"Average Precision : {avg_p:.4f}")
    print(f"Average Recall    : {avg_r:.4f}")
    print(f"Average F1-score  : {avg_f1:.4f}")
    print("="*30)

if __name__ == "__main__":
    run_cv()