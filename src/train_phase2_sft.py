import os
import json
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# --- 1. Configuration & Constants ---
BASE_MODEL_ID = "meta-llama/Llama-3.2-3B"
PHASE1_ADAPTER_DIR = "./llama-3.2-3b-dart-sft"
TEMP_MERGED_DIR = "./temp_merged_phase1"
OUTPUT_DIR = "./llama-3.2-3b-deid-sft"

LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)

SYSTEM_PROMPT = (
    "Sei un assistente medico esperto specializzato in de-identificazione. "
    "Estrai tutte le informazioni sensibili dal seguente referto clinico. "
    "Restituisci ESCLUSIVAMENTE un array JSON di oggetti, dove ogni oggetto ha due chiavi: "
    "'text' (l'entità sensibile con 2-3 parole di contesto circostante per disambiguare) e "
    "'type' (rigorosamente uno tra NOME, ETÀ, DATA, LUOGO/INDIRIZZO). Se non ci sono entità, restituisci []."
)

VALID_TYPES = {"NOME", "ETÀ", "DATA", "LUOGO/INDIRIZZO"}

# --- 2. Data Preprocessing Functions ---
def sanitize_type(etype):
    # Mapping to strictly expected valid types
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

def process_synthetic_records(file_path):
    print("Processing Synthetic Records...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    formatted_data = []
    for row in data:
        original_text = row.get("original_text", "")
        entities = row.get("entities", [])
        
        extracted_entities = []
        for ent in entities:
            etype = sanitize_type(ent.get("type", ""))
            if not etype:
                continue
                
            start = ent.get("start")
            end = ent.get("end")
            if start is not None and end is not None:
                context_str = extract_with_context_by_index(original_text, start, end)
                extracted_entities.append({"text": context_str, "type": etype})
                
        formatted_data.append({
            "system": SYSTEM_PROMPT,
            "input": original_text,
            "output": json.dumps(extracted_entities, ensure_ascii=False)
        })
    return Dataset.from_list(formatted_data)

def process_gold_standard(file_path):
    print("Processing Gold Standard Records...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    formatted_data = []
    padding = 20
    for row in data:
        text = row.get("text", "")
        entities = row.get("entities", [])
        
        extracted_entities = []
        for ent in entities:
            etype = sanitize_type(ent.get("type", ""))
            if not etype:
                continue
                
            ent_text = ent.get("text", "")
            idx = text.find(ent_text)
            
            if idx != -1:
                context_str = extract_with_context_by_index(text, idx, idx + len(ent_text), padding)
            else:
                # Fallback if not found exactly
                context_str = ent_text
                
            extracted_entities.append({"text": context_str, "type": etype})
            
        formatted_data.append({
            "system": SYSTEM_PROMPT,
            "input": text,
            "output": json.dumps(extracted_entities, ensure_ascii=False)
        })
    return Dataset.from_list(formatted_data)

def process_crf_dataset():
    print("Processing CRF Dataset (Negative Samples)...")
    dataset = load_dataset('NLP-FBK/synthetic-crf-train', split='it')
    
    formatted_data = []
    for row in dataset:
        clinical_note = row.get("clinical_note", "")
        formatted_data.append({
            "system": SYSTEM_PROMPT,
            "input": clinical_note,
            "output": "[]" # Always empty array for negative sampling
        })
    return Dataset.from_list(formatted_data)

def assemble_dataset(tokenizer):
    ds1 = process_synthetic_records("./data/synthetic_clinical_1000.json")
    ds2 = process_gold_standard("./data/gold_standard_80.json")
    ds3 = process_crf_dataset()

    combined_ds = concatenate_datasets([ds1, ds2, ds3])

    def apply_template(example):
        msgs = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        return {"text": tokenizer.apply_chat_template(msgs, tokenize=False)}

    return combined_ds.map(apply_template, remove_columns=combined_ds.column_names)

# --- 3. Model & Training Setup ---
def setup_and_train():
    print("Loading Base Model for Merging...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("Loading and Merging Phase 1 Adapter...")
    peft_model = PeftModel.from_pretrained(base_model, PHASE1_ADAPTER_DIR)
    merged_model = peft_model.merge_and_unload()
    
    print(f"Saving temporary merged model to {TEMP_MERGED_DIR}...")
    merged_model.save_pretrained(TEMP_MERGED_DIR)
    tokenizer.save_pretrained(TEMP_MERGED_DIR)
    
    del merged_model
    del peft_model
    del base_model
    torch.cuda.empty_cache()
    
    print("Reloading Merged Model in 4-bit Quantization...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        TEMP_MERGED_DIR,
        quantization_config=quant_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    
    print("Preparing Phase 2 SFT LoRA Config...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    
    print("Preparing Dataset...")
    full_dataset = assemble_dataset(tokenizer)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_grad_norm=1.0,
        dataset_text_field="text",
        max_length=2048,
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=full_dataset,
        args=training_args,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    
    print("Starting SFT Phase 2 Training...")
    trainer.train()
    
    print("Training Complete. Saving Phase 2 Adapter...")
    trainer.save_model(OUTPUT_DIR)
    
if __name__ == "__main__":
    setup_and_train()