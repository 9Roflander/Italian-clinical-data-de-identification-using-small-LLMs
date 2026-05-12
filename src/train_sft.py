import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def format_instruction(example):
    """
    Formats the DART dataset into a Medical ChatML / Instruction prompt 
    mapping Drug Names to their Therapeutic Indications.
    """
    drug_name = example.get('Nome Medicinale', 'Farmaco Sconosciuto')
    indications = example.get('04.1 Indicazioni terapeutiche', '')
    
    prompt = (
        "Sei un assistente medico farmacologico esperto. "
        f"Descrivi le indicazioni terapeutiche per il seguente farmaco: {drug_name}\n\n"
        f"### Indicazioni Terapeutiche:\n{indications}"
    )
    return {"text": prompt}

def load_data():
    from datasets import load_dataset
    print("Loading DART pharmacological dataset from Hugging Face...")
    
    # Load the DART dataset directly from HF
    dataset = load_dataset("praiselab-picuslab/DART", split="DART")
    
    # Filter out rows where Indications are empty
    dataset = dataset.filter(lambda x: x['04.1 Indicazioni terapeutiche'] is not None and len(x['04.1 Indicazioni terapeutiche'].strip()) > 10)
    
    # Since 16000 rows is huge for a quick SFT test, we'll take a subset of 1000 rows
    dataset = dataset.select(range(min(1000, len(dataset))))
    
    # Format instructions
    dataset = dataset.map(format_instruction)
    
    # Split into train/validation
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train size: {len(dataset['train'])}, Eval size: {len(dataset['test'])}")
    
    return dataset

def main():
    # You can point this to your CPT output folder if you want to resume from Phase 1,
    # e.g., model_id = "./llama-3.2-3b-italian-medical-cpt"
    # We will use the base model here for the supervised fine-tuning phase.
    model_id = "meta-llama/Llama-3.2-3B"
    output_dir = "./llama-3.2-3b-dart-sft"
    
    # 1. Load Data
    dataset = load_data()
    
    # 2. Configure Quantization (4-bit)
    compute_dtype = getattr(torch, "bfloat16") if torch.cuda.is_bf16_supported() else getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )
    
    print("Initializing Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False 

    # 3. Configure LoRA for SFT
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 4. Define Training Arguments using SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        optim="paged_adamw_32bit",
        learning_rate=2e-5, # Lower learning rate for SFT compared to CPT
        lr_scheduler_type="cosine",
        num_train_epochs=2,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        report_to="none"
    )
    
    # 5. Setup SFT Trainer
    print("Starting SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )
    
    trainer.train()
    
    print(f"Training Complete! Saving SFT adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
