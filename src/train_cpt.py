import os
import torch
import itertools
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    default_data_collator,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- Function 1: Data Ingestion and Preprocessing ---
def load_and_prepare_data(tokenizer, max_seq_length=2048):
    """
    Loads datastes from Hugging Face, extracts relevant unannotated text,
    and returns a tokenized, sequence-packed dataset for Causal LM training.
    """
    print("Loading Dyspnea Clinical Notes...")
    # Using the Italian split of the clinical notes
    clinical_ds = load_dataset("NLP-FBK/dyspnea-clinical-notes", split="it")
    
    # Extract raw text into a uniform 'text' column
    def extract_clinical_text(example):
        return {"text": example["clinical_note"]}
    
    clinical_ds = clinical_ds.map(extract_clinical_text, remove_columns=clinical_ds.column_names)

    print("Loading DART Pharmacological dataset...")
    # Load DART dataset (adjust split name if necessary based on the dataset structure)
    try:
        dart_ds = load_dataset("praiselab-picuslab/DART", split="train")
    except Exception as e:
        print(f"Could not load train split, attempting default split. Error: {e}")
        dart_ds = load_dataset("praiselab-picuslab/DART")
        # Get the first split available
        first_split = list(dart_ds.keys())[0]
        dart_ds = dart_ds[first_split]

    # Map DART fields into a single text string
    # Assuming standard European SmPC section names or generalized column mapping.
    # We combine them into a single string per document context.
    def extract_dart_text(example):
        sections = [
            "Therapeutic Indications",
            "Posology and Use",
            "Contraindications",
            "Interactions",
            "Pregnancy/Lactation",
            "Undesirable Effects"
        ]
        
        content = []
        for section in sections:
            # Map friendly names to probable dataset keys (can be adapted if exact column names vary)
            key_candidate = section.lower().replace(" ", "_").replace("/", "_")
            # If standard keys aren't found, fall back to matching any column with similar wording
            found_val = example.get(key_candidate, example.get(section, None))
            if found_val:
                content.append(f"[{section}]\n{found_val}")
                
        return {"text": "\n\n".join(content)}

    dart_ds = dart_ds.map(extract_dart_text, remove_columns=dart_ds.column_names)
    
    # Filter out empty rows
    clinical_ds = clinical_ds.filter(lambda x: len(x["text"].strip()) > 0)
    dart_ds = dart_ds.filter(lambda x: len(x["text"].strip()) > 0)

    print(f"Clinical notes count: {len(clinical_ds)}")
    print(f"DART pharmacological notes count: {len(dart_ds)}")

    # Combine both datasets into a single corpus
    combined_ds = concatenate_datasets([clinical_ds, dart_ds])

    # Tokenization function for the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False, return_attention_mask=False)

    print("Tokenizing combined dataset...")
    tokenized_ds = combined_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing text"
    )

    # Sequence Packing / Grouping Logic
    # -------------------------------------------------------------------------
    # Instead of training on varying sentence lengths (which requires heavy padding
    # and wastes compute), we concatenate all tokens from all documents into a 
    # continuous stream, and then chunk that stream into fixed-size blocks (e.g., 2048 tokens).
    # This maximizes GPU utilization and provides maximum context length for CPT.
    # -------------------------------------------------------------------------
    def group_texts(examples):
        # OPTIMIZED: Use itertools.chain instead of sum(...,) to prevent RAM crashes
        concatenated_examples = {
            k: list(itertools.chain.from_iterable(examples[k])) 
            for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # We drop the small remainder that doesn't fit exactly into block_size
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
            
        # Split by chunks of max_seq_length
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        # In causal LM, labels are typically identical to input_ids; the model shifts them internally
        result["labels"] = result["input_ids"].copy()
        return result

    print("Packing tokenized dataset into fixed-length blocks...")
    packed_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4, # parallelize block creation
        desc=f"Grouping texts into chunks of {max_seq_length}"
    )

    return packed_ds


# --- Function 2: Model and Tokenizer Initialization ---
def initialize_model_and_tokenizer(model_id):
    """
    Initializes Llama-3.2-3B with 4-bit QLoRA configuration.
    """
    print(f"Loading Base Model ({model_id}) with 4-bit Quantization...")
    # Determine compute dtype based on hardware
    compute_dtype = getattr(torch, "bfloat16") if torch.cuda.is_bf16_supported() else getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Llama 3 models typically don't have a default pad token, we use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Prepare model for kbit training (gradient checkpointing, layer norm fixes)
    model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA Adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()

    return model, tokenizer


# --- Main Continuous Pre-Training Function ---
def main():
    model_id = "meta-llama/Llama-3.2-3B"
    output_dir = "./llama-3.2-3b-italian-medical-cpt"
    
    # Initialize components
    model, tokenizer = initialize_model_and_tokenizer(model_id)
    
    # Prepare Continual Pre-Training packed data
    train_dataset = load_and_prepare_data(tokenizer, max_seq_length=2048)

    print("Initializing Training Configuration...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,     # Simulate batch size 32 (4 * 8)
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        max_steps=500,                     # Modify locally for full epochs config
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        gradient_checkpointing=True,       # Further saves VRAM
        report_to="none"                   # Change to "wandb" or "tensorboard" if configured
    )

    # Note: We use Trainer instead of SFTTrainer because we already performed 
    # the advanced token packing sequence logic manually in `load_and_prepare_data`.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    
    # Needed for older versions of PEFT to silence warnings
    model.config.use_cache = False 

    print("Beginning Continual Pre-Training...")
    trainer.train()

    print(f"Training Complete! Saving final PEFT LoRA adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
