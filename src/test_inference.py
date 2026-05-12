import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def main():
    base_model_id = "meta-llama/Llama-3.2-3B"
    adapter_path = "../models/../models/llama-3.2-3b-dart-sft"

    print("Configuring quantization...")
    compute_dtype = getattr(torch, "bfloat16") if torch.cuda.is_bf16_supported() else getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("Loading specialized tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("Injecting LoRA adapter weights...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    drugs_to_test = ["Paracetamolo", "Ibuprofene"]

    print("\nStarting inference test...\n")
    for drug in drugs_to_test:
        prompt = (
            "Sei un assistente medico farmacologico esperto. "
            f"Descrivi le indicazioni terapeutiche per il seguente farmaco: {drug}\n\n"
            "### Indicazioni Terapeutiche:\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the generated part
        generated_text = response[len(prompt):]
        
        print(f"=== {drug} ===")
        print(f"Prompt fornito: {drug}")
        print(f"Risultato Generato:\n{generated_text}\n")

if __name__ == "__main__":
    main()
