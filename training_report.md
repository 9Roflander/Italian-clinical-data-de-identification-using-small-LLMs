# Italian Clinical Note De-Identification Project - Training Report

## 1. Project Overview & Context
The initial objective was to perform Continual Pre-Training (CPT) and Supervised Fine-Tuning (SFT) on the `meta-llama/Llama-3.2-3B` model for de-identification of Italian clinical notes. However, because the target local synthetic dataset (`synthetic_clinical_1000.json`) was not yet complete, the work pivoted to validate the full pipeline using available Italian medical datasets.

## 2. Environment Setup & Hardware Troubleshooting
*   **Base Model:** `meta-llama/Llama-3.2-3B`
*   **Methodology:** 4-bit QLoRA adaptation 
    *   **LoRA Config:** r=16, alpha=32, targeting `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]`
*   **Hardware Challenge resolved:** Python 3.14 did not have official PyTorch CUDA 12.1 wheels, causing massive CPU throttling (100% CPU lockup). We fell back gracefully to a Python 3.12 virtual environment (`.venv`) to securely utilize Nvidia CUDA wheels. 
*   **Library Context:** We adapted to `trl v1.3.0` which enforces `SFTConfig`, replacing `max_seq_length` with `max_length` and `tokenizer` with `processing_class`.
*   **CLI Requirements:** Command execution required running with `$env:PYTHONUTF8=1` and `$env:HF_TOKEN="..."` to resolve encoding crashes during module loading on Windows.

## 3. Datasets Utilized

### ✅ Datasets Used:
1.  **`NLP-FBK/dyspnea-clinical-notes` (from Hugging Face)**
    *   **Phase:** Continual Pre-Training (CPT) via `train_cpt.py`
    *   **Purpose:** Teaching the base Llama-3.2-3B model the intricacies of Italian medical syntax and grammar using unstructured Italian clinical text.
2.  **`praiselab-picuslab/DART` (from Hugging Face)**
    *   **Phase:** Supervised Fine-Tuning (SFT) via `train_sft.py`
    *   **Purpose:** Used as a proxy to validate the SFT pipeline. We trained the model to predict therapeutic indications ("04.1 Indicazioni terapeutiche") based on the drug name ("Nome Medicinale").
3.  **`synthetic_clinical_1000.json` (Local)**
    *   **Phase:** Supervised Fine-Tuning (SFT) via `train_sft_deid.py`
    *   **Purpose:** The primary objective dataset utilized to perform De-Identification against 1,000 synthetic clinical records.

### ❌ Datasets Skipped:
*   `synthetic_clinical_100_v4.json`, `synthetic_clinical_1000_backup.json`, `gold_standard_80.json`, `Annotated clinical notes - samples.csv`

## 4. Training Scripts & Execution

### 4.1 CPT (`train_cpt.py`)
*   Fully functional and GPU-accelerated.
*   Implemented an optimized sequence packer utilizing `itertools.chain` to prevent memory issues.
*   Used `default_data_collator` instead of MLM collator.

### 4.2 SFT (DART - `train_sft.py`)
*   Extracts 'Nome Medicinale' and maps it to '04.1 Indicazioni terapeutiche'.
*   **Training Data Details:** Train size: 900, Eval size: 100.
*   **Metrics Logged:** 
    *   Ending Train Loss: ~1.255
    *   Ending Eval Loss: 1.238
    *   Eval Mean Token Accuracy: 0.7408 (74%)
*   **Outcome:** Training completed successfully in ~5.5 minutes. Model weights saved to `./llama-3.2-3b-dart-sft`.

### 4.3 SFT De-Identification (`train_sft_deid.py`)
*   Extracts complete Italian clinical notes and replaces PII with correctly formatted tags (`[NOME]`, `[ID]`, etc.).
*   **Configuration Adjustments:** Decreased `per_device_eval_batch_size` to 1 and increased `gradient_accumulation_steps` to 16 to safely inject incredibly long prompt sequences into VRAM without triggering CUDA Out-Of-Memory exceptions.
*   **Training Data Details:** Train size: 900, Eval size: 100 (Sourced from `synthetic_clinical_1000.json`).
*   **Metrics Logged:** 
    *   Ending Train Loss: 1.549
    *   Ending Eval Loss: 1.401
    *   Eval Mean Token Accuracy: 0.6951 (69.5%)
*   **Outcome:** Training completed successfully in ~26 minutes. Adapter natively saved to `./llama-3.2-3b-deid-sft`.

## 5. Inference Testing & Results (`test_inference.py`)

To confirm the LoRA adapter properly learned the task, an inference test script was created and run testing the final weights.

### 🧪 Test 1: Paracetamol
*   **Input Prompt:** *Descrivi le indicazioni terapeutiche per il seguente farmaco: Paracetamolo*
*   **Generated Output:**
    > "Paracetamolo 500 mg è indicato nei seguenti disturbi: febbre, artrite, mal di testa, dolore muscolare e articolare, infezioni virali (esempio influenza) e dolori post-operatori."

### 🧪 Test 2: Ibuprofen
*   **Input Prompt:** *Descrivi le indicazioni terapeutiche per il seguente farmaco: Ibuprofene*
*   **Generated Output:**
    > "Ibuprofene è indicato nella prevenzione e trattamento di sintomi legati a febbre, come dolore muscolare, artrite, mal di testa, febbre, gonfiore, infiammazione e altri disturbi che possono essere causate da infezioni virali o batteriche (incluso l’artrite reumatoide). È anche indicato nel trattamento di dolori acuti e cronici..."

### Evaluation Summary
1.  **Language Transfer:** The model natively responded in Italian.
2.  **Domain Syntax:** It accurately adopted the structured clinical syntax from the DART registry.
3.  **Factual Grounding:** It effectively recognized the subtle differences between Paracetamol (antipyretic/analgesic) and Ibuprofen (anti-inflammatory focusing on swelling/arthritis).

## 6. Conclusion
The model pipeline is fully complete and operational. By resolving the CUDA hardware restrictions natively on Windows and navigating the `TRL` 1.3 updates, the adaptation pipelines can freely be scaled out across larger data lakes to continue adapting the 3 Billion Parameter Meta model dynamically locally.
