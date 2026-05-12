# Italian Clinical Note De-Identification

## Project Overview
A 3-phase fine-tuning pipeline on `meta-llama/Llama-3.2-3B` for GDPR-compliant de-identification of Italian clinical notes. The pipeline extracts Protected Health Information (PHI) entities — **NOME**, **ETÀ**, **DATA**, **LUOGO/INDIRIZZO** — and returns them as a structured JSON array.

**Pipeline phases:**
1. **CPT** — Continual Pre-Training on Italian medical text to adapt domain syntax
2. **SFT** — Supervised Fine-Tuning with LoRA (4-bit NF4) to learn PHI extraction
3. **CV Eval** — 5-Fold Cross-Validation over 80 gold-standard annotated notes

---

## Hardware Requirements

- **GPU:** NVIDIA RTX 4090 (24 GB VRAM) recommended. RTX 3090/4080 (16–24 GB) also works with the default batch size.
- All training scripts use 4-bit NF4 quantization + LoRA, keeping peak VRAM under ~16 GB during SFT/CV phases.

---

## Environment Setup

**1. Clone the repository:**
```bash
git clone https://github.com/9Roflander/Italian-Clinical-Note-Deidentification.git
cd Italian-Clinical-Note-Deidentification
```

**2. Create a virtual environment and install dependencies:**
```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

> **Windows note:** Set `PYTHONUTF8=1` before running any script to avoid encoding errors with the TRL library:
> ```powershell
> $env:PYTHONUTF8 = "1"
> ```

**3. Authenticate with Hugging Face** (required for the gated Llama-3.2-3B model):
```bash
huggingface-cli login
```
You need a Hugging Face account with access approved for `meta-llama/Llama-3.2-3B`.

---

## Data

The `data/` directory contains all training and evaluation data — no additional downloads needed for local datasets:

| File | Description |
|---|---|
| `data/synthetic_clinical_1000.json` | 1,000 synthetically generated EHRs with PHI annotations (training) |
| `data/gold_standard_80.json` | 80 manually annotated clinical notes (evaluation gold standard) |

Two external datasets are fetched automatically from Hugging Face at runtime:
- **DART** corpus (Phase 1 CPT)
- **NLP-FBK/synthetic-crf-train** (negative samples for SFT/CV)

---

## Execution Pipeline

All scripts are run from the **project root directory**.

### Phase 1: Continual Pre-Training (CPT)
Adapts the base Llama-3.2-3B model to Italian medical syntax.
```bash
python src/train_cpt.py
```
Output: `./llama-3.2-3b-italian-medical-cpt/` (LoRA adapter)

---

### Phase 2: Supervised Fine-Tuning (SFT)
Trains the CPT-adapted model to perform clinical de-identification.

> Requires Phase 1 output at `./llama-3.2-3b-dart-sft/`.
> If you skipped CPT, point `PHASE1_ADAPTER_DIR` in `src/train_phase2_sft.py` to your adapter path.

```bash
python src/train_phase2_sft.py
```
Output: `./llama-3.2-3b-deid-sft/` (LoRA adapter)

---

### Phase 3: 5-Fold Cross-Validation
Strict cross-validation against the 80 gold-standard annotations. Reports Precision, Recall, and F1.

> Requires the Phase 1 merged model at `./temp_merged_phase1/`.
> This directory is created automatically by Phase 2. If running CV standalone, run Phase 2 first or merge manually.

```bash
python src/train_evaluate_cv.py
```
Output: `./cv_fold_outputs/fold_N/` per fold, then averaged P/R/F1 printed to stdout.

---

## Output Directory Structure

After running the full pipeline, the project root will contain:

```
Italian-Clinical-Note-Deidentification/
├── llama-3.2-3b-italian-medical-cpt/   # Phase 1 CPT LoRA adapter
├── llama-3.2-3b-dart-sft/              # Phase 1 DART SFT adapter (if separate)
├── temp_merged_phase1/                  # Merged base + Phase 1 (auto-generated)
├── llama-3.2-3b-deid-sft/              # Phase 2 SFT adapter
├── cv_fold_outputs/                     # Phase 3 fold checkpoints
└── data/                               # Training and evaluation data
```

---

## Entity Types

| Type | Description | Example |
|---|---|---|
| `NOME` | Patient or doctor names | *Maria Rossi* |
| `ETÀ` | Age references | *45 anni* |
| `DATA` | Dates and timestamps | *12 marzo 2023* |
| `LUOGO/INDIRIZZO` | Locations and addresses | *Ospedale San Raffaele, Milano* |
