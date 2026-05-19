# ChatDED Agent

ChatDED Agent is an autonomous assistant for dry eye disease (DED) management in primary care scenarios.  
It combines multimodal clinical context into one conversation flow:

- meibography image quality control
- meibomian gland grading interpretation
- retrieval-augmented generation (RAG) for evidence-informed suggestions

The current codebase focuses on an end-to-end simulation pipeline that reads structured patient data, performs staged dialogue generation, and produces final patient-facing recommendations.

## Project Overview

This project is designed to support clinicians, especially in resource-limited settings, by improving consistency, efficiency, and communication quality in DED-related triage and follow-up conversations.

At a high level, the workflow:

1. collects patient symptom descriptions through guided dialogue;
2. links each conversation with structured patient records;
3. incorporates meibography analysis outputs (quality + gland metrics);
4. retrieves relevant clinical text chunks via RAG;
5. generates an integrated, patient-friendly response.

## Repository Structure

```text
.
├── README.md                      # Project entry and quick start
├── environment.yml                # Conda environment dependencies
├── ded.html                       # Demo HTML page (if needed for display)
├── embeddings.pkl                 # Cached embedding vectors for retrieval
├── docs/
│   └── PROJECT_INTRO_ZH.md        # Chinese project introduction
├── data/
│   ├── data_loader.py             # CSV/TXT IO helpers
│   ├── dry_eye_data_v3.csv        # Main structured patient dataset
│   └── txt_files/                 # Knowledge documents for RAG
├── prompts/
│   └── prompt.py                  # Prompt templates for each conversation stage
├── rag/
│   └── chunk.py                   # Chunking + embedding similarity retrieval
├── llm_qa/
│   ├── qa.py                      # Main pipeline entry script (CLI unchanged)
│   ├── conversation_engine.py     # State-driven multi-turn conversation engine
│   └── request_llm.py             # LLM request wrapper
└── results/
    ├── conversations_all.txt      # Full generated conversations
    ├── total_time.txt             # End-to-end latency per case
    └── rag_time.txt               # Retrieval latency per case
```

## Multi-turn Refactor Notes

The input/output framework is preserved:

- CLI still uses `python llm_qa/qa.py` with `--doc_path` and `--data_path`.
- Output files remain:
  - `results/conversations_all.txt`
  - `results/total_time.txt`
  - `results/rag_time.txt`

Internal implementation is now state-driven:

- one patient case maps to one `DialogueState`
- each stage is an explicit turn handler (opening, record retrieval, summary, image analysis, RAG + final suggestion)
- `qa.py` is now a lightweight orchestrator

## Quick Start

### 1) Create and activate Conda environment

```bash
conda env create -f environment.yml
conda activate agent_env
```

### 2) Export OpenAI API key

Preferred environment variable: `OPENAI_API_KEY`  
Backward-compatible fallback: `API_KEY`

Linux/macOS (bash/zsh):

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
# optional fallback format:
# export API_KEY="YOUR_OPENAI_API_KEY_HERE"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
# optional fallback format:
# $env:API_KEY="YOUR_OPENAI_API_KEY_HERE"
```

### 3) Run the pipeline

From repository root:

```bash
python llm_qa/qa.py
```

Optional arguments:

```bash
python llm_qa/qa.py --doc_path "data/txt_files" --data_path "data/dry_eye_data_v3.csv"
```

## Runtime Outputs

After execution, outputs are written under `results/`:

- `conversations_all.txt`: full case-by-case dialogue and generated responses
- `total_time.txt`: total processing time per case
- `rag_time.txt`: RAG retrieval time per case

## Notes

- The pipeline currently assumes required patient columns are present in the CSV.
- `embeddings.pkl` is reused for retrieval speed; regenerate when document corpus changes significantly.
- This repository is oriented to research/prototyping workflows and can be further modularized for production deployment.
