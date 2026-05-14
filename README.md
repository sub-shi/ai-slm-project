# AI SLM Trading Signal Project

## Overview

This project implements a lightweight financial signal-generation system using supervised fine-tuning (SFT) on TinyLlama with QLoRA-based parameter-efficient adaptation.

The system generates structured JSON trading signals for NIFTY options market states and includes:

* dataset auditing
* schema cleaning
* supervised instruction fine-tuning
* deterministic orchestration
* structured evaluation
* retrieval-augmented prompting (RAG)

The project was developed as part of an AI SLM engineering assignment focused on reliability, structured generation, and practical ML systems engineering.

---

# Project Structure

```text
ai-slm-project/
│
├── data/
│   ├── finetune_instructions.jsonl
│   ├── rag_corpus.jsonl
│   ├── market_states.parquet
│   └── retrieve.py
│
├── notebook/
│   └── ai-slm-project.ipynb
│
├── output/
│   ├── cleaned-finetune_instructions.jsonl
│   ├── tinyllama-slm/
│   └── tinyllama_slm_adapter/
│
├── report/
│   └── report.html
│
├── requirements.txt
│
└── README.md
```

---

# Kaggle Notebook

Kaggle notebook link:

https://www.kaggle.com/code/subrat9910/ai-slm-project
---

# Important Note About Paths

The original Kaggle notebook uses Kaggle-specific paths such as:

```python
/kaggle/input/...
```

For repository portability and local execution, the notebook inside this repository has been modified to use relative paths such as:

```python
../data/
../output/
```

This allows the notebook to run locally in:

* VS Code
* Jupyter Notebook
* JupyterLab

without requiring Kaggle filesystem paths.

---

# Setup

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
OR 
Run the first cell of the notebook
```

---

# Run the Notebook

From the repository root:

```bash
jupyter notebook notebook/ai-slm-project.ipynb
```

---

# Model Details

## Base Model

* TinyLlama

## Fine-Tuning Method

* QLoRA
* LoRA adapters
* 4-bit quantization

## Training Objective

Structured JSON generation for NIFTY trading signals.

---

# Evaluation Summary

## Dataset

* Raw samples: 300
* Cleaned samples retained: 261
* Removed samples: 39

## Structured Generation

* Schema pass rate: 100%
* Parse failures: 0

## Observed Limitation

The fine-tuned model exhibited conservative generation behavior with heavy bias toward `NEUTRAL` predictions under deterministic decoding.

---

# RAG Experiment

A lightweight retrieval-augmented prompting experiment was implemented using:

* `rag_corpus.jsonl`
* provided `retrieve.py`

Retrieved historical market regimes and summarized outcomes were injected into prompts as contextual guidance.

The experiment improved contextual richness but occasionally reduced strict schema-following reliability.

---

# Outputs

Generated artifacts include:

* cleaned datasets
* LoRA adapter weights
* checkpoints
* evaluation outputs
* final HTML report

Stored under:

```text
output/
report/
```

---

# Notes

This project prioritizes:

* reliability
* structured generation
* honest evaluation
* practical engineering tradeoffs

rather than directional trading performance claims.
