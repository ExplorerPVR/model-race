# 🏎️ Model Race: scikit-learn vs Cloud LLM vs Local Ollama

> **Assignment 8:** Learn when a small, cheap ML model beats an LLM, by measuring instead of guessing.

---

## 📌 Overview

Can a lightweight classical machine learning model outperform state-of-the-art Large Language Models (LLMs) on narrow classification tasks? 

This project benchmarks **three distinct classification paradigms** on the exact same dataset of **500 labeled customer reviews** (positive and negative sentiment):

1. **Classical ML:** `scikit-learn` using TF-IDF feature extraction and Logistic Regression.
   - *Key Constraint:* TF-IDF is fit strictly on the training set (400 reviews) to prevent data leakage, then evaluated on the test set.
2. **Cloud LLM:** Groq Cloud (`qwen/qwen3.8-27b`) / Google Gemini (`gemini-3.8-flash`) via API, using the Assignment 6 streaming generator, prompted for zero-shot single-word classification (`positive` or `negative`).
   - *Token & Pricing:* Tracked at published paid list price rates ($ per 1M tokens).
3. **Local LLM:** Ollama (`llama3:latest` 8B) running completely offline on localhost with the identical zero-shot prompt.

All three models are scored on the **exact same 100-sample test set (20%)** using a fixed `random_state=42` for 100% deterministic reproducibility. All LLM responses are persistently cached in JSON files to prevent unnecessary API expenses and stay within free-tier rate limits on re-runs.

---

## 📊 Summary Results Table

| Model | Accuracy | Precision | Recall | F1 | Cost at paid list price (USD) | Time per review (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **scikit-learn (TF-IDF + logistic regression)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **$0.000000** | **0.10 ms** |
| **Cloud LLM (Groq: qwen/qwen3.8-27b)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **$0.001608** | **2,217.31 ms** |
| **Ollama (llama3:latest)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **$0.000000** | **4,487.46 ms** |

*(Detailed analysis and breakdown available in [`results.md`](./results.md))*

---

## 💡 Production Recommendation: Which Model to Ship?

**I would unequivocally ship scikit-learn (TF-IDF + Logistic Regression) for this task.**

1. **Speed & Latency:** At **0.10 ms per review**, scikit-learn is over **20,000x faster** than the Cloud LLM (~2,217 ms per request including network roundtrips) and **40,000x faster** than local Ollama (~4,487 ms).
2. **Cost:** Running on CPU requires **$0.00** in token costs, eliminating recurring vendor fees ($1.60 per 100k reviews on Groq, significantly higher on larger proprietary LLMs).
3. **Reliability & Simplicity:** Classical ML runs completely offline without network dependencies, provider rate limits (e.g. HTTP 429), API keys, or GPU infrastructure requirements.
4. **When to use LLMs:** LLMs remain indispensable for unstructured generative tasks, reasoning over ambiguous contexts, or zero-shot cold starts without labeled data. However, once 400 labeled domain examples are available, classical linear models achieve equal or superior accuracy with near-zero latency and cost.

---

## ⚖️ Class Balance Analysis & Meaning for Accuracy

- **Total Reviews:** 500 reviews (250 Positive [50.0%], 250 Negative [50.0%]).
- **Test Set:** 100 reviews (50 Positive [50.0%], 50 Negative [50.0%]).

### What this means for Accuracy:
1. **Unskewed Metric:** Because the dataset is perfectly balanced (50/50), raw accuracy is a valid, unskewed performance indicator. A trivial majority-class classifier or random coin toss would achieve exactly 50.0% accuracy.
2. **Symmetric Penalties:** In balanced datasets, false positives and false negatives carry equal weight, avoiding the classic imbalance paradox where a model predicting "positive" 100% of the time gets 95% accuracy on a 95/5 skewed dataset.
3. **F1 & Recall Alignment:** The precision, recall, and F1 scores align symmetrically with accuracy, validating that neither positive nor negative sentiment is disadvantaged.

---

## 📂 Repository Structure

```text
model-race/
├── data/
│   ├── reviews.csv             # 500 labeled customer reviews
│   ├── cache_cloud_llm.json    # Cached Cloud LLM responses and latencies
│   └── cache_ollama.json       # Cached Ollama responses and latencies
├── race.py                     # End-to-end benchmark pipeline
├── results.md                  # Final generated results table & report
├── pyproject.toml              # Project dependencies and metadata
├── uv.lock                     # Deterministic dependency lockfile
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or standard `pip`
- [Ollama](https://ollama.ai) installed with `llama3:latest` pulled (`ollama pull llama3`)

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/ExplorerPVR/model-race.git
cd model-race

# Install dependencies with uv
uv sync
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```
Key configuration parameters:
```ini
GROQ_API_KEY="your-groq-api-key"
GROQ_MODEL="qwen/qwen3.8-27b"
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_MODEL="gemini-3.8-flash"
CLOUD_PROVIDER="groq"

OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="llama3:latest"
```

### 4. Run the Model Race
```bash
# Run with uv
uv run python race.py

# Or run with active virtual environment
python race.py
```

### 5. CLI Options
```bash
# Test with Gemini instead of Groq
uv run python race.py --cloud-provider gemini

# Use custom reviews dataset
uv run python race.py --data path/to/my_reviews.csv

# Bypass cache and re-query all models
uv run python race.py --no-cache
```

---

## 👥 Development Branch & Review Workflow

This implementation is developed on the development branch `development` ready for pull request submission and peer code review.
- **Base branch:** `main`
- **Development branch:** `development`
- **Verification:** All checklist items passed, test split deterministically reproduced, and cache persisted.
