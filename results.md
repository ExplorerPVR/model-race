# Assignment 8: Model Race Results

## Results Table

| Model                                       |   Accuracy |   Precision |   Recall |   F1 | Cost at paid list price (USD)   | Time per review (ms)   |
|---------------------------------------------|------------|-------------|----------|------|---------------------------------|------------------------|
| scikit-learn (TF-IDF + logistic regression) |          1 |           1 |        1 |    1 | $0.000000                       | 0.16 ms                |
| Cloud LLM (Groq: qwen/qwen3.8-27b)          |          1 |           1 |        1 |    1 | $0.002002                       | 2195.85 ms             |
| Ollama (llama3:latest)                      |          1 |           1 |        1 |    1 | $0.000000                       | 4718.61 ms             |

---

## Production Recommendation (Which Model to Ship)

For this customer review sentiment classification task, I would ship **scikit-learn (TF-IDF + Logistic Regression)**. It delivers a perfect 1.0000 accuracy and F1 score on the test set while running at an astonishing **0.16 ms per review**—over **13,349x faster** than the cloud LLM and **28,686x faster** than local Ollama. Furthermore, it operates completely offline with **zero token cost ($0.00)**, no network latency, and no external API failure modes or rate limit constraints. While LLMs provide unmatched value for unstructured generative tasks and multi-step reasoning, classical ML is the unequivocal, battle-tested production choice for high-throughput, latency-critical text classification.

---

## Class Balance Analysis & Implications for Accuracy

- **Full Dataset (500 reviews):** 250 Positive (50.0%), 250 Negative (50.0%).
- **Test Set (100 reviews):** 50 Positive (50.0%), 50 Negative (50.0%).

The classes in this dataset are **completely balanced** (exactly 50% positive and 50% negative).

### What This Means for Accuracy:
1. **Meaningful Metric:** When classes are balanced, raw accuracy is an honest and unskewed performance metric. A naive "majority class" or random-guessing baseline would only achieve 50.0% accuracy, meaning any score in the 90%+ range reflects genuine discriminative power.
2. **Symmetric Error Penalty:** Unlike severely imbalanced datasets (e.g., 98% positive where a dummy model predicting "positive" trivially scores 98% accuracy while being completely broken), balanced data guarantees that false positives and false negatives carry equal weight.
3. **Precision & Recall Alignment:** Because precision, recall, and F1 align closely with accuracy in balanced settings, the F1-score confirms that neither class is disproportionately favored or neglected by any of the three models.

---

## Benchmark Configuration Details

- **Dataset Size:** 500 total customer reviews (400 train / 100 test).
- **Train/Test Split:** 80/20 Stratified Split (`random_state=42`).
- **TF-IDF Fitting:** Fit strictly on the training set (no data leakage), then applied via `.transform()` to the test set.
- **Cloud LLM Model:** `Cloud LLM (Groq: qwen/qwen3.8-27b)` (zero-shot prompt, temperature 0.0).
- **Local LLM Model:** `Ollama (llama3:latest)` (zero-shot prompt, temperature 0.0).
- **Caching:** LLM responses are persistently cached in `data/cache_cloud_llm.json` and `data/cache_ollama.json` to guarantee idempotency and avoid unnecessary token consumption or rate limits.
