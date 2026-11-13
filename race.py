"""Assignment 8: Model Race.

Compares three classification approaches on the same 500 labeled customer reviews:
1. scikit-learn (TF-IDF + Logistic Regression, fit on training set only)
2. Cloud LLM (Groq / Gemini with one-word classification and paid token pricing)
3. Local LLM (Ollama running locally with open weights)

All three models are evaluated on the exact same 100-sample test set (20% split with fixed random_state).
Results are cached to ensure zero duplicate API calls on re-runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_race")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "reviews.csv"
RESULTS_PATH = BASE_DIR / "results.md"
CLOUD_CACHE_PATH = BASE_DIR / "data" / "cache_cloud_llm.json"
OLLAMA_CACHE_PATH = BASE_DIR / "data" / "cache_ollama.json"

# Pricing rates per million tokens (USD list price)
PRICING_PER_MILLION: dict[str, dict[str, float]] = {
    # Groq paid list prices ($ per 1M tokens)
    "qwen/qwen3.8-27b": {"input": 0.20, "output": 0.20},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    # Google Gemini paid list prices ($ per 1M tokens)
    "gemini-3.8-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    # Local / Ollama models ($0.00 compute token rate)
    "llama3:latest": {"input": 0.0, "output": 0.0},
    "llama3": {"input": 0.0, "output": 0.0},
}


@dataclass
class ModelResult:
    """Stores benchmarking evaluation metrics for a single model."""

    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    cost_usd: float
    time_per_review_ms: float
    total_time_s: float
    test_size: int

    def to_row(self) -> list[str]:
        """Format metrics for markdown table."""
        return [
            self.name,
            f"{self.accuracy:.4f}",
            f"{self.precision:.4f}",
            f"{self.recall:.4f}",
            f"{self.f1:.4f}",
            f"${self.cost_usd:.6f}",
            f"{self.time_per_review_ms:.2f} ms",
        ]


def load_dataset(csv_path: Path) -> tuple[list[dict[str, str]], list[str], list[str]]:
    """Load reviews dataset and normalize columns and labels using standard csv."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Reviews CSV not found at: {csv_path}")

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        text_col = None
        for cand in ["text", "review_text", "review", "content"]:
            if cand in fieldnames:
                text_col = cand
                break
        if not text_col:
            raise ValueError(f"Could not identify review text column. Found: {fieldnames}")

        label_col = None
        for cand in ["label", "sentiment", "target", "category"]:
            if cand in fieldnames:
                label_col = cand
                break
        if not label_col:
            raise ValueError(f"Could not identify label column. Found: {fieldnames}")

        records: list[dict[str, str]] = []
        texts: list[str] = []
        labels: list[str] = []

        for row in reader:
            raw_text = (row.get(text_col) or "").strip()
            raw_label = (row.get(label_col) or "").strip().lower()
            if raw_label in ("positive", "negative") and raw_text:
                record = {
                    "review_id": row.get("review_id", str(len(records) + 1)),
                    "text": raw_text,
                    "label": raw_label,
                }
                records.append(record)
                texts.append(raw_text)
                labels.append(raw_label)

    return records, texts, labels


def split_data(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Perform fixed stratified train/test split.

    Fit on training set only; test set remains identical across all model runs.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_train, X_test, y_train, y_test


def evaluate_sklearn(
    X_train: list[str],
    y_train: list[str],
    X_test: list[str],
    y_test: list[str],
    random_state: int = 42,
) -> ModelResult:
    """Evaluate scikit-learn Logistic Regression with TF-IDF features.

    TF-IDF vectorizer is fit strictly on X_train to prevent data leakage,
    then applied via .transform() on X_test.
    """
    logger.info("Training scikit-learn (TF-IDF + Logistic Regression)...")
    t0 = time.perf_counter()

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(random_state=random_state)
    clf.fit(X_train_vec, y_train)

    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)
    total_time_s = time.perf_counter() - t0

    time_per_review_ms = (total_time_s * 1000) / len(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="positive", zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label="positive", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="positive", zero_division=0)

    logger.info(
        "scikit-learn complete: Acc=%.4f, F1=%.4f, Time/rev=%.2fms",
        acc,
        f1,
        time_per_review_ms,
    )

    return ModelResult(
        name="scikit-learn (TF-IDF + logistic regression)",
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        cost_usd=0.0,
        time_per_review_ms=time_per_review_ms,
        total_time_s=total_time_s,
        test_size=len(X_test),
    )


def _get_cache_key(model: str, idx: int, text: str) -> str:
    """Generate a deterministic cache key across process invocations."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{model}::{idx}::{h}"


def _load_cache(cache_path: Path) -> dict[str, Any]:
    """Load cached model responses from JSON file."""
    if cache_path.exists():
        try:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Could not read cache %s: %s. Starting fresh.", cache_path, e)
    return {}


def _save_cache(cache_path: Path, data: dict[str, Any]) -> None:
    """Persist cached responses to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _clean_prediction(raw: str) -> str:
    """Extract strictly 'positive' or 'negative' from model output."""
    cleaned = raw.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    tokens = cleaned.split()
    for token in tokens:
        if token in ("positive", "negative"):
            return token
    if "positive" in cleaned and "negative" not in cleaned:
        return "positive"
    if "negative" in cleaned and "positive" not in cleaned:
        return "negative"
    return "unknown"


def generate(
    prompt: str,
    provider: str = "groq",
    model: str | None = None,
    timeout: float = 30.0,
    usage: dict[str, int] | None = None,
) -> Iterator[str]:
    """Generate streaming text response from the configured LLM provider.

    Reused from Assignment 6 multi-provider streaming generator (Groq, Gemini, Ollama).
    Yields chunks of text as received from the LLM.
    """
    normalized_provider = provider.lower().strip()

    if normalized_provider == "groq":
        from groq import Groq

        api_key = (os.getenv("GROQ_API_KEY") or "").strip("'\" ")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not configured in environment")
        client = Groq(api_key=api_key, timeout=timeout)
        target_model = model or os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        completion = client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
            temperature=0.0,
            max_tokens=5,
        )
        for chunk in completion:
            if usage is not None and hasattr(chunk, "usage") and chunk.usage:
                u = chunk.usage
                if getattr(u, "prompt_tokens", None) is not None:
                    usage["input_tokens"] = u.prompt_tokens
                if getattr(u, "completion_tokens", None) is not None:
                    usage["output_tokens"] = u.completion_tokens
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None) or ""
                if content:
                    yield content

    elif normalized_provider == "gemini":
        from google import genai

        api_key = (os.getenv("GEMINI_API_KEY") or "").strip("'\" ")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured in environment")
        timeout_ms = int(timeout * 1000) if timeout else None
        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": timeout_ms} if timeout_ms else None,
        )
        target_model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        response = client.models.generate_content_stream(
            model=target_model,
            contents=prompt,
        )
        for chunk in response:
            if usage is not None and hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                meta = chunk.usage_metadata
                prompt_tokens = getattr(meta, "prompt_token_count", None)
                if prompt_tokens is not None:
                    usage["input_tokens"] = prompt_tokens
                cand_tokens = getattr(meta, "candidates_token_count", None)
                if cand_tokens is not None:
                    usage["output_tokens"] = cand_tokens
            if chunk.text:
                yield chunk.text

    elif normalized_provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        url = f"{base_url.rstrip('/')}/api/generate"
        target_model = model or os.getenv("OLLAMA_MODEL", "llama3:latest")
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url,
                json={
                    "model": target_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 5},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            raw_text = data.get("response", "")
            if usage is not None:
                usage["input_tokens"] = data.get("prompt_eval_count", 0)
                usage["output_tokens"] = data.get("eval_count", 0)
            yield raw_text
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def ask_llm_one_word(
    review_text: str,
    provider: str,
    model: str,
    usage: dict[str, int] | None = None,
) -> tuple[str, float]:
    """Classify review sentiment asking for exactly one word using Assignment 6 generate function."""
    prompt = (
        "You are a binary sentiment classifier.\n"
        "Classify the following customer review as either 'positive' or 'negative'.\n"
        "Respond with ONLY ONE WORD: either 'positive' or 'negative', nothing else.\n\n"
        f'Review: "{review_text}"\n'
        "Sentiment:"
    )
    t0 = time.perf_counter()
    chunks = []
    for chunk in generate(prompt=prompt, provider=provider, model=model, usage=usage):
        chunks.append(chunk)
    latency_ms = (time.perf_counter() - t0) * 1000
    raw_answer = "".join(chunks).strip()
    return raw_answer, latency_ms


def evaluate_cloud_llm(
    X_test: list[str],
    y_test: list[str],
    provider: str = "groq",
    model: str | None = None,
    cache_path: Path = CLOUD_CACHE_PATH,
    force_no_cache: bool = False,
) -> ModelResult:
    """Benchmark Cloud LLM (Groq or Gemini) over the test set with response caching."""
    if not model:
        if provider == "groq":
            model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        else:
            model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

    display_name = f"Cloud LLM ({provider.capitalize()}: {model})"
    logger.info("Evaluating %s on %d test reviews...", display_name, len(X_test))

    cache = {} if force_no_cache else _load_cache(cache_path)
    predictions = []
    total_cost_usd = 0.0
    latencies = []

    pricing = PRICING_PER_MILLION.get(
        model, {"input": 0.20, "output": 0.20}
    )

    for idx, text in enumerate(X_test):
        cache_key = _get_cache_key(model, idx, text)
        cached_entry = cache.get(cache_key)

        if cached_entry and not force_no_cache:
            pred = cached_entry["prediction"]
            cost = cached_entry.get("cost_usd", 0.0)
            latency = cached_entry.get("latency_ms", 0.0)
        else:
            retries = 5
            raw_output = ""
            usage: dict[str, int] = {}
            latency = 0.0

            for attempt in range(1, retries + 1):
                try:
                    usage = {}
                    raw_output, latency = ask_llm_one_word(
                        text, provider=provider, model=model, usage=usage
                    )
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    logger.warning(
                        "Cloud call failed (attempt %d/%d): %s", attempt, retries, e
                    )
                    if "429" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str:
                        sleep_s = 5.0 * attempt
                        logger.info("Rate limit hit; sleeping %.1fs...", sleep_s)
                        time.sleep(sleep_s)
                    else:
                        time.sleep(1.0 * attempt)
                    if attempt == retries:
                        raw_output = "unknown"
                        latency = 0.0
                        break

            pred = _clean_prediction(raw_output)
            in_tokens = usage.get("input_tokens", max(1, len(text) // 4 + 30))
            out_tokens = usage.get("output_tokens", 2)
            cost = (in_tokens * (pricing["input"] / 1_000_000.0)) + (
                out_tokens * (pricing["output"] / 1_000_000.0)
            )

            cache[cache_key] = {
                "test_index": idx,
                "text": text,
                "raw_output": raw_output,
                "prediction": pred,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "cost_usd": cost,
                "latency_ms": latency,
            }
            _save_cache(cache_path, cache)

            # Polite delay between live API calls to stay within free-tier rate limits
            time.sleep(0.12)

        predictions.append(pred)
        total_cost_usd += cost
        latencies.append(latency)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(X_test):
            logger.info("Cloud LLM progress: %d/%d processed.", idx + 1, len(X_test))

    avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else 0.0
    total_time_s = sum(latencies) / 1000.0

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, pos_label="positive", zero_division=0)
    rec = recall_score(y_test, predictions, pos_label="positive", zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label="positive", zero_division=0)

    logger.info(
        "Cloud LLM complete: Acc=%.4f, F1=%.4f, Cost=$%.6f, Time/rev=%.2fms",
        acc,
        f1,
        total_cost_usd,
        avg_latency_ms,
    )

    return ModelResult(
        name=display_name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        cost_usd=total_cost_usd,
        time_per_review_ms=avg_latency_ms,
        total_time_s=total_time_s,
        test_size=len(X_test),
    )


def evaluate_ollama(
    X_test: list[str],
    y_test: list[str],
    model: str = "llama3:latest",
    cache_path: Path = OLLAMA_CACHE_PATH,
    force_no_cache: bool = False,
) -> ModelResult:
    """Benchmark local Ollama model over the test set with response caching."""
    model_env = os.getenv("OLLAMA_MODEL", model)
    display_name = f"Ollama ({model_env})"
    logger.info("Evaluating %s on %d test reviews...", display_name, len(X_test))

    cache = {} if force_no_cache else _load_cache(cache_path)
    predictions = []
    latencies = []

    for idx, text in enumerate(X_test):
        cache_key = _get_cache_key(model_env, idx, text)
        cached_entry = cache.get(cache_key)

        if cached_entry and not force_no_cache:
            pred = cached_entry["prediction"]
            latency = cached_entry.get("latency_ms", 0.0)
        else:
            raw_output = ""
            latency = 0.0
            usage: dict[str, int] = {}
            try:
                raw_output, latency = ask_llm_one_word(
                    text, provider="ollama", model=model_env, usage=usage
                )
            except Exception as e:
                logger.warning("Ollama call failed at index %d: %s", idx, e)
                raw_output = "unknown"
                latency = 0.0

            pred = _clean_prediction(raw_output)

            cache[cache_key] = {
                "test_index": idx,
                "text": text,
                "raw_output": raw_output,
                "prediction": pred,
                "cost_usd": 0.0,
                "latency_ms": latency,
            }
            _save_cache(cache_path, cache)

        predictions.append(pred)
        latencies.append(latency)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(X_test):
            logger.info("Ollama progress: %d/%d processed.", idx + 1, len(X_test))

    avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else 0.0
    total_time_s = sum(latencies) / 1000.0

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, pos_label="positive", zero_division=0)
    rec = recall_score(y_test, predictions, pos_label="positive", zero_division=0)
    f1 = f1_score(y_test, predictions, pos_label="positive", zero_division=0)

    logger.info(
        "Ollama complete: Acc=%.4f, F1=%.4f, Cost=$0.00, Time/rev=%.2fms",
        acc,
        f1,
        avg_latency_ms,
    )

    return ModelResult(
        name=display_name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        cost_usd=0.0,
        time_per_review_ms=avg_latency_ms,
        total_time_s=total_time_s,
        test_size=len(X_test),
    )
