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
import logging
import time
from dataclasses import dataclass
from pathlib import Path

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
