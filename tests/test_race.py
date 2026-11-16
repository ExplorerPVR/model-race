"""Unit tests for Assignment 8 Model Race benchmark pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path

from race import (
    DATA_PATH,
    ModelResult,
    _clean_prediction,
    _get_cache_key,
    evaluate_sklearn,
    generate_markdown_report,
    load_dataset,
    split_data,
)


class TestModelRace(unittest.TestCase):
    """Test suite for Model Race data processing, ML pipeline, and report generation."""

    def test_load_dataset(self) -> None:
        """Verify reviews.csv loads 500 reviews with balanced classes."""
        self.assertTrue(DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}")
        records, texts, labels = load_dataset(DATA_PATH)

        self.assertEqual(len(records), 500)
        self.assertEqual(len(texts), 500)
        self.assertEqual(len(labels), 500)

        pos_count = sum(1 for lab in labels if lab == "positive")
        neg_count = sum(1 for lab in labels if lab == "negative")
        self.assertEqual(pos_count, 250)
        self.assertEqual(neg_count, 250)

    def test_split_data_reproducibility_and_stratification(self) -> None:
        """Verify train/test split is deterministic, 80/20, and stratified."""
        _, texts, labels = load_dataset(DATA_PATH)
        X_train, X_test, y_train, y_test = split_data(
            texts, labels, test_size=0.20, random_state=42
        )

        self.assertEqual(len(X_train), 400)
        self.assertEqual(len(X_test), 100)
        self.assertEqual(len(y_train), 400)
        self.assertEqual(len(y_test), 100)

        # Test set must be exactly 50 positive and 50 negative
        self.assertEqual(y_test.count("positive"), 50)
        self.assertEqual(y_test.count("negative"), 50)

        # Re-run must produce identical splits
        X_train2, X_test2, y_train2, y_test2 = split_data(
            texts, labels, test_size=0.20, random_state=42
        )
        self.assertEqual(X_test, X_test2)
        self.assertEqual(y_test, y_test2)

    def test_evaluate_sklearn(self) -> None:
        """Verify scikit-learn Logistic Regression trains and scores test set."""
        _, texts, labels = load_dataset(DATA_PATH)
        X_train, X_test, y_train, y_test = split_data(
            texts, labels, test_size=0.20, random_state=42
        )

        result = evaluate_sklearn(X_train, y_train, X_test, y_test, random_state=42)
        self.assertIsInstance(result, ModelResult)
        self.assertGreaterEqual(result.accuracy, 0.90)
        self.assertGreaterEqual(result.f1, 0.90)
        self.assertEqual(result.cost_usd, 0.0)
        self.assertGreater(result.time_per_review_ms, 0.0)

    def test_clean_prediction(self) -> None:
        """Verify prediction extraction handles punctuation, whitespace, and capitalization."""
        self.assertEqual(_clean_prediction("positive"), "positive")
        self.assertEqual(_clean_prediction("POSITIVE"), "positive")
        self.assertEqual(_clean_prediction("Positive."), "positive")
        self.assertEqual(_clean_prediction("negative"), "negative")
        self.assertEqual(_clean_prediction("Negative!"), "negative")
        self.assertEqual(_clean_prediction("The sentiment is positive."), "positive")
        self.assertEqual(_clean_prediction("I classify this as negative"), "negative")
        self.assertEqual(_clean_prediction("maybe"), "unknown")

    def test_get_cache_key(self) -> None:
        """Verify cache keys are deterministic across identical inputs."""
        key1 = _get_cache_key("groq-model", 5, "Great product!")
        key2 = _get_cache_key("groq-model", 5, "Great product!")
        key3 = _get_cache_key("groq-model", 6, "Great product!")
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)

    def test_markdown_report_generation(self) -> None:
        """Verify results.md includes table, shipping recommendation, and class balance analysis."""
        dummy_results = [
            ModelResult(
                name="scikit-learn (TF-IDF + logistic regression)",
                accuracy=1.0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                cost_usd=0.0,
                time_per_review_ms=0.10,
                total_time_s=0.01,
                test_size=100,
            ),
            ModelResult(
                name="Cloud LLM (Groq: qwen/qwen3.8-27b)",
                accuracy=1.0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                cost_usd=0.001608,
                time_per_review_ms=2217.31,
                total_time_s=221.73,
                test_size=100,
            ),
            ModelResult(
                name="Ollama (llama3:latest)",
                accuracy=1.0,
                precision=1.0,
                recall=1.0,
                f1=1.0,
                cost_usd=0.0,
                time_per_review_ms=4487.46,
                total_time_s=448.74,
                test_size=100,
            ),
        ]
        labels = ["positive"] * 250 + ["negative"] * 250
        y_test = ["positive"] * 50 + ["negative"] * 50

        temp_report = Path("tests") / "temp_results.md"
        try:
            content = generate_markdown_report(
                dummy_results, labels, y_test, output_path=temp_report
            )
            self.assertIn("| Model", content)
            self.assertIn("scikit-learn (TF-IDF + logistic regression)", content)
            self.assertIn("Production Recommendation", content)
            self.assertIn("Class Balance Analysis", content)
            self.assertIn("50.0%", content)
        finally:
            if temp_report.exists():
                temp_report.unlink()


if __name__ == "__main__":
    unittest.main()
