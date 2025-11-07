"""
sentiment_analysis.py
======================

This script implements a basic sentiment analysis pipeline for classifying movie reviews
as positive or negative. It uses scikit‑learn to train a logistic regression classifier
on a labelled dataset.  The code is organised into functions to load data, train a
model, evaluate performance and make predictions on new text.

Example usage:

    # Train the model and save it to disk
    python sentiment_analysis.py --train

    # Evaluate the model on the held‑out test set
    python sentiment_analysis.py --evaluate

    # Predict the sentiment of a custom review
    python sentiment_analysis.py --predict "This film was hilarious and heart‑warming."

The training data is drawn from the NLTK `movie_reviews` corpus by default.  To use
your own CSV file with `text` and `label` columns, supply the `--dataset` argument.
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

try:
    import nltk
    from nltk.corpus import movie_reviews, stopwords
    nltk_available = True
except ImportError:
    nltk_available = False


def load_nltk_movie_reviews() -> Tuple[List[str], List[str]]:
    """Load the NLTK movie reviews dataset.  Returns a tuple of texts and labels."""
    if not nltk_available:
        raise RuntimeError("NLTK is not installed; cannot load the movie_reviews corpus.")
    # Download required corpora lazily
    nltk.download("movie_reviews", quiet=True)
    nltk.download("stopwords", quiet=True)
    texts, labels = [], []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            texts.append(movie_reviews.raw(fileid))
            labels.append(category)
    return texts, labels


def load_csv_dataset(path: str) -> Tuple[List[str], List[str]]:
    """Load a custom CSV dataset with 'text' and 'label' columns."""
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV dataset must contain 'text' and 'label' columns.")
    return df["text"].tolist(), df["label"].tolist()


def build_pipeline(stop_words: List[str] = None) -> Pipeline:
    """Construct a scikit‑learn Pipeline for TF‑IDF vectorisation and classification."""
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    classifier = LogisticRegression(max_iter=1000)
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train_model(texts: List[str], labels: List[str]) -> Pipeline:
    """Train the classifier and return the fitted pipeline."""
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    stop_words = stopwords.words("english") if nltk_available else None
    pipeline = build_pipeline(stop_words)
    pipeline.fit(X_train, y_train)
    # Evaluate on test set
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}")
    print(classification_report(y_test, preds))
    return pipeline


def evaluate_model(pipeline: Pipeline, texts: List[str], labels: List[str]):
    """Evaluate a trained pipeline on the provided dataset."""
    preds = pipeline.predict(texts)
    acc = accuracy_score(labels, preds)
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(labels, preds))


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate a sentiment classifier.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model and save it to sentiment_model.pkl",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate a saved model on the NLTK movie reviews test split.",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Predict the sentiment of the provided review text.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to a CSV file with 'text' and 'label' columns. If not provided, NLTK movie_reviews will be used.",
    )
    args = parser.parse_args()

    # Determine dataset
    if args.dataset:
        texts, labels = load_csv_dataset(args.dataset)
    else:
        texts, labels = load_nltk_movie_reviews()

    model_path = Path("sentiment_model.pkl")

    if args.train:
        print("Training the sentiment model...")
        pipeline = train_model(texts, labels)
        joblib.dump(pipeline, model_path)
        print(f"Model saved to {model_path}")
        return

    # Load existing model for evaluation or prediction
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Train the model first with --train."
        )
    pipeline: Pipeline = joblib.load(model_path)

    if args.evaluate:
        print("Evaluating the sentiment model...")
        evaluate_model(pipeline, texts, labels)

    if args.predict:
        review = args.predict
        pred_label = pipeline.predict([review])[0]
        print(f"Review: {review}")
        print(f"Predicted sentiment: {pred_label}")


if __name__ == "__main__":
    main()
