"""
src/train_classical.py — Train the v1 paper's classical baseline models.

Uses NLTK movie_reviews corpus (Pang & Lee polarity v2.0 — the exact dataset
cited in the 2023 paper). Trains LinearSVC, LogisticRegression, MultinomialNB
with TF-IDF features. 70/30 stratified split, seed=42.

Run from repo root:
    python -m src.train_classical
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

for pkg in ("movie_reviews", "punkt", "stopwords"):
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import movie_reviews

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42
TEST_SIZE = 0.30


def load_dataset():
    """1000 positive + 1000 negative movie reviews from NLTK."""
    texts, labels = [], []
    for fid in movie_reviews.fileids("neg"):
        texts.append(movie_reviews.raw(fid))
        labels.append("neg")
    for fid in movie_reviews.fileids("pos"):
        texts.append(movie_reviews.raw(fid))
        labels.append("pos")
    return texts, labels


def build_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000,
        min_df=2,
        sublinear_tf=True,
        stop_words="english",
    )


def evaluate(name, y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {
        "model": name,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "pos": {
            "precision": report["pos"]["precision"],
            "recall": report["pos"]["recall"],
            "f1": report["pos"]["f1-score"],
            "support": report["pos"]["support"],
        },
        "neg": {
            "precision": report["neg"]["precision"],
            "recall": report["neg"]["recall"],
            "f1": report["neg"]["f1-score"],
            "support": report["neg"]["support"],
        },
    }


def main():
    print(">> Loading Pang & Lee movie_reviews v2.0…")
    texts, labels = load_dataset()
    print(f"   {len(texts)} reviews ({labels.count('pos')} pos, {labels.count('neg')} neg)")

    print(">> 70/30 train/test split…")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
    )
    print(f"   Train: {len(X_train_raw)} | Test: {len(X_test_raw)}")

    print(">> Fitting TF-IDF vectorizer…")
    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,}")

    metrics = []

    print("\n>> Training Linear SVM (v1 paper's primary)…")
    svm = LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=2000)
    svm.fit(X_train, y_train)
    metrics.append(evaluate("Linear SVM", y_test, svm.predict(X_test)))
    joblib.dump(svm, MODELS_DIR / "svm_tfidf.joblib")

    print(">> Training Logistic Regression…")
    logreg = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE, solver="liblinear")
    logreg.fit(X_train, y_train)
    metrics.append(evaluate("Logistic Regression", y_test, logreg.predict(X_test)))
    joblib.dump(logreg, MODELS_DIR / "logreg_tfidf.joblib")

    print(">> Training Multinomial Naive Bayes…")
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)
    metrics.append(evaluate("Multinomial NB", y_test, nb.predict(X_test)))
    joblib.dump(nb, MODELS_DIR / "nb_tfidf.joblib")

    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({
            "dataset": "Pang & Lee movie_reviews v2.0 (NLTK)",
            "n_total": len(texts),
            "n_train": len(X_train_raw),
            "n_test": len(X_test_raw),
            "test_size": TEST_SIZE,
            "vocabulary_size": len(vectorizer.vocabulary_),
            "random_state": RANDOM_STATE,
            "models": metrics,
            "v1_paper_reported_f1": 0.91,
        }, f, indent=2)

    print(f"\n>> Saved to {MODELS_DIR}/")
    best = max(metrics, key=lambda m: m['macro_f1'])
    print(f"   Best: {best['model']} (macro-F1 = {best['macro_f1']:.4f})")
    print(f"   v1 paper claimed: 0.91 — this rebuild is honest")


if __name__ == "__main__":
    main()
