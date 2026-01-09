#!/usr/bin/env python3
"""
Spam vs Legitimate email classifier (Logistic Regression)

Dataset columns expected:
- words: total number of words in the email
- links: number of URL-like links (http/https/www)
- capital_words: number of ALL-CAPS words (length>=2)
- spam_word_count: number of "spammy" keyword hits from a fixed list
- is_spam: target (1=spam, 0=legitimate)

Usage examples:
  # Train + evaluate + save model
  python -m src.app train --data data/g_jokharidze25_59612.csv --model-out models.joblib

  # Classify a typed email
  python -m src.app predict --model models.joblib

  # Classify an email from a text file
  python -m src.app predict --model models.joblib --email-file samples/spam_email.txt
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# A small, human-readable list of common spam trigger words/phrases.
# You can extend this list, but KEEP IT CONSISTENT with training/evaluation.
SPAM_KEYWORDS = [
    "free", "winner", "win", "prize", "claim", "urgent", "act now",
    "limited", "offer", "bonus", "cash", "money", "credit", "loan",
    "risk-free", "guarantee", "click", "subscribe", "unsubscribe",
    "congratulations", "deal", "investment", "bitcoin", "crypto",
    "password", "verify", "account", "suspended", "refund", "exclusive",
]

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
WORD_REGEX = re.compile(r"[A-Za-z0-9']+")

@dataclass
class EmailFeatures:
    words: int
    links: int
    capital_words: int
    spam_word_count: int

def extract_features(email_text: str) -> EmailFeatures:
    """
    Parse raw email text and extract the SAME features as in the dataset.
    """
    text = email_text.strip()

    words_list = WORD_REGEX.findall(text)
    words = len(words_list)

    links = len(URL_REGEX.findall(text))

    capital_words = sum(1 for w in words_list if len(w) >= 2 and w.isalpha() and w.isupper())

    # Count keyword hits (case-insensitive). For multi-word phrases, search in full text.
    lower_text = text.lower()
    spam_word_count = 0
    for kw in SPAM_KEYWORDS:
        if " " in kw:
            spam_word_count += lower_text.count(kw)
        else:
            # count whole-word matches
            spam_word_count += len(re.findall(rf"\b{re.escape(kw)}\b", lower_text))

    return EmailFeatures(
        words=words,
        links=links,
        capital_words=capital_words,
        spam_word_count=spam_word_count,
    )

def load_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    required_cols = {"words", "links", "capital_words", "spam_word_count", "is_spam"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    X = df[["words", "links", "capital_words", "spam_word_count"]]
    y = df["is_spam"]
    return X, y

def train_and_evaluate(data_path: Path, model_out: Path, test_size: float = 0.30, seed: int = 42) -> None:
    X, y = load_dataset(data_path)

    # Train on 70%, validate on 30% (stratified so class ratio stays similar)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Validation Results (30% holdout) ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix (rows=Actual, cols=Predicted):")
    print(cm)

    print("\n=== Model Coefficients ===")
    for name, val in zip(X.columns, model.coef_[0]):
        print(f"{name:16s} {val:+.6f}")
    print(f"{'intercept':16s} {model.intercept_[0]:+.6f}")

    joblib.dump(model, model_out)
    print(f"\nSaved model to: {model_out}")

def predict_interactive(model_path: Path, email_file: Path | None = None) -> None:
    model = joblib.load(model_path)

    if email_file:
        email_text = email_file.read_text(encoding="utf-8", errors="ignore")
    else:
        print("Paste email text. End with Ctrl-D (Linux/macOS) or Ctrl-Z then Enter (Windows):\n")
        email_text = sys.stdin.read()

    feats = extract_features(email_text)
    X = pd.DataFrame([feats.__dict__])

    prob_spam = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])

    label = "SPAM" if pred == 1 else "LEGITIMATE"
    print("\n=== Prediction ===")
    print(f"Predicted class: {label} (is_spam={pred})")
    print(f"Spam probability: {prob_spam:.4f}")
    print("\nExtracted features:")
    for k, v in feats.__dict__.items():
        print(f"- {k}: {v}")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Spam email classifier (Logistic Regression)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train + evaluate + save model")
    p_train.add_argument("--data", type=Path, required=True, help="Path to CSV dataset")
    p_train.add_argument("--model-out", type=Path, default=Path("models.joblib"), help="Output model path")
    p_train.add_argument("--test-size", type=float, default=0.30, help="Holdout fraction (default 0.30)")
    p_train.add_argument("--seed", type=int, default=42, help="Random seed")

    p_pred = sub.add_parser("predict", help="Predict spam/legitimate for an email text")
    p_pred.add_argument("--model", type=Path, required=True, help="Path to trained model .joblib")
    p_pred.add_argument("--email-file", type=Path, default=None, help="Optional path to a text file with email body")

    return p

def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.cmd == "train":
        train_and_evaluate(args.data, args.model_out, args.test_size, args.seed)
        return 0
    if args.cmd == "predict":
        predict_interactive(args.model, args.email_file)
        return 0

    return 2

if __name__ == "__main__":
    raise SystemExit(main())
