# sql_injection_pipeline.py
# -----------------------------------------------------------
# End-to-end pipeline for SQL Injection detection
# - Preprocess (clean, TF-IDF, handcrafted features)
# - Train Random Forest
# - Evaluate (confusion matrix, classification report, ROC-AUC)
# - Save artifacts & model
# - Optional SHAP local explanations (stable settings for sparse TF-IDF)
# - Optional encrypted + hashed logging of predictions via secure_log.py
# -----------------------------------------------------------

import os
import re
import sys
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.sparse import hstack, csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import joblib

# ---------------- Config defaults (updated for your folder) ----------------
BASE_DIR = r"C:\Users\HP\Downloads\DB_Sql_injection_proj"
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "data", "Modified_SQL_Dataset.csv")
DEFAULT_ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_ARTIFACT_DIR, "rf_model.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.30
MAX_TFIDF_FEATURES = 3000
NGRAM_RANGE = (1, 2)

# ---------------- Regex patterns ----------------
SQL_KEYWORD_PATTERN = re.compile(
    r"\b(select|insert|update|delete|drop|union|create|from|where|and|or|exec|into|values|join|order|group|by|having)\b",
    re.IGNORECASE
)
SPECIAL_CHARS_PATTERN = re.compile(r"['\";#\-]")  # ' " ; # -

# ---------------- Text utils ----------------
def clean_query(text: str) -> str:
    """Lowercase, normalize comments/dashes, strip punctuation noise, collapse spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.replace("--", " -- ")
    text = re.sub(r"['\";#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_handcrafted_features(cleaned: str) -> Tuple[int, int, int, int, float]:
    """(length, special_chars, digits, sql_keywords, digit_ratio)"""
    length = len(cleaned)
    special_chars = len(SPECIAL_CHARS_PATTERN.findall(cleaned))
    digits = sum(ch.isdigit() for ch in cleaned)
    sql_keywords = len(SQL_KEYWORD_PATTERN.findall(cleaned))
    digit_ratio = (digits / length) if length > 0 else 0.0
    return length, special_chars, digits, sql_keywords, digit_ratio

# ---------------- Core pipeline steps ----------------
def preprocess(data_path: str,
               artifact_dir: str,
               max_tfidf_features: int = MAX_TFIDF_FEATURES,
               ngram_range = NGRAM_RANGE,
               test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE):
    """Load CSV, clean, vectorize, add features, split, save artifacts."""
    if not os.path.exists(data_path):
        print(f"[ERROR] File not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(data_path)
    expected_cols = {"Query", "Label"}
    if not expected_cols.issubset(df.columns):
        print(f"[ERROR] CSV must contain columns {expected_cols}. Found: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    df = df.dropna(subset=["Query", "Label"]).copy()
    df["Label"] = df["Label"].astype(int)

    # Clean
    df["Cleaned_Query"] = df["Query"].apply(clean_query)

    # Handcrafted features
    feats = df["Cleaned_Query"].apply(extract_handcrafted_features)
    df[["Length", "SpecialChars", "Digits", "SQLKeywords", "DigitRatio"]] = pd.DataFrame(
        feats.tolist(), index=df.index
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_tfidf_features, ngram_range=ngram_range)
    X_tfidf = vectorizer.fit_transform(df["Cleaned_Query"])

    # Combine
    X_extra = df[["Length", "SpecialChars", "Digits", "SQLKeywords", "DigitRatio"]].to_numpy(dtype=float)
    X_combined = hstack([X_tfidf, csr_matrix(X_extra)], format="csr")
    y = df["Label"].to_numpy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Report
    print("=== Preprocessing Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Train size: {X_train.shape[0]}   Test size: {X_test.shape[0]}")
    print(f"TF-IDF features: {X_tfidf.shape[1]}   Handcrafted: 5   Total: {X_combined.shape[1]}")
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Label distribution (train): {dict(zip(unique, counts))}")
    unique, counts = np.unique(y_test, return_counts=True)
    print(f"Label distribution (test):  {dict(zip(unique, counts))}")

    # Save artifacts
    os.makedirs(artifact_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(artifact_dir, "tfidf_vectorizer.pkl"))
    save_npz(os.path.join(artifact_dir, "X_train.npz"), X_train)
    save_npz(os.path.join(artifact_dir, "X_test.npz"), X_test)
    np.save(os.path.join(artifact_dir, "y_train.npy"), y_train)
    np.save(os.path.join(artifact_dir, "y_test.npy"), y_test)

    # Sample rows to verify cleaning
    df.sample(5, random_state=random_state)[["Query", "Cleaned_Query", "Label"]].to_csv(
        os.path.join(artifact_dir, "sample_rows.csv"), index=False
    )

    print(f"\nArtifacts saved to: {os.path.abspath(artifact_dir)}")
    print(" - tfidf_vectorizer.pkl")
    print(" - X_train.npz, X_test.npz")
    print(" - y_train.npy, y_test.npy")
    print(" - sample_rows.csv")

def train_and_eval(artifact_dir: str,
                   model_path: str,
                   n_estimators: int = 200,
                   random_state: int = RANDOM_STATE):
    """Train RandomForest on saved artifacts and print metrics."""
    X_train = load_npz(os.path.join(artifact_dir, "X_train.npz"))
    X_test  = load_npz(os.path.join(artifact_dir, "X_test.npz"))
    y_train = np.load(os.path.join(artifact_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(artifact_dir, "y_test.npy"))

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=random_state
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    try:
        y_proba = rf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    print("\n=== Evaluation ===")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
    if auc is not None:
        print(f"ROC-AUC: {auc:.4f}")

    joblib.dump(rf, model_path)
    print(f"\nSaved model to: {os.path.abspath(model_path)}")

def load_and_predict(artifact_dir: str, model_path: str, raw_query: str, log_preds: bool = False):
    """Predict a single raw query string; optionally encrypt + hash log."""
    vectorizer = joblib.load(os.path.join(artifact_dir, "tfidf_vectorizer.pkl"))
    rf = joblib.load(model_path)

    cleaned = clean_query(raw_query)
    length, special, digits, sql_kw, digit_ratio = extract_handcrafted_features(cleaned)
    X_text = vectorizer.transform([cleaned])
    X_extra = csr_matrix(np.array([[length, special, digits, sql_kw, digit_ratio]], dtype=float))
    X = hstack([X_text, X_extra], format="csr")

    pred = rf.predict(X)[0]
    proba = rf.predict_proba(X)[0, 1]

    if log_preds:
        try:
            from secure_log import encrypt_and_append, hash_record
            record = {
                "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "query": raw_query,
                "pred": int(pred),
                "proba": float(proba)
            }
            record["hash"] = hash_record(record)
            encrypt_and_append(record)
        except Exception as e:
            print(f"[Warn] Failed to secure-log prediction: {e}")

    return pred, proba

# ---------------- CLI ----------------
def build_argparser():
    p = argparse.ArgumentParser(description="SQL Injection Detection Pipeline (Preprocess + Train + Evaluate)")
    p.add_argument("--data", default=DEFAULT_DATASET_PATH, help="Path to Modified_SQL_Dataset.csv")
    p.add_argument("--artifacts", default=DEFAULT_ARTIFACT_DIR, help="Output dir for artifacts")
    p.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to save RandomForest model")
    p.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing (reuse existing artifacts)")
    p.add_argument("--no-shap", action="store_true", help="Skip SHAP explanations")
    p.add_argument("--predict", type=str, default=None, help="Predict on a single raw SQL query")
    p.add_argument("--log-preds", action="store_true", help="Encrypt-log predictions (requires secure_log.py)")
    p.add_argument("--n-estimators", type=int, default=200, help="RandomForest n_estimators")
    return p

def main():
    args = build_argparser().parse_args()

    if not args.skip_preprocess:
        preprocess(args.data, args.artifacts)
    else:
        print("[Info] Skipping preprocessing. Using existing artifacts…")

    train_and_eval(args.artifacts, args.model, n_estimators=args.n_estimators)

    if args.predict:
        pred, proba = load_and_predict(args.artifacts, args.model, args.predict, log_preds=args.log_preds)
        label = "ATTACK (1)" if pred == 1 else "NORMAL (0)"
        print(f"\nPrediction for query:\n  {args.predict}\n=> {label} | probability={proba:.4f}")

if __name__ == "__main__":
    main()
