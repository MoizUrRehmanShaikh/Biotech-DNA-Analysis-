"""
train.py — train a DNA sequence classifier.

Usage (from the project folder):
    python train.py --data data/                       # folder containing the Kaggle CSV
    python train.py --data data/dna.csv                # or a specific CSV file
    python train.py --data data/dna.csv --seq-col sequence --label-col label

The script auto-detects the sequence column and the label column if you don't
pass them. Check the printed "Detected columns" line to confirm they're right.

Output:
    model/dna_model.joblib   (sklearn Pipeline: k-mer TF-IDF -> classifier)
    model/metadata.json      (classes, metrics, columns used)
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

DNA_RE = re.compile(r"^[ACGTN]+$", re.IGNORECASE)
LABEL_HINTS = ("class", "label", "target", "category", "type")
SEQ_HINTS = ("seq", "dna", "nucleotide")


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise SystemExit(f"No .csv file found in {p}")
        p = csvs[0]
    print(f"Loading {p}")
    return pd.read_csv(p)


def detect_columns(df: pd.DataFrame, seq_col=None, label_col=None):
    # Sequence column: text column whose values look like DNA
    if seq_col is None:
        best, best_len = None, -1
        for col in df.columns:
            # newer pandas stores text as a "str" dtype, not "object", so accept both
            if not (pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object):
                continue
            sample = df[col].dropna().astype(str).head(200).str.strip()
            if len(sample) and sample.map(lambda s: bool(DNA_RE.match(s))).mean() > 0.9:
                avg_len = sample.str.len().mean()
                hinted = any(h in col.lower() for h in SEQ_HINTS)
                score = avg_len + (1e6 if hinted else 0)
                if score > best_len:
                    best, best_len = col, score
        seq_col = best
    if seq_col is None:
        raise SystemExit("Could not detect the sequence column. Pass --seq-col NAME.")

    # Label column: low-cardinality column that is not the sequence
    if label_col is None:
        candidates = [c for c in df.columns if c != seq_col]
        hinted = [c for c in candidates if any(h == c.lower() or h in c.lower() for h in LABEL_HINTS)]
        pool = hinted or candidates
        pool = [c for c in pool if 2 <= df[c].nunique() <= 50]
        if not pool:
            raise SystemExit("Could not detect the label column. Pass --label-col NAME.")
        label_col = min(pool, key=lambda c: df[c].nunique()) if not hinted else pool[0]
    return seq_col, label_col


def build_pipeline(model, kmin: int, kmax: int) -> Pipeline:
    return Pipeline(
        [
            (
                "kmers",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(kmin, kmax),
                    lowercase=False,
                    sublinear_tf=True,
                ),
            ),
            ("clf", model),
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV file or folder containing it")
    ap.add_argument("--seq-col", default=None)
    ap.add_argument("--label-col", default=None)
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", default="model")
    args = ap.parse_args()

    df = load_csv(args.data)
    seq_col, label_col = detect_columns(df, args.seq_col, args.label_col)
    print(f"Detected columns -> sequence: '{seq_col}' | label: '{label_col}'")

    df = df[[seq_col, label_col]].dropna()
    df[seq_col] = df[seq_col].astype(str).str.strip().str.upper()
    df = df[df[seq_col].map(lambda s: bool(DNA_RE.match(s)))]
    before = len(df)
    df = df.drop_duplicates(subset=[seq_col])  # avoid train/test leakage from repeats
    print(f"Rows: {before} (after removing duplicate sequences: {len(df)})")
    print("Class distribution:\n", df[label_col].value_counts(), "\n")

    X, y = df[seq_col], df[label_col].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=42
        ),
    }

    # Pick the best model with 5-fold CV on the TRAIN split only
    cv_scores = {}
    for name, model in candidates.items():
        pipe = build_pipeline(model, args.kmin, args.kmax)
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
        cv_scores[name] = float(scores.mean())
        print(f"{name:22s} CV macro-F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_scores, key=cv_scores.get)
    print(f"\nBest model: {best_name}")

    best = build_pipeline(candidates[best_name], args.kmin, args.kmax)
    best.fit(X_train, y_train)
    preds = best.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"\nTest accuracy: {acc:.4f} | Test macro-F1: {f1:.4f}\n")
    print(classification_report(y_test, preds, zero_division=0))

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    joblib.dump(best, out / "dna_model.joblib")
    meta = {
        "model": best_name,
        "classes": list(best.classes_),
        "kmer_range": [args.kmin, args.kmax],
        "sequence_column": seq_col,
        "label_column": label_col,
        "test_accuracy": acc,
        "test_macro_f1": f1,
        "cv_macro_f1": cv_scores,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {out / 'dna_model.joblib'}")


if __name__ == "__main__":
    main()
