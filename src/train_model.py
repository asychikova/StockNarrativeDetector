# src/train_model.py
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

import joblib

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def main():
    labeled_path = DATA_DIR / "labeled_news.csv"
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"{labeled_path} not found. Create it from news_with_auto_labels.csv."
        )

    df = pd.read_csv(labeled_path, on_bad_lines="skip", engine="python")

    required_cols = {"title", "description", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in labeled_news.csv: {missing}. "
            f"Expected at least: {required_cols}"
        )

    df = df.dropna(subset=["title", "label"])
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    if "full_text" in df.columns:
        df["full_text"] = df["full_text"].fillna("")
        df["text"] = (
            df["title"].astype(str)
            + " "
            + df["description"].astype(str)
            + " "
            + df["full_text"].astype(str)
        )
    else:
        df["text"] = df["title"].astype(str) + " " + df["description"].astype(str)

    df = df[(df["text"].str.strip() != "") & (df["label"].str.strip() != "")]
    if df.empty:
        raise ValueError("No valid rows left after cleaning. Check labeled_news.csv.")

    print("Label distribution:")
    print(df["label"].value_counts())

    label_counts_series = df["label"].value_counts()
    target_n = 80


    print(f"\nTarget samples per class for balancing: {target_n}")

    balanced_frames = []
    for label, group in df.groupby("label"):
        group_bal = resample(
            group,
            n_samples=target_n,
            replace=len(group) < target_n,  # oversample if too few
            random_state=42,
        )
    balanced_frames.append(group_bal)
    print("Original label distribution:")
    print(label_counts_series)
    print("\nBalanced label distribution:")
    print(df["label"].value_counts())


    print("\nBalanced label distribution:")
    print(df["label"].value_counts())
    # ----------------------------------------

    X = df["text"].values
    y = df["label"].values

    label_counts = Counter(y)
    unique_labels = list(label_counts.keys())

    if len(unique_labels) < 2:
        raise ValueError(
            f"Need at least 2 classes to train a classifier, found: {unique_labels}"
        )

    min_count = min(label_counts.values())
    use_stratify = min_count >= 2

    if use_stratify:
        stratify_arg = y
        print("Using stratified train/test split.")
    else:
        stratify_arg = None
        print(
            "Not using stratified split because at least one label has < 2 samples. "
            "Consider labeling a few more examples."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_arg,
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    stop_words="english",
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced")),
        ]
    )

    print("Training model...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    labels_sorted = sorted(df["label"].unique())
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=labels_sorted))
    print("Labels order:", labels_sorted)

    model_path = MODELS_DIR / "narrative_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved model to {model_path}")


if __name__ == "__main__":
    main()

