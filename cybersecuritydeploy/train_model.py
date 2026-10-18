
import json
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_PATH = "data/network_traffic.csv"

FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "SYN Flag Count",
    "ACK Flag Count",
]


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Place the labeled CSV in the data folder."
        )

    print("Loading cybersecurity dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()

    # Normalize column names and find the label column.
    label_column = next(
        (c for c in df.columns if c.lower() == "label"),
        None,
    )

    if label_column is None:
        raise ValueError(
            "No Label column found. Use a labeled CIC-IDS2017 CSV."
        )

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns include: {list(df.columns[:25])}"
        )

    # Convert feature values to numbers and clean invalid values.
    X = df[FEATURES].copy()

    for column in FEATURES:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([float("inf"), float("-inf")], float("nan"))

    labels = df[label_column].astype(str).str.strip().str.upper()

    # Binary classification: BENIGN versus any attack label.
    y = (labels != "BENIGN").astype(int)

    # Remove rows with no usable features.
    usable_rows = X.notna().any(axis=1)
    X = X.loc[usable_rows].reset_index(drop=True)
    y = y.loc[usable_rows].reset_index(drop=True)

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(
            "The dataset must contain at least two benign records "
            "and two attack records."
        )

    print(f"Usable records: {len(X):,}")
    print("Class counts (0 = benign, 1 = attack):")
    print(y.value_counts().sort_index())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "test_records": int(len(y_test)),
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "recall": round(
            recall_score(y_test, predictions, zero_division=0), 4
        ),
        "f1_score": round(
            f1_score(y_test, predictions, zero_division=0), 4
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
    }

    joblib.dump(model, "model.joblib")

    with open("features.json", "w", encoding="utf-8") as file:
        json.dump(FEATURES, file, indent=4)

    with open("metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nMODEL EVALUATION")
    print(json.dumps(metrics, indent=4))
    print("\nSaved model.joblib, features.json and metrics.json")


if __name__ == "__main__":
    main()
