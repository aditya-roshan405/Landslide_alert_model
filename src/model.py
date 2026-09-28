"""
Landslide risk model: synthetic data generation, training, and evaluation.

Run directly to (re)train the model and write:
  - models/landslide_model.pkl
  - models/confusion_matrix.png
  - models/metrics_report.txt   (accuracy, precision/recall/F1 per class)
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

FEATURES = ["soil_moisture", "tilt_angle", "rainfall_rate"]
LABELS = {0: "Safe", 1: "Warning", 2: "Danger"}
LABEL_NAMES = [LABELS[i] for i in sorted(LABELS)]

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "landslide_model.pkl")
CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")
METRICS_REPORT_PATH = os.path.join(MODELS_DIR, "metrics_report.txt")


def classify_risk(row: pd.Series) -> int:
    """Rule-based ground-truth label for synthetic training data."""
    if row["soil_moisture"] > 70 and row["tilt_angle"] > 25 and row["rainfall_rate"] > 100:
        return 2  # Danger
    if row["soil_moisture"] > 40 and row["tilt_angle"] > 15 and row["rainfall_rate"] > 50:
        return 1  # Warning
    return 0  # Safe


def generate_synthetic_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a labeled synthetic dataset of sensor readings."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "soil_moisture": rng.uniform(0, 100, n),
            "tilt_angle": rng.uniform(0, 45, n),
            "rainfall_rate": rng.uniform(0, 200, n),
        }
    )
    df["risk_level"] = df.apply(classify_risk, axis=1)
    return df


def train_model(df: pd.DataFrame, seed: int = 42):
    """Split, train, and return (model, X_test, y_test)."""
    X = df[FEATURES]
    y = df["risk_level"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute accuracy, per-class precision/recall/F1, and confusion matrix."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(LABELS))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=sorted(LABELS), zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, target_names=LABEL_NAMES, zero_division=0
    )

    per_class = {
        LABELS[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(LABELS))
    }

    return {
        "accuracy": float(accuracy),
        "confusion_matrix": cm,
        "per_class": per_class,
        "report_text": report_text,
    }


def save_confusion_matrix_plot(cm, path: str = CONFUSION_MATRIX_PATH) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Landslide Risk - Confusion Matrix")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = generate_synthetic_data()
    print("Class distribution:\n", df["risk_level"].value_counts(), "\n")

    model, X_test, y_test = train_model(df)
    results = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {results['accuracy']:.4f}\n")
    print("Classification Report:\n", results["report_text"])
    print("Confusion Matrix:\n", results["confusion_matrix"])

    joblib.dump(model, MODEL_PATH)
    save_confusion_matrix_plot(results["confusion_matrix"])

    with open(METRICS_REPORT_PATH, "w") as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n\n")
        f.write(results["report_text"])
        f.write("\nConfusion Matrix (rows=actual, cols=predicted):\n")
        f.write(str(results["confusion_matrix"]))

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Confusion matrix plot saved to {CONFUSION_MATRIX_PATH}")
    print(f"Metrics report saved to {METRICS_REPORT_PATH}")


if __name__ == "__main__":
    main()
