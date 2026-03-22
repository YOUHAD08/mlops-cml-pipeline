import json
import time
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def save_confusion_png(cm, out_path: str):
    """Save confusion matrix as PNG figure."""
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Add numbers inside each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Figure saved → {out_path}")


def main():
    # 1. Start timer
    t0 = time.time()

    # 2. Create artifacts folder
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    # 3. Read metadata from train.py
    with open("artifacts/run_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    # 4. Load dataset
    X, y = load_breast_cancer(return_X_y=True)

    # 5. Reproduce EXACT same split using seed from meta!
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.2,
        random_state=meta["seed"],
        stratify=y
    )

    # 6. Load saved model
    model = joblib.load("artifacts/model.joblib")

    # 7. Predict
    pred = model.predict(Xte)

    # 8. Calculate metrics
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred)
    cm = confusion_matrix(yte, pred)

    # 9. Save confusion matrix figure
    save_confusion_png(cm, "artifacts/confusion.png")

    # 10. Save full classification report
    report = classification_report(yte, pred, output_dict=True)
    with open("artifacts/report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # 11. Save metrics.json
    metrics = {
        "accuracy": round(float(acc), 6),
        "f1": round(float(f1), 6),
        "eval_seconds": round(time.time() - t0, 4),
        "n_estimators": meta["n_estimators"],
        "seed": meta["seed"],
    }
    with open("artifacts/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 12. Logs
    print("Evaluate OK")
    print(f"accuracy = {acc:.6f}")
    print(f"f1 = {f1:.6f}")
    print(f"eval_seconds = {metrics['eval_seconds']}")


if __name__ == "__main__":
    main()