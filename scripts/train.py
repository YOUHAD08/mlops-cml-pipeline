import os
import json
import time
from pathlib import Path
import joblib
import numpy as np
import yaml
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def load_config():
    with open("config/train.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # 1. Start timer
    t0 = time.time()

    # 2. Create artifacts folder
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    # 3. Load config
    cfg = load_config()

    # 4. Read from environment variables (for CI matrix)
    #    If not set → use config file values!
    n_estimators = int(os.getenv(
        "N_ESTIMATORS",
        str(cfg["model"]["n_estimators"])
    ))
    seed = int(os.getenv(
        "SEED",
        str(cfg["model"]["random_state"])
    ))

    # 5. Load dataset
    X, y = load_breast_cancer(return_X_y=True)

    # 6. Reproducible split
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=seed,
        stratify=y
    )

    # 7. Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=cfg["model"]["n_jobs"]
    )
    model.fit(Xtr, ytr)

    # 8. Save model
    joblib.dump(model, "artifacts/model.joblib")

    # 9. Save metadata
    meta = {
        "n_estimators": n_estimators,
        "seed": seed,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "train_seconds": round(time.time() - t0, 4),
    }
    with open("artifacts/run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # 10. Logs
    print("Train OK")
    print(f"n_estimators = {n_estimators}")
    print(f"seed = {seed}")
    print(f"n_train = {Xtr.shape[0]}")
    print(f"n_test = {Xte.shape[0]}")
    print(f"train_seconds = {meta['train_seconds']}")


if __name__ == "__main__":
    main()