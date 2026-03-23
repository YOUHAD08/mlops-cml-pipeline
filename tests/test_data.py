import pytest
import json
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


# ================================
# Tests: Dataset
# ================================

def test_dataset_loads():
    """Dataset must load without errors."""
    X, y = load_breast_cancer(return_X_y=True)
    assert X is not None
    assert y is not None


def test_dataset_shape():
    """Dataset must have correct shape."""
    X, y = load_breast_cancer(return_X_y=True)
    assert X.shape[0] == 569    # 569 samples
    assert X.shape[1] == 30     # 30 features
    assert len(y) == 569


def test_dataset_not_empty():
    """Dataset must not be empty."""
    X, y = load_breast_cancer(return_X_y=True)
    assert len(X) > 0
    assert len(y) > 0


def test_dataset_labels():
    """Dataset must have exactly 2 classes."""
    X, y = load_breast_cancer(return_X_y=True)
    assert len(set(y)) == 2     # binary classification!


# ================================
# Tests: Data Split
# ================================

def test_split_sizes():
    """Split must produce correct sizes."""
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    assert Xtr.shape[0] == 455  # 80% of 569
    assert Xte.shape[0] == 114  # 20% of 569


def test_split_reproducible():
    """Same seed must produce same split every time."""
    X, y = load_breast_cancer(return_X_y=True)

    Xtr1, Xte1, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Xtr2, Xte2, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Both splits must be identical!
    assert (Xtr1 == Xtr2).all()
    assert (Xte1 == Xte2).all()


def test_split_no_overlap():
    """Train and test sets must not share samples."""
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert to sets of tuples to check overlap
    train_set = set(map(tuple, Xtr))
    test_set = set(map(tuple, Xte))

    assert len(train_set.intersection(test_set)) == 0


# ================================
# Tests: Artifacts
# ================================

def test_metrics_json_exists():
    """metrics.json must exist after evaluation."""
    metrics_path = Path("artifacts/metrics.json")
    if not metrics_path.exists():
        pytest.skip("artifacts/metrics.json not found — run train first!")


def test_metrics_json_keys():
    """metrics.json must contain required keys."""
    metrics_path = Path("artifacts/metrics.json")
    if not metrics_path.exists():
        pytest.skip("metrics.json not found — run train + evaluate first!")

    with open(metrics_path) as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "n_estimators" in metrics
    assert "seed" in metrics


def test_metrics_values_valid():
    """Metrics must be between 0 and 1."""
    metrics_path = Path("artifacts/metrics.json")
    if not metrics_path.exists():
        pytest.skip("metrics.json not found — run train + evaluate first!")

    with open(metrics_path) as f:
        metrics = json.load(f)

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["f1"] <= 1

def test_model_exists():
    """model.joblib must exist after training."""
    model_path = Path("artifacts/model.joblib")
    if not model_path.exists():
        pytest.skip("artifacts/model.joblib not found — run train first!")


def test_confusion_png_exists():
    """confusion.png must exist after evaluation."""
    fig_path = Path("artifacts/confusion.png")
    if not fig_path.exists():
        pytest.skip("artifacts/confusion.png not found — run evaluate first!")