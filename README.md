# 🚀 MLOps CML Pipeline

A fully industrialized MLOps project combining:

- **Git** → code versioning + collaboration
- **CML** → automated ML reports in Pull Requests
- **GitHub Actions** → CI/CD pipeline
- **Experiment Matrix** → test multiple configs automatically

---

## 📋 Project Structure

```
mlops-cml-pipeline/
├── README.md
├── requirements.txt
├── config/
│   └── train.yaml         ← hyperparameters + thresholds
├── src/
│   ├── data.py
│   ├── features.py
│   └── model.py
├── scripts/
│   ├── train.py           ← training script
│   ├── evaluate.py        ← evaluation + figures
│   └── make_report.py     ← builds PR report
├── tests/
│   └── test_data.py       ← automated tests
├── notebooks/             ← exploration only
├── data/                  ← NOT tracked by Git (DVC later!)
├── artifacts/             ← NOT tracked by Git
└── .github/
    └── workflows/
        └── cml.yml        ← full CI/CD pipeline
```

---

## ⚙️ Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/mlops-cml-pipeline.git
cd mlops-cml-pipeline

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# source .venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run training:

```bash
python scripts/train.py
```

### Run evaluation:

```bash
python scripts/evaluate.py
```

### Run tests:

```bash
pytest tests/ -v
```

---

## 🔁 CI/CD Pipeline

Every Pull Request targeting `main` automatically:

1. Runs tests (`pytest -q`)
2. Trains model with 3 configs (n_estimators: 100, 200, 400)
3. Evaluates each config
4. Aggregates results
5. Compares with baseline (main branch)
6. Posts a full report as PR comment

---

## 📊 Automated PR Report contains:

- ✅ Aggregated metrics table (accuracy + f1)
- ✅ Baseline vs PR comparison
- ✅ Experiment matrix details
- ✅ Confusion matrix figure
- ✅ Notes (timing, config)
- ✅ Quality threshold warnings

---

## 🌿 Branch Strategy

```
main     → stable releases only (tagged)
dev      → integration branch
feature/ → short-lived work branches
```

---

## 🏷️ Releases

| Version | Description                                           |
| ------- | ----------------------------------------------------- |
| v0.1.0  | Baseline model (Random Forest, breast cancer dataset) |

---

## ⚠️ Quality Thresholds

| Metric   | Minimum |
| -------- | ------- |
| accuracy | 0.90    |
| f1       | 0.95    |

> CI fails automatically if thresholds not met!

---

## 🛠️ Tech Stack

| Tool           | Purpose              |
| -------------- | -------------------- |
| scikit-learn   | ML model + metrics   |
| CML            | Automated PR reports |
| GitHub Actions | CI/CD pipeline       |
| pytest         | Automated testing    |
| pyyaml         | Config management    |
| joblib         | Model serialization  |

---

_MLOps project — Prof. Soufiane HAMIDA — 2025/2026_
