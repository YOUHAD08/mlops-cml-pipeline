import json
import os
from pathlib import Path


def md_table_single(title, metrics):
    """Create a simple one-row markdown table."""
    rows = [
        "| Item | accuracy | f1 |",
        "| ---| ---:| ---:|",
        f"| {title} | {metrics['accuracy']:.6f} | {metrics['f1']:.6f} |",
    ]
    return "\n".join(rows)


def md_table_compare(base, pr):
    """Create baseline vs PR comparison table."""
    def delta(a, b):
        return b - a

    rows = [
        "| Metric | Baseline (main) | PR | Delta (PR - Base) |",
        "| ---| ---:| ---:| ---:|",
        f"| accuracy | {base['accuracy']:.6f} | {pr['accuracy']:.6f} | {delta(base['accuracy'], pr['accuracy']):+.6f} |",
        f"| f1 | {base['f1']:.6f} | {pr['f1']:.6f} | {delta(base['f1'], pr['f1']):+.6f} |",
    ]
    return "\n".join(rows)


def md_table_matrix(matrix):
    """Create matrix details table."""
    rows = [
        "| N_ESTIMATORS | accuracy | f1 |",
        "| ---:| ---:| ---:|",
    ]
    for r in matrix:
        rows.append(
            f"| {r['n_estimators']} | {r['accuracy']:.6f} | {r['f1']:.6f} |"
        )
    return "\n".join(rows)


def check_thresholds(metrics, min_accuracy=0.90, min_f1=0.95):
    """Check if metrics meet minimum thresholds."""
    warnings = []
    if metrics["accuracy"] < min_accuracy:
        warnings.append(
            f"⚠️ accuracy {metrics['accuracy']:.6f} < threshold {min_accuracy}"
        )
    if metrics["f1"] < min_f1:
        warnings.append(
            f"⚠️ f1 {metrics['f1']:.6f} < threshold {min_f1}"
        )
    return warnings


def main():
    # 1. Create artifacts folder
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    # 2. Read aggregated PR metrics
    with open("artifacts/metrics_pr.json", "r", encoding="utf-8") as f:
        pr = json.load(f)

    # 3. Try to read baseline metrics
    baseline_path = Path("artifacts/baseline_metrics.json")
    base = None
    if baseline_path.exists():
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    base = json.loads(content)
                else:
                    print("Baseline file is empty — skipping!")
        except json.JSONDecodeError:
            print("Baseline file is invalid JSON — skipping!")

    # 4. Extract notes + matrix
    notes = pr.get("notes", {})
    train_seconds = notes.get("train_seconds", "n/a")
    eval_seconds = notes.get("eval_seconds", "n/a")
    matrix = pr.get("matrix", [])

    # 5. Check thresholds
    warnings = check_thresholds(pr)

    # 6. Build report section by section
    report = []

    # Title
    report.append("# 🚀 MLOps CML Pipeline Report\n")

    # Section 1: PR Summary
    report.append("## 📊 PR Summary (aggregated over matrix)\n")
    report.append(md_table_single("PR (avg over matrix)", pr))
    report.append("\n")

    # Section 2: Baseline vs PR
    report.append("## 🔁 Baseline vs PR\n")
    if base is None:
        report.append(
            "_Baseline not found — first run or main has no artifacts yet._"
        )
    else:
        report.append(md_table_compare(base, pr))
    report.append("\n")

    # Section 3: Matrix details
    report.append("## 🧪 Experiment Matrix Details\n")
    report.append(md_table_matrix(matrix))
    report.append("\n")

    # Section 4: Figure
    report.append("## 🖼️ Confusion Matrix\n")
    report.append("![confusion](artifacts/confusion.png)\n")

    # Section 5: Notes
    report.append("## 📝 Notes\n")
    report.append(f"- train_seconds (last run): `{train_seconds}`")
    report.append(f"- eval_seconds (last run): `{eval_seconds}`")
    report.append(f"- runs_in_matrix: `{len(matrix)}`")

    # Section 6: Warnings
    report.append("\n## ⚠️ Warnings\n")
    if warnings:
        for w in warnings:
            report.append(f"- {w}")
    else:
        report.append("- ✅ All metrics meet thresholds!")

    report.append("\n")

    # 7. Save report.md
    with open("report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print("Report built → report.md")

    # 8. Fail CI if thresholds not met! (optional extension)
    if warnings:
        print("❌ Quality thresholds not met!")
        exit(1)
    else:
        print("✅ All quality thresholds met!")


if __name__ == "__main__":
    main()