"""Build Final_Project_Report.pdf from report/report_template.html.

Every number in the report comes from results/metrics.csv, so the report always
matches the latest run of run_all.py. Placeholders in the template:

    {{xgboost.roc_auc}}      a metric for one model (4 decimals)
    {{xgboost.recall:pct}}   the same metric as a percentage (1 decimal)
    {{xgboost.fp:int}}       an integer metric with thousands separators
    {{RESULTS_TABLE}}        the full results table
    {{CONFUSION_TABLE}}      confusion-matrix counts for every model

The PDF is printed with Microsoft Edge (or Chrome) in headless mode.

Usage:
    python report/build_report.py
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent
TEMPLATE = REPORT_DIR / "report_template.html"
HTML_OUT = REPORT_DIR / "report.html"
PDF_OUT = ROOT / "Final_Project_Report.pdf"

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chrome",
]

SLUGS = {
    "Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
    "Random Forest": "random_forest", "SVM": "svm", "XGBoost": "xgboost",
    "MLP": "mlp", "DNN": "dnn", "CNN": "cnn", "LSTM": "lstm", "Autoencoder": "autoencoder",
}
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def load_metrics():
    df = pd.read_csv(ROOT / "results" / "metrics.csv")
    df["slug"] = df["model"].map(SLUGS)
    return df


def results_table(df):
    best = {c: df[c].max() for c in METRIC_COLS}
    head = ("<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th>"
            "<th>F1-score</th><th>ROC AUC</th><th>PR AUC</th><th>Train time (s)</th></tr>")
    rows = []
    for fam in ("Classical", "Neural"):
        label = "Classical machine learning" if fam == "Classical" else "Neural networks"
        rows.append(f'<tr class="group"><td colspan="8">{label}</td></tr>')
        for _, r in df[df["family"] == fam].iterrows():
            cells = "".join(
                f'<td class="{"best" if abs(r[c] - best[c]) < 1e-12 else ""}">{r[c]:.4f}</td>'
                for c in METRIC_COLS
            )
            rows.append(f"<tr><td>{r['model']}</td>{cells}<td>{r['train_seconds']:.1f}</td></tr>")
    return f"<table class='results'><thead>{head}</thead><tbody>{''.join(rows)}</tbody></table>"


def confusion_table(df):
    head = "<tr><th>Model</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th></tr>"
    rows = "".join(
        f"<tr><td>{r['model']}</td><td>{r['tn']:,}</td><td>{r['fp']:,}</td>"
        f"<td>{r['fn']:,}</td><td>{r['tp']:,}</td></tr>"
        for _, r in df.iterrows()
    )
    return f"<table class='results compact'><thead>{head}</thead><tbody>{rows}</tbody></table>"


def fill(template, df):
    by_slug = df.set_index("slug")

    def value(match):
        slug, col, fmt = match.group(1), match.group(2), match.group(3)
        v = by_slug.loc[slug, col]
        if fmt == "pct":
            return f"{100 * v:.1f}%"
        if fmt == "int":
            return f"{int(v):,}"
        if fmt == "1":
            return f"{v:.1f}"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    html = template.replace("{{RESULTS_TABLE}}", results_table(df))
    html = html.replace("{{CONFUSION_TABLE}}", confusion_table(df))
    html = re.sub(r"\{\{(\w+)\.(\w+)(?::(\w+))?\}\}", value, html)
    leftover = re.findall(r"\{\{[^}]+\}\}", html)
    if leftover:
        sys.exit(f"Unfilled placeholders: {leftover}")
    return html


def find_browser():
    for b in BROWSERS:
        if Path(b).exists() or shutil.which(b):
            return b
    sys.exit("Microsoft Edge or Google Chrome is needed to print the PDF.")


def main():
    df = load_metrics()
    HTML_OUT.write_text(fill(TEMPLATE.read_text(encoding="utf-8"), df), encoding="utf-8")
    subprocess.run(
        [find_browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
         f"--print-to-pdf={PDF_OUT}", HTML_OUT.as_uri()],
        check=True, capture_output=True, timeout=180,
    )
    print(f"Wrote {HTML_OUT.relative_to(ROOT)} and {PDF_OUT.name}")


if __name__ == "__main__":
    main()
