"""Build Final_Project_Report.pdf from report/report_template.tex.

Every number in the report comes from results/metrics.csv, so the report always
matches the latest run of run_all.py. Placeholders in the template:

    <<xgboost.roc_auc>>      a metric for one model (4 decimals)
    <<xgboost.recall:pct>>   the same metric as a percentage (1 decimal)
    <<xgboost.fp:int>>       an integer with thousands separators
    <<xgboost.train_seconds:1>>  one decimal place
    <<RESULTS_TABLE>>        Table I, all metrics for every model
    <<CONFUSION_TABLE>>      Table II, confusion-matrix counts for every model

The filled-in LaTeX is written to report/report.tex and compiled with Tectonic
(https://tectonic-typesetting.github.io), a self-contained LaTeX engine.

Usage:
    python report/build_report.py
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent
TEMPLATE = REPORT_DIR / "report_template.tex"
TEX_OUT = REPORT_DIR / "report.tex"
PDF_OUT = ROOT / "Final_Project_Report.pdf"

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
    lines = [
        r"\begin{table*}[!t]", r"\centering",
        r"\caption{Performance Comparison of All Models on the Shared Test Set "
        r"(85,443 transactions, 148 fraud). Best value per metric in bold.}",
        r"\label{tab:results}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
        r"Model & Accuracy & Precision & Recall & F1-score & ROC AUC & PR AUC & Train time (s) \\",
        r"\midrule",
    ]
    for fam, label in (("Classical", "Classical machine learning"), ("Neural", "Neural networks")):
        lines.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{label}}}}} \\")
        for _, r in df[df["family"] == fam].iterrows():
            cells = [
                rf"\textbf{{{r[c]:.4f}}}" if abs(r[c] - best[c]) < 1e-12 else f"{r[c]:.4f}"
                for c in METRIC_COLS
            ]
            lines.append(rf"\quad {r['model']} & {' & '.join(cells)} & {r['train_seconds']:.1f} \\")
        if fam == "Classical":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def confusion_table(df):
    lines = [
        r"\begin{table}[!t]", r"\centering",
        r"\caption{Confusion-Matrix Counts on the Test Set (85,295 legitimate, 148 fraud)}",
        r"\label{tab:cm}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
        r"Model & TN & FP & FN & TP \\", r"\midrule",
    ]
    for _, r in df.iterrows():
        lines.append(rf"{r['model']} & {r['tn']:,} & {r['fp']:,} & {r['fn']:,} & {r['tp']:,} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def fill(template, df):
    by_slug = df.set_index("slug")

    def value(match):
        slug, col, fmt = match.group(1), match.group(2), match.group(3)
        v = by_slug.loc[slug, col]
        if fmt == "pct":
            return f"{100 * v:.1f}\\%"
        if fmt == "int":
            return f"{int(v):,}"
        if fmt == "1":
            return f"{v:.1f}"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    tex = template.replace("<<RESULTS_TABLE>>", results_table(df))
    tex = tex.replace("<<CONFUSION_TABLE>>", confusion_table(df))
    tex = re.sub(r"<<(\w+)\.(\w+)(?::(\w+))?>>", value, tex)
    leftover = re.findall(r"<<[^>]+>>", tex)
    if leftover:
        sys.exit(f"Unfilled placeholders: {leftover}")
    return tex


def find_tectonic():
    local = Path(os.environ.get("LOCALAPPDATA", "")) / "tectonic" / "tectonic.exe"
    found = shutil.which("tectonic") or (str(local) if local.exists() else None)
    if not found:
        sys.exit("Tectonic is needed to build the PDF: https://tectonic-typesetting.github.io")
    return found


def main():
    df = load_metrics()
    TEX_OUT.write_text(fill(TEMPLATE.read_text(encoding="utf-8"), df), encoding="utf-8")
    result = subprocess.run(
        [find_tectonic(), "--keep-logs", "--outdir", str(REPORT_DIR), TEX_OUT.name],
        cwd=REPORT_DIR, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        sys.exit(result.stdout + result.stderr)
    shutil.move(REPORT_DIR / "report.pdf", PDF_OUT)
    print(f"Wrote {TEX_OUT.relative_to(ROOT)} and {PDF_OUT.name}")


if __name__ == "__main__":
    main()
