import textwrap
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")   

BASE     = Path("/Users/hapikid/Documents/QC4SG")
CSV_PATH = BASE / "benchmark_results.csv"
OUT_PNG  = BASE / "benchmark_classification.png"


CLF_ROWS = {
    "SVM-RBF (clf, pond-holdout)":            "SVM-RBF",
    "ClassicalReservoir (pond-holdout) (clf)": "Cls. Reservoir",
    "QRC (pond-holdout) (clf)":                "QRC",
}



matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10.5,
    "xtick.labelsize":   10,
    "ytick.labelsize":   9.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#dddddd",
    "grid.linewidth":    0.4,
    "axes.axisbelow":    True,
})

COLOR_AUC    = "#3A7EC6"   # steel blue
COLOR_ACC    = "#E87D2A"   # burnt orange
EDGE_DEFAULT = dict(linewidth=0.7,  edgecolor="#555555")
EDGE_QRC     = dict(linewidth=2.4,  edgecolor="#000000")   # bold so QRC stands out


def load_clf_rows():
    df  = pd.read_csv(CSV_PATH)
    clf = (df[df["model"].isin(CLF_ROWS)]
             .copy()
             .assign(label=lambda d: d["model"].map(CLF_ROWS))
             .set_index("label")
             .loc[list(CLF_ROWS.values())])    # enforce display order
    return clf


def print_markdown_table(clf):
    def _f(v): return "—" if pd.isna(v) else f"{v:.4f}"

    print("\n## Pond-holdout benchmark — classification\n")
    print("| Model | AUC-ROC | Accuracy | F1-weighted | F1-class0 | F1-class1 |")
    print("|---|---|---|---|---|---|")
    for label, row in clf.iterrows():
        print(f"| {label} | {_f(row.auc)} | {_f(row.accuracy)} | "
              f"{_f(row.f1_weighted)} | {_f(row.f1_class0)} | {_f(row.f1_class1)} |")
    print()


def _bar_label(ax, x, y, text, bold=False):
    ax.text(x, y + 0.012, text,
            ha="center", va="bottom", fontsize=8.5,
            fontweight="bold" if bold else "normal",
            color="#111111")



def make_figure(clf):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    labels = clf.index.tolist()          # ["SVM-RBF", "Cls. Reservoir", "QRC"]
    x      = np.arange(len(labels))
    bw     = 0.30    # width of one bar
    gap    = 0.05    # gap between the two bars in each group

    for i, model in enumerate(labels):
        is_qrc  = (model == "QRC")
        ekw     = EDGE_QRC if is_qrc else EDGE_DEFAULT
        auc_val = clf.loc[model, "auc"]
        acc_val = clf.loc[model, "accuracy"]

        # AUC bar (left of group)
        ax.bar(x[i] - bw/2 - gap/2, auc_val, width=bw, color=COLOR_AUC, **ekw)
        _bar_label(ax, x[i] - bw/2 - gap/2, auc_val, f"{auc_val:.2f}", bold=is_qrc)

        # Accuracy bar (right of group)
        ax.bar(x[i] + bw/2 + gap/2, acc_val, width=bw, color=COLOR_ACC, **ekw)
        _bar_label(ax, x[i] + bw/2 + gap/2, acc_val, f"{acc_val:.2f}", bold=is_qrc)

    # Axis formatting — start y below the lowest value so gaps are visible
    all_vals = list(clf["auc"]) + list(clf["accuracy"])
    y_min    = max(0.0, min(all_vals) - 0.10)   # ~0.38 given data
    ax.set_ylim(y_min, 1.07)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=10.5)

    ax.set_title("QuteShrimp — quantum vs classical, unseen ponds", pad=10)

    ax.legend(
        handles=[
            mpatches.Patch(color=COLOR_AUC, label="AUC-ROC"),
            mpatches.Patch(color=COLOR_ACC, label="Accuracy"),
        ],
        loc="upper left", fontsize=9.5, framealpha=0.9, edgecolor="#cccccc"
    )

    # Footnote below the axes — wrapped to fit the figure width
    footnote = (
        "RandomForest omitted (AUC=1.0 reconstructs rule-derived labels). "
        "QRC = simulated analog quantum reservoir; quantum dynamics improve "
        "at-risk classification over the matched classical reservoir (AUC 0.83 vs 0.68)."
    )
    fig.text(
        0.5, -0.04,
        textwrap.fill(footnote, width=100),
        ha="center", va="top", fontsize=8.5, style="italic", color="#444444",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                  edgecolor="#cccccc", linewidth=0.6),
    )

    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    clf = load_clf_rows()
    print_markdown_table(clf)
    make_figure(clf)
