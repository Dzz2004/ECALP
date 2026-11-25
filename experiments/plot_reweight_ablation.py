"""Plot the context reweighting mechanism and measured accuracy."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def plot_mechanism():
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (0.02, 0.59, "Class text\nprototypes"),
        (0.26, 0.59, "Variance in each\nfeature dimension"),
        (0.51, 0.59, "Weight current\nimage features"),
        (0.76, 0.59, "Cosine with\nhistorical image"),
        (0.51, 0.13, "Raw image features\n(no reweighting)"),
    ]
    for x, y, label in boxes:
        ax.text(x + 0.105, y + 0.115, label, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#e8f0f8", edgecolor="#456a8a"))
    for start, end in ((0.23, 0.26), (0.47, 0.51), (0.72, 0.76)):
        ax.annotate("", xy=(end, 0.705), xytext=(start, 0.705),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.78, 0.58), xytext=(0.71, 0.36),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.87, 0.42, "Image–image edge", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "reweight_mechanism.png", dpi=180)
    plt.close(fig)


def plot_accuracy():
    with (RESULTS / "reweight_ablation.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    latest_run = rows[-1]["run_id"]
    measured = {(row["dataset"], row["variant"]): float(row["accuracy"])
                for row in rows if row["run_id"] == latest_run and row["source"] == "ours"}
    datasets = ("DTD", "eurosat")
    if len(measured) != 4:
        raise ValueError("Both variants for DTD and eurosat are required")
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = range(len(datasets))
    for offset, variant, label, color in ((-0.18, "raw_cosine", "Raw cosine", "#91abc2"),
                                          (0.18, "context_reweight", "Context reweighting", "#315f85")):
        bars = ax.bar([i + offset for i in positions],
                      [measured[(dataset, variant)] for dataset in datasets],
                      width=0.34, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set_xticks(list(positions), ("DTD", "EuroSAT"))
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_ylim(0, max(measured.values()) + 10)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "reweight_ablation.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_mechanism()
    plot_accuracy()
