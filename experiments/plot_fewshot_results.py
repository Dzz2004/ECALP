"""Render the DTD shot comparison and the ECALP propagation flow."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS = Path(__file__).resolve().parents[1] / "results"


def plot_accuracy():
    with (RESULTS / "fewshot_results.csv").open(newline="", encoding="utf-8") as source:
        rows = sorted(csv.DictReader(source), key=lambda row: int(row["num_shots"]))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar([row["num_shots"] for row in rows],
                  [float(row["accuracy"]) for row in rows],
                  color=["#738fa7", "#45819a", "#17536f"], width=0.58)
    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set(xlabel="Labeled examples per class", ylabel="Top-1 accuracy (%)",
           title="DTD · RN50 · ECALP", ylim=(0, 100))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fewshot_accuracy.png", dpi=180)
    plt.close(fig)


def plot_flow():
    fig, ax = plt.subplots(figsize=(11, 4.1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    steps = [
        (0.25, 0.73, "Text prototypes and support images\nstart with fixed class labels"),
        (0.75, 0.73, "Append the current test image\nand expand the KNN graph"),
        (0.75, 0.29, "Propagate scores for T steps\nand restore fixed labels each step"),
        (0.25, 0.29, "Predict the current image\nand keep attenuated top-class scores"),
    ]
    for x, y, label in steps:
        ax.text(x, y, label, ha="center", va="center", fontsize=13,
                bbox=dict(boxstyle="round,pad=0.65", fc="#e9f2f7", ec="#416983"))
    for start, end in (((0.43, 0.73), (0.57, 0.73)), ((0.75, 0.61), (0.75, 0.41)),
                       ((0.56, 0.29), (0.44, 0.29))):
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.61, 0.62), xytext=(0.37, 0.41),
                arrowprops=dict(arrowstyle="->", lw=1.4, linestyle="--"))
    ax.text(0.49, 0.55, "Next image", ha="center", va="bottom", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "label_propagation_flow.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    plot_accuracy()
    plot_flow()
