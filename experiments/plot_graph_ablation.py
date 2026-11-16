"""Generate the two presentation figures required by task B."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np


FULL_COLOR = "#8495B3"
ABLATION_COLOR = "#2F8292"
TEXT_COLOR = "#4C72B0"
FEWSHOT_COLOR = "#55A868"
IMAGE_COLOR = "#DD8452"
CURRENT_COLOR = "#C44E52"
EDGE_TEXT = "#4BAFC8"
EDGE_IMAGE = "#E48B32"


def load_results(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    values = {(row["dataset"].lower(), row["variant"]): float(row["accuracy"]) for row in rows}
    return {
        "DTD": [values[("dtd", "full_ecalp")], values[("dtd", "no_image_image_edges")]],
        "EuroSAT": [values[("eurosat", "full_ecalp")], values[("eurosat", "no_image_image_edges")]],
    }


def plot_accuracy(results, output_path):
    labels = list(results)
    full = [results[name][0] for name in labels]
    ablated = [results[name][1] for name in labels]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 7.5))
    bars_full = ax.bar(x - width / 2, full, width, label="Full ECALP", color=FULL_COLOR)
    bars_ablated = ax.bar(
        x + width / 2,
        ablated,
        width,
        label="Without image-image edges",
        color=ABLATION_COLOR,
    )
    ax.set_title("Image-image edge ablation (RN50 zero-shot)", fontsize=20, pad=12)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=15)
    ax.set_xticks(x, labels, fontsize=14)
    ax.set_ylim(0, 62)
    ax.tick_params(axis="y", labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=13, frameon=True)
    ax.bar_label(bars_full, fmt="%.2f", padding=5, fontsize=13)
    ax.bar_label(bars_ablated, fmt="%.2f", padding=5, fontsize=13)
    for index, (base, reduced) in enumerate(zip(full, ablated)):
        ax.text(
            index,
            max(base, reduced) + 4.0,
            f"-{base - reduced:.2f} pp",
            ha="center",
            va="bottom",
            fontsize=12,
            color="#333333",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_node(ax, center, text, color, width=1.55, height=0.62, linestyle="-"):
    x, y = center
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        facecolor=color,
        edgecolor=color,
        linewidth=1.8,
        linestyle=linestyle,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=11, weight="bold", zorder=4)


def connect(ax, start, end, color, width=2.0, linestyle="-", alpha=0.9):
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=width,
        linestyle=linestyle,
        alpha=alpha,
        zorder=1,
    )


def connect_path(ax, points, color, width=2.0, linestyle="-", alpha=0.9):
    ax.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        color=color,
        linewidth=width,
        linestyle=linestyle,
        alpha=alpha,
        zorder=1,
    )


def plot_graph(output_path):
    fig, ax = plt.subplots(figsize=(13, 7.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.0)
    ax.axis("off")
    ax.set_title("ECALP dynamic graph", fontsize=22, pad=18)
    ax.text(1.8, 5.35, "Class text", ha="center", fontsize=15, weight="bold", color=TEXT_COLOR)
    ax.text(5.4, 5.35, "Past test images", ha="center", fontsize=15, weight="bold", color=IMAGE_COLOR)
    ax.text(8.7, 4.45, "New test image", ha="center", fontsize=14, weight="bold", color=CURRENT_COLOR)
    ax.text(1.8, 2.15, "Few-shot samples (optional)", ha="center", fontsize=14, weight="bold", color=FEWSHOT_COLOR)

    text1, text2 = (1.8, 4.45), (1.8, 3.35)
    history1, history2, history3 = (5.4, 4.45), (5.4, 3.0), (5.4, 1.8)
    current, support = (8.7, 3.35), (1.8, 1.35)
    add_node(ax, text1, "Class 1", TEXT_COLOR, width=1.65, height=0.58)
    add_node(ax, text2, "Class 2", TEXT_COLOR, width=1.65, height=0.58)
    add_node(ax, history1, "Past test 1", IMAGE_COLOR, width=1.65, height=0.58)
    add_node(ax, history2, "Past test 2", IMAGE_COLOR, width=1.65, height=0.58)
    add_node(ax, history3, "Past test 3", IMAGE_COLOR, width=1.65, height=0.58)
    add_node(ax, current, "New test", CURRENT_COLOR, width=1.65, height=0.68)
    add_node(ax, support, "Few-shot sample", FEWSHOT_COLOR, width=1.8, height=0.56, linestyle="--")

    connect(ax, text1, current, EDGE_TEXT, width=2.0)
    connect(ax, text2, current, EDGE_TEXT, width=2.0)
    connect(ax, history1, history2, EDGE_IMAGE, width=3.2)
    connect(ax, history2, history3, EDGE_IMAGE, width=3.2)
    connect(ax, history1, current, EDGE_IMAGE, width=3.2)
    connect(ax, history2, current, EDGE_IMAGE, width=3.2)
    connect(ax, support, current, FEWSHOT_COLOR, width=1.8, linestyle="--", alpha=0.76)
    ax.text(1.8, 0.88, "unused in zero-shot", ha="center", fontsize=10, color="#666666")

    inset = FancyBboxPatch((10.25, 2.15), 1.9, 3.05, boxstyle="round,pad=0.05,rounding_size=0.12",
                           facecolor="#FAFBFD", edgecolor="#D9DEE8", linewidth=1.4, zorder=0)
    ax.add_patch(inset)
    ax.text(11.2, 4.82, "B ablation", ha="center", fontsize=13.5, weight="bold", color="#222222")
    ax.text(11.2, 4.34, "Remove image-image\nedges only", ha="center", fontsize=10.5, color=CURRENT_COLOR, weight="bold")
    ax.plot([10.7, 11.7], [3.65, 3.65], color=EDGE_IMAGE, linewidth=4.0, zorder=2)
    ax.text(11.2, 3.65, "×", ha="center", va="center", fontsize=30, weight="bold", color=CURRENT_COLOR, zorder=4)
    ax.text(11.2, 2.72, "--disable_image_\nimage_edges", ha="center", fontsize=9.6,
            family="monospace", color=CURRENT_COLOR, weight="bold")

    legend_handles = [
        Line2D([0], [0], color=EDGE_TEXT, lw=2, label="Image-text edge"),
        Line2D([0], [0], color=EDGE_IMAGE, lw=4, label="Image-image edge (ablated)"),
        Line2D([0], [0], color=FEWSHOT_COLOR, lw=2, ls="--", label="Few-shot edge (optional)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.11), ncol=3, fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot task B graph-ablation figures")
    parser.add_argument("--csv", default="results/graph_ablation.csv")
    parser.add_argument("--output_dir", default="presentation/B/figures")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = (repo_root / args.csv).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(load_results(csv_path), output_dir / "accuracy_comparison.png")
    plot_graph(output_dir / "graph_structure.png")
    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
