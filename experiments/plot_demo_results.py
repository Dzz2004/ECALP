"""Create presentation figures from full observed predictions and actual KNN cases."""
import argparse
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLIP, ECALP, RED, MUTED = "#7888a7", "#087e8b", "#c65155", "#64748b"


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=.7)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results/demo")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/demo/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets, all_rows, metadata, cases = ["DTD", "eurosat"], {}, {}, {}
    for d in datasets:
        with (args.data_dir / f"{d}_predictions.csv").open(newline="") as stream:
            all_rows[d] = [{k: (v if k in ["dataset", "outcome"] else int(v))
                            for k, v in row.items()} for row in csv.DictReader(stream)]
        metadata[d] = json.loads((args.data_dir / f"{d}_metadata.json").read_text())
        cases[d] = json.loads((args.data_dir / f"{d}_cases.json").read_text())
    with (args.data_dir / "summary.csv").open("w", newline="") as stream:
        columns = ["dataset", "sample_count", "clip_accuracy", "ecalp_accuracy", "gain_pp",
                   "corrected", "regressed", "net_correct", "both_correct", "both_wrong"]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for d in datasets:
            m = metadata[d]
            writer.writerow(dict(dataset=d, sample_count=m["sample_count"],
                                 clip_accuracy=m["accuracy"]["clip"], ecalp_accuracy=m["accuracy"]["ecalp"],
                                 gain_pp=m["accuracy"]["ecalp"]-m["accuracy"]["clip"],
                                 net_correct=m["correct_ecalp"]-m["correct_clip"], **m["outcomes"]))
    display = lambda d: "EuroSAT" if d == "eurosat" else d
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 14, "axes.labelsize": 11,
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "pdf.fonttype": 42})

    # 1. Overall accuracy and the exact source of the net gain.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for j, (method, color) in enumerate([("clip", CLIP), ("ecalp", ECALP)]):
        values = [metadata[d]["accuracy"][method] for d in datasets]
        bars = ax.bar(np.arange(2) + (j-.5)*.34, values, width=.34,
                      color=color, label="CLIP (same features)" if j == 0 else "ECALP")
        ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11)
    ax.set_xticks(range(2), [display(d) for d in datasets])
    ax.set_ylim(0, 70)
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Frozen RN50, zero-shot")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax)
    ax = axes[1]
    for j, (kind, color, label) in enumerate([("corrected", ECALP, "Wrong → right"),
                                            ("regressed", RED, "Right → wrong")]):
        values = [100*metadata[d]["outcomes"][kind]/metadata[d]["sample_count"] for d in datasets]
        bars = ax.bar(np.arange(2) + (j-.5)*.34, values, width=.34, color=color, label=label)
        labels = [f"{metadata[d]['outcomes'][kind]} images\n({v:.2f}%)" for d, v in zip(datasets, values)]
        ax.bar_label(bars, labels=labels, padding=4, fontsize=10)
    ax.set_xticks(range(2), [display(d) for d in datasets])
    ax.set_ylim(0, max(100*metadata[d]["outcomes"]["corrected"]/metadata[d]["sample_count"] for d in datasets)+9)
    ax.set_ylabel("Share of the full test set (%)")
    ax.set_title("Corrections and regressions")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    style(ax)
    fig.suptitle("Where does the accuracy gain come from?", fontsize=18, y=1.04)
    fig.tight_layout(w_pad=3)
    save(fig, args.output_dir, "accuracy_and_corrections")

    # 2. A full paired 2x2 accounting; failures have equal visibility.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, d in zip(axes, datasets):
        counts = metadata[d]["outcomes"]
        cells = [[("both_correct", "Both correct"), ("regressed", "Regression")],
                 [("corrected", "Correction"), ("both_wrong", "Both wrong")]]
        for row in range(2):
            for col in range(2):
                key, label = cells[row][col]
                color = {"corrected": "#ccebe8", "regressed": "#f6d4d5",
                         "both_correct": "#e2e8f0", "both_wrong": "#f1f3f6"}[key]
                ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor="white", linewidth=4))
                count = counts[key]
                ax.text(col+.5, row+.38, f"{count:,}", ha="center", va="center", fontsize=25, weight="bold")
                ax.text(col+.5, row+.68, f"{label} · {100*count/metadata[d]['sample_count']:.2f}%",
                        ha="center", va="center", fontsize=10)
        ax.set(xlim=(0, 2), ylim=(2, 0))
        ax.set_xticks([.5, 1.5], ["ECALP correct", "ECALP wrong"])
        ax.set_yticks([.5, 1.5], ["CLIP correct", "CLIP wrong"])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        net = counts["corrected"] - counts["regressed"]
        ax.set_title(f"{display(d)} · net +{net} correct images", pad=14)
    fig.tight_layout(w_pad=4)
    save(fig, args.output_dir, "paired_outcomes")

    # 3. Cumulative performance at each arrival, with the original stream order.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, d in zip(axes, datasets):
        rows = all_rows[d]
        indices = np.unique(np.r_[np.arange(99, len(rows), 20), len(rows)-1])
        for method, color, label in [("clip", CLIP, "CLIP"), ("ecalp", ECALP, "ECALP")]:
            cumulative = 100*np.cumsum([r[f"{method}_correct"] for r in rows])/np.arange(1, len(rows)+1)
            ax.plot(indices+1, cumulative[indices], color=color, linewidth=2, label=label)
        ax.set_title(display(d))
        ax.set_xlabel("Test images observed (original file order)")
        ax.set_ylabel("Cumulative Top-1 accuracy (%)")
        ax.set_ylim(0, 100)
        ax.legend(frameon=False)
        style(ax)
    fig.suptitle("Sequential inference: how the observed accuracy evolves", fontsize=16, y=1.03)
    fig.tight_layout(w_pad=3)
    save(fig, args.output_dir, "stream_accuracy")

    # 4. Two corrections and two regressions selected by a published fixed rule.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for row, d in enumerate(datasets):
        for col, category in enumerate(["corrected", "regressed"]):
            ax = axes[row, col]
            case = next(c for c in cases[d] if c["outcome"] == category)
            color = ECALP if category == "corrected" else RED
            ax.axis("off")
            ax.set_title(f"{display(d)} · sample #{case['sample_index']} · "
                         f"{'Correction' if category == 'corrected' else 'Regression'}",
                         loc="left", color=color, weight="bold", pad=12)
            ax.text(.01, .85, f"Ground truth   C{case['ground_truth']:02d}", fontsize=15, weight="bold")
            ax.text(.01, .65, f"CLIP   C{case['clip_pred']:02d}   →   ECALP   C{case['ecalp_pred']:02d}", fontsize=17, color=color)
            ax.text(.01, .44, "CLIP top-3 cosine similarities", color=MUTED, fontsize=11)
            ax.text(.01, .28, "   |   ".join(f"C{x['class_id']:02d}: {x['cosine']:.3f}" for x in case["clip_top3"]), fontsize=12)
            ax.text(.01, .07, "Past KNN pseudo-labels: " + ", ".join(
                f"C{x['previous_pseudo_label']:02d}" for x in case["historical_neighbors"]), fontsize=10, color=MUTED)
    fig.suptitle("Actual cases: successes and failures", fontsize=20, y=1.01)
    fig.text(.5, .015, "Selection: first case in each category after 100 arrivals. IDs identify feature rows/classes; raw images are unavailable.",
             ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, .97), h_pad=2, w_pad=3)
    save(fig, args.output_dir, "case_comparison")

    # 5. Real outgoing KNN connections at one captured DTD correction.
    case = next(c for c in cases["DTD"] if c["outcome"] == "corrected")
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set(xlim=(-.2, 1.2), ylim=(-.1, 1.1))
    ax.axis("off")
    current = (.5, .5)
    texts, history = case["text_neighbors"], case["historical_neighbors"]
    for side, nodes in [("text", texts), ("history", history)]:
        ys = np.linspace(.9, .1, len(nodes))
        x = .03 if side == "text" else .97
        for y, node in zip(ys, nodes):
            color = CLIP if side == "text" else ECALP
            ax.plot([current[0], x], [current[1], y], color=color, linewidth=1.4, alpha=.6, zorder=1)
            ax.scatter([x], [y], s=1800 if side == "text" else 650, marker="s" if side == "text" else "o",
                       color="#e5eaf3" if side == "text" else "#d6efeb", edgecolors=color, zorder=2)
            if side == "text":
                text = f"C{node['class_id']:02d}"
                ax.text(x-.1, y, f"cos {node['cosine']:.3f}", va="center", ha="right", fontsize=9, color=MUTED)
            else:
                text = f"#{node['sample_index']}"
                ax.text(x+.055, y, f"prior pseudo-label\nC{node['previous_pseudo_label']:02d}", va="center", fontsize=10)
            ax.text(x, y, text, ha="center", va="center", fontsize=10, weight="bold", zorder=3)
    ax.scatter([.5], [.5], s=5200, color="#ccebe8", edgecolors=ECALP, zorder=4)
    ax.text(.5, .5, f"Current\nsample\n#{case['sample_index']}", ha="center", va="center", fontsize=12, weight="bold", zorder=5)
    ax.text(.03, 1.03, "Text prototypes", ha="center", weight="bold", color=CLIP)
    ax.text(.97, 1.03, "Historical images", ha="center", weight="bold", color=ECALP)
    ax.set_title(f"DTD · actual local KNN context\nCLIP C{case['clip_pred']:02d} → ECALP C{case['ecalp_pred']:02d} · ground truth C{case['ground_truth']:02d}", fontsize=16, pad=25)
    fig.text(.5, .03, "Only the query-selected KNN edges are shown; layout is schematic. These observations do not establish which edge caused the correction.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, .05, 1, 1))
    save(fig, args.output_dir, "local_graph_case")
    print(f"Saved five PNG/PDF figure pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
