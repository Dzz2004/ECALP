"""Plot the latest measured CLIP/ECALP pairs, excluding synthetic smoke tests."""
import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/main_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/main_results.png"))
    args = parser.parse_args()
    with args.results.open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream)
                if r["source"] == "ours" and r["clip_model"] == "RN"
                and r["mode"] == "ZS" and r["variant"] in ("clip", "ecalp")]
    # Require paired methods with the same feature archive and experiment run.
    groups = {}
    for row in rows:
        key = (row["dataset"], row["run_id"], row["feature_sha256"])
        groups.setdefault(key, {})[row["variant"]] = row
    pairs = {}
    for (dataset, run_id, _), values in sorted(groups.items()):
        if set(values) == {"clip", "ecalp"}:
            pairs[dataset] = values
    if not pairs:
        parser.error("No completed same-run CLIP/ECALP pair with source=ours")
    datasets = list(pairs)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for offset, variant, label, color in [(-0.18, "clip", "CLIP (same features)", "#7b8caa"),
                                           (0.18, "ecalp", "ECALP", "#287c8e")]:
        values = [float(pairs[d][variant]["accuracy"]) for d in datasets]
        bars = ax.bar([i + offset for i in range(len(datasets))], values,
                      width=0.36, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_xticks(range(len(datasets)), ["EuroSAT" if d == "eurosat" else d for d in datasets])
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("RN50 zero-shot reproduction")
    ax.set_ylim(0, min(100, max(float(r["accuracy"]) for p in pairs.values() for r in p.values()) + 12))
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    print(args.output)


if __name__ == "__main__":
    main()
