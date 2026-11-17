"""Show verified original photos beside outputs from actual-photo inference."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def main():
    matches = json.loads((ROOT / "results/demo/image_matches.json").read_text())
    fig, axes = plt.subplots(len(matches), 4, figsize=(12.8, 3.4*len(matches)+.6),
                             gridspec_kw={"width_ratios": [1.6, 1, 1, 1]}, squeeze=False)
    plt.rcParams["pdf.fonttype"] = 42
    outputs = []
    for row, match in enumerate(matches):
        index = match["sample_index"]
        output = json.loads((ROOT / f"results/demo/live_outputs/DTD_{index}_history_{index}.json").read_text())
        if not output["matches_recorded_case"]:
            raise ValueError("Only actual-photo predictions matching the recorded case can be presented")
        path = ROOT / match["image"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == match["image_sha256"] == output["image_sha256"]
        outputs.append(output)
        for ax in axes[row]:
            ax.axis("off")
        axes[row, 0].imshow(Image.open(path).convert("RGB"))
        axes[row, 0].set_title(f"Input photo · sample #{index}", fontsize=12, pad=9)
        for col, method, header in [(1, "clip", "CLIP output"), (2, "ecalp", "ECALP output")]:
            ax = axes[row, col]
            correct = output[f"{method}_pred"] == output["ground_truth"]
            color = "#087e8b" if correct else "#c65155"
            ax.text(.5, .78, header, ha="center", fontsize=13, color="#64748b")
            ax.text(.5, .51, output[f"{method}_name"], ha="center", fontsize=23, weight="bold", color=color)
            ax.text(.5, .36, f"C{output[f'{method}_pred']:02d}", ha="center", fontsize=13, color=color)
            ax.text(.5, .19, "CORRECT" if correct else "WRONG", ha="center", fontsize=12, color=color)
        ax = axes[row, 3]
        ax.text(.5, .78, "Ground truth", ha="center", fontsize=13, color="#64748b")
        ax.text(.5, .51, output["ground_truth_name"], ha="center", fontsize=23, weight="bold")
        ax.text(.5, .34, f"C{output['ground_truth']:02d}", ha="center", fontsize=13)
        ax.text(.5, .10, f"ECALP context:\n{index} preceding images", ha="center", fontsize=10, color="#64748b")
    fig.suptitle("Real input images → CLIP and ECALP outputs", fontsize=20, y=.99)
    fig.text(.5, .015, "Same RN50 center crop for both methods. Photos match stored features (cosine ≥ 0.9999); all displayed outputs were recomputed from these photos.",
             ha="center", fontsize=9, color="#64748b")
    fig.tight_layout(rect=(0, .045, 1, .96), w_pad=2, h_pad=2)
    out = ROOT / "results/demo/figures"
    fig.savefig(out / "real_image_input_output.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "real_image_input_output.pdf", bbox_inches="tight")
    print(out / "real_image_input_output.png")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
