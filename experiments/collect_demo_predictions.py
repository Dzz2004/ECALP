"""Observe official ECALP inference to export actual sample-level comparisons."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from run_main_results import ROOT, git, load_features, sha256

OUTCOMES = ("corrected", "regressed", "both_correct", "both_wrong")


def outcome(clip, ecalp, truth):
    if clip != truth and ecalp == truth:
        return "corrected"
    if clip == truth and ecalp != truth:
        return "regressed"
    return "both_correct" if clip == truth else "both_wrong"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["DTD", "eurosat"], required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda:PHYSICAL_INDEX")
    parser.add_argument("--features-path", type=Path, default=ROOT / "features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/demo")
    parser.add_argument("--reference", type=Path, default=ROOT / "results/main_results.csv")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device.startswith("cuda:") and args.device[5:].isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device[5:]
    else:
        parser.error("--device must be cpu or cuda:PHYSICAL_INDEX")
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    torch.set_num_threads(4)
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU access may require execution outside the sandbox")
    images, texts, labels, feature_info = load_features(
        args.features_path / "fine_grained/RN" / f"{args.dataset}.pt")
    with args.reference.open(newline="") as stream:
        reference_rows = [r for r in csv.DictReader(stream)
                          if r["dataset"] == args.dataset and r["mode"] == "ZS"
                          and r["clip_model"] == "RN" and r["source"] == "ours"
                          and r["feature_sha256"] == feature_info["sha256"]]
    by_run = {}
    for r in reference_rows:
        by_run.setdefault(r["run_id"], {})[r["variant"]] = r
    complete = [k for k in sorted(by_run) if {"clip", "ecalp"} <= by_run[k].keys()]
    if not complete:
        raise ValueError("No completed reference run for the same dataset and feature hash")
    reference = by_run[complete[-1]]
    reference_meta = json.loads((args.reference.parent / "logs" / complete[-1] / "metadata.json").read_text())
    for name in ("main.py", "utils/iterative_label_propagation.py"):
        if sha256(ROOT / name) != reference_meta["source_sha256"][name]:
            raise ValueError(f"Algorithm source differs from reference run: {name}")
    image_vectors = images.mean(1) if images.ndim == 3 else images
    similarities = F.normalize(image_vectors.float(), dim=1) @ F.normalize(texts.float(), dim=1).T
    clip_predictions = similarities.argmax(1)
    top_scores, top_classes = similarities.topk(min(3, len(texts)), dim=1)
    sys.path.insert(0, str(ROOT))
    upstream = importlib.import_module("main")
    lp = importlib.import_module("utils.iterative_label_propagation")
    original = upstream.iterative_label_propagation
    rows, cases = [], []
    selected = Counter()

    def observe(*call_args, **call_kwargs):
        index = len(rows)
        previous = getattr(lp, "Y_0", None)
        # The official function receives exactly its original arguments. Ground truth
        # is accessed only AFTER inference for evaluation and case selection below.
        prediction = original(*call_args, **call_kwargs)
        predicted = int(prediction.argmax(1).item())
        truth, clip = int(labels[index]), int(clip_predictions[index])
        category = outcome(clip, predicted, truth)
        row = dict(dataset=args.dataset, sample_index=index, ground_truth=truth,
                   clip_pred=clip, ecalp_pred=predicted, outcome=category,
                   clip_correct=int(clip == truth), ecalp_correct=int(predicted == truth))
        rows.append(row)
        # Reproducible selection, with both successes and failures. At least 100
        # samples have arrived; pick the first three in each outcome category.
        if index >= 99 and selected[category] < 3:
            selected[category] += 1
            neighbors = []
            for neighbor, score in zip(lp.knn_im2im[-1].tolist(), lp.sim_im2im[-1].tolist()):
                if neighbor >= index or any(n["sample_index"] == neighbor for n in neighbors):
                    continue  # Current sample, its duplicate, or repeated neighbor.
                pseudo = int(previous[neighbor].to_dense().argmax().item())
                neighbors.append(dict(sample_index=neighbor, previous_pseudo_label=pseudo,
                                      reweighted_similarity=score))
            cases.append({**row, "clip_top3": [dict(class_id=int(c), cosine=float(s))
                          for c, s in zip(top_classes[index], top_scores[index])],
                          "historical_neighbors": neighbors,
                          "text_neighbors": [dict(class_id=int(c), cosine=float(s))
                          for c, s in zip(lp.knn_im2text[-1], lp.sim_im2text[-1])],
                          "selection": "first three per outcome after 100 samples have arrived"})
        if len(rows) % 1000 == 0:
            print(f"{args.dataset}: {len(rows)}/{len(labels)} samples", file=sys.__stdout__, flush=True)
        return prediction

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"{args.dataset}_official.log"
    upstream.iterative_label_propagation = observe
    start = time.perf_counter()
    try:
        with log_path.open("w") as stream, redirect_stdout(stream):
            upstream.main(argparse.Namespace(
                datasets=args.dataset, features_path=str(args.features_path),
                task="fine_grained", clip_model="RN", mode="ZS", num_shots=16,
                k_text=3, k_image=8, k_fewshot=8, gamma=10., alpha=1.,
                num_iterations=3, output_file=None))
    finally:
        upstream.iterative_label_propagation = original
    if len(rows) != len(labels):
        raise ValueError("Observed sample count differs from the full dataset")
    counts = {k: sum(r["outcome"] == k for r in rows) for k in OUTCOMES}
    correct_clip = sum(r["clip_correct"] for r in rows)
    correct_ecalp = sum(r["ecalp_correct"] for r in rows)
    assert counts["corrected"] - counts["regressed"] == correct_ecalp - correct_clip
    accuracy = {"clip": 100 * correct_clip / len(rows), "ecalp": 100 * correct_ecalp / len(rows)}
    for variant, value in accuracy.items():
        if abs(value - float(reference[variant]["accuracy"])) > .00501:
            raise ValueError(f"Captured {variant} accuracy {value} differs from first formal run")
    predictions_path = args.output_dir / f"{args.dataset}_predictions.csv"
    with predictions_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = dict(dataset=args.dataset, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    source="observed_official_inference", reference_run=complete[-1],
                    feature=feature_info, commit=git("rev-parse", "HEAD"),
                    source_sha256={name: sha256(ROOT / name) for name in
                                   ("main.py", "utils/iterative_label_propagation.py",
                                    "experiments/collect_demo_predictions.py")},
                    torch=torch.__version__, device=args.device,
                    sample_count=len(rows), correct_clip=correct_clip, correct_ecalp=correct_ecalp,
                    accuracy=accuracy, outcomes=counts, seconds=time.perf_counter()-start,
                    predictions_sha256=sha256(predictions_path),
                    notes="IDs are original feature-row indices; class names and raw images unavailable. "
                    "Historical neighbor pseudo-labels are captured BEFORE current-sample propagation. "
                    "All counts cover the full dataset; selected cases include failures.")
    (args.output_dir / f"{args.dataset}_cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (args.output_dir / f"{args.dataset}_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(dataset=args.dataset, accuracy=accuracy, outcomes=counts), indent=2), flush=True)


if __name__ == "__main__":
    main()
