"""Actual photo -> frozen RN50 -> CLIP and ECALP predictions with recorded history."""
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from match_image_cases import ROOT, load_encoder, transform, encode_image


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, default=102, help="Verified original sample index")
    parser.add_argument("--image", type=Path, help="Optional different input photo")
    parser.add_argument("--history-count", type=int, help="Number of preceding DTD feature rows")
    parser.add_argument("--cuda-device", default="4")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    matches = json.loads((ROOT / "results/demo/image_matches.json").read_text())
    if args.image is None:
        selected = next((m for m in matches if m["sample_index"] == args.case), None)
        if selected is None:
            parser.error("--case must identify a verified image in image_matches.json")
        path = ROOT / selected["image"]
    else:
        path = args.image.resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    selected = next((m for m in matches if m["image_sha256"] == digest), None)
    history_count = args.history_count if args.history_count is not None else (selected["sample_index"] if selected else 100)
    names = json.loads((ROOT / "results/demo/class_names.json").read_text())
    data = torch.load(ROOT / "features/fine_grained/RN/DTD.pt", map_location="cpu", weights_only=False)
    if not 0 <= history_count <= len(data["image_features"]):
        parser.error("--history-count is outside the available history")
    print(f"INPUT: {path}", flush=True)
    print(f"HISTORY: first {history_count} DTD test features, without true labels", flush=True)
    encoder = load_encoder(device)
    tensor = transform()(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    image_vector = F.normalize(encode_image(encoder, tensor), dim=1)
    del encoder
    texts = F.normalize(data["text_features"].float(), dim=1).to(device)
    scores = image_vector @ texts.T
    clip = int(scores.argmax(1))
    history = data["image_features"][:history_count].to(device)
    sequence = torch.cat((history, image_vector), dim=0)
    sys.path.insert(0, str(ROOT))
    propagate = importlib.import_module("utils.iterative_label_propagation").iterative_label_propagation
    for i in range(len(sequence)):
        observed = torch.cat((sequence[:i+1], sequence[i].unsqueeze(0)), dim=0)
        output = propagate(observed, texts, 3, 8, 8, 10., 1., max_iter=3)
    ecalp = int(output.argmax(1))
    truth = selected["class_id"] if selected else None
    print(f"CLIP OUTPUT: {names[str(clip)]} (C{clip:02d})", flush=True)
    print(f"ECALP OUTPUT: {names[str(ecalp)]} (C{ecalp:02d})", flush=True)
    if truth is not None:
        print(f"GROUND TRUTH: {names[str(truth)]} (C{truth:02d})", flush=True)
    matched = selected is not None and history_count == selected["sample_index"]
    if matched:
        with (ROOT / "results/demo/DTD_predictions.csv").open(newline="") as stream:
            reference = next(r for r in csv.DictReader(stream) if int(r["sample_index"]) == selected["sample_index"])
        if (clip, ecalp) != (int(reference["clip_pred"]), int(reference["ecalp_pred"])):
            raise RuntimeError("Actual-photo predictions differ from the captured feature-row predictions")
        print("Actual-photo outputs match the originally recorded case.", flush=True)
    record = dict(image=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                  image_sha256=digest, history_count=history_count, clip_pred=clip,
                  clip_name=names[str(clip)], ecalp_pred=ecalp, ecalp_name=names[str(ecalp)],
                  ground_truth=truth, ground_truth_name=names[str(truth)] if truth is not None else None,
                  matches_recorded_case=matched, case_index=selected["sample_index"] if selected else None,
                  source="actual_image_encoded_and_history_replayed", torch=torch.__version__,
                  input_transform="CLIP RN50 resize 224 bicubic, center crop, RGB normalization")
    out = ROOT / "results/demo/live_outputs"
    out.mkdir(exist_ok=True)
    label = f"DTD_{selected['sample_index']}" if selected else digest[:12]
    (out / f"{label}_history_{history_count}.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
