"""Validate official features, evaluate CLIP, and run unmodified ECALP separately."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["dataset", "clip_model", "mode", "num_shots", "variant", "accuracy",
          "command", "commit", "notes", "source", "run_id", "seconds", "device",
          "torch_version", "feature_sha256", "log"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def norm_stats(tensor):
    norms = tensor.float().norm(dim=-1)
    return {"min": norms.min().item(), "mean": norms.mean().item(),
            "max": norms.max().item()}


def load_features(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing official feature file: {path}")
    # Use only the trusted upstream feature archive; it uses torch serialization.
    data = torch.load(path, map_location="cpu", weights_only=False)
    images, texts, labels = (data[key] for key in
                             ("image_features", "text_features", "labels"))
    if not all(isinstance(t, torch.Tensor) for t in (images, texts, labels)):
        raise ValueError("image_features, text_features and labels must be tensors")
    if images.ndim not in (2, 3) or texts.ndim != 2:
        raise ValueError("Expected images [N,D] or [N,V,D], texts [C,D]")
    if images.shape[0] == 0 or texts.shape[0] < 2 or images.shape[-1] != texts.shape[-1]:
        raise ValueError("Empty features or incompatible feature dimensions")
    if labels.ndim != 1 or len(labels) != len(images):
        raise ValueError("Expected integer labels [N] matching the image count")
    if labels.is_floating_point() or labels.is_complex() or labels.dtype == torch.bool:
        raise ValueError("Labels must have an integer dtype")
    if labels.min() < 0 or labels.max() >= len(texts):
        raise ValueError("Labels must lie in [0, num_classes)")
    for name, tensor in (("images", images), ("texts", texts)):
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"{name}: expected finite floating-point features")
        if (tensor.float().norm(dim=-1) == 0).any():
            raise ValueError(f"{name}: found zero feature vectors")
    summary = {"path": str(path.resolve()), "sha256": sha256(path),
               "image_shape": list(images.shape), "text_shape": list(texts.shape),
               "label_shape": list(labels.shape), "image_dtype": str(images.dtype),
               "text_dtype": str(texts.dtype), "label_dtype": str(labels.dtype),
               "label_range": [labels.min().item(), labels.max().item()],
               "class_counts": torch.bincount(labels.long(), minlength=len(texts)).tolist(),
               "image_norms": norm_stats(images), "text_norms": norm_stats(texts)}
    return images, texts, labels, summary


@torch.no_grad()
def clip_accuracy(images, texts, labels):
    # Average views as upstream iterative_label_propagation does, then use cosine.
    images = images.mean(dim=1) if images.ndim == 3 else images
    if (images.float().norm(dim=-1) == 0).any():
        raise ValueError("View averaging produced a zero image vector")
    images = F.normalize(images.float(), dim=-1)
    texts = F.normalize(texts.float(), dim=-1)
    predictions = (images @ texts.T).argmax(dim=-1)
    return (predictions == labels).sum().item() * 100 / len(labels)


def append_result(path, row):
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as stream:
            if next(csv.reader(stream)) != FIELDS:
                raise ValueError(f"Unexpected CSV schema: {path}")
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["DTD", "eurosat"],
                        default=["DTD", "eurosat"])
    parser.add_argument("--features-path", type=Path, default=ROOT / "features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--variant", choices=["both", "clip", "ecalp"], default="both")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:PHYSICAL_INDEX")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Mark synthetic runs as smoke_test")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif re.fullmatch(r"cuda:\d+", args.device):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]
    elif args.device != "auto":
        parser.error("--device must be auto, cpu, or cuda:PHYSICAL_INDEX")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    cuda = torch.cuda.is_available()
    if args.device.startswith("cuda:") and not cuda:
        raise RuntimeError("Requested CUDA is unavailable; check GPU access outside the sandbox")
    device = torch.cuda.get_device_name(0) if cuda else "cpu"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = args.output_dir / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=False)
    commit = git("rev-parse", "HEAD")
    tracked = git("status", "--porcelain")
    # Includes uncommitted experiment code, so a base commit alone is not misleading.
    sources = [ROOT / "main.py", ROOT / "utils/iterative_label_propagation.py",
               Path(__file__).resolve(), ROOT / "requirements.txt"]
    metadata = {"run_id": run_id, "commit": commit, "git_status": tracked,
                "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
                "python": sys.version, "executable": sys.executable,
                "torch": torch.__version__, "cuda_build": torch.version.cuda,
                "device": device, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "threads": args.threads, "command": shlex.join([sys.executable, *sys.argv]),
                "source": "smoke_test" if args.smoke_test else "ours",
                "ecalp_parameters": {"k_text": 3, "k_image": 8, "k_fewshot": 8,
                                     "gamma": 10, "alpha": 1, "iterations": 3, "beta": 0.2},
                "features": {}}
    metadata_path = log_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for dataset in args.datasets:
        path = args.features_path / "fine_grained/RN" / f"{dataset}.pt"
        images, texts, labels, summary = load_features(path)
        metadata["features"][dataset] = summary
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{dataset}: images {list(images.shape)}, texts {list(texts.shape)}, "
              f"labels {summary['label_range']}", flush=True)
        if args.inspect_only:
            continue
        for variant in (["clip", "ecalp"] if args.variant == "both" else [args.variant]):
            log_path = log_dir / f"{dataset}_{variant}.log"
            start = time.perf_counter()
            if variant == "clip":
                accuracy = clip_accuracy(images, texts, labels)
                command = metadata["command"]
                log_path.write_text(f"Same-feature CLIP cosine Top-1: {accuracy:.8f}%\n")
            else:
                cmd = [sys.executable, "-u", str(ROOT / "main.py"), "--task", "fine_grained",
                       "--datasets", dataset, "--clip_model", "RN", "--mode", "ZS",
                       "--features_path", str(args.features_path.resolve())]
                command = shlex.join(cmd)
                print(f"Running official ECALP: {dataset}; log: {log_path}", flush=True)
                with log_path.open("w") as stream:
                    stream.write(command + "\n")
                    stream.flush()
                    result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"Official ECALP exited {result.returncode}; see {log_path}")
                match = re.search(r"Top-1 Accuracy\(LP\):\s*([\d.]+)%", log_path.read_text())
                if not match:
                    raise RuntimeError(f"No official accuracy found; see {log_path}")
                accuracy = float(match.group(1))
            seconds = time.perf_counter() - start
            row = dict(dataset=dataset, clip_model="RN", mode="ZS", num_shots=0,
                       variant=variant, accuracy=f"{accuracy:.8f}", command=command,
                       commit=commit, notes="same official features; see metadata.json for hashes, "
                       "input order and uncommitted code; ECALP accuracy rounded upstream to 0.01%",
                       source=metadata["source"], run_id=run_id, seconds=f"{seconds:.4f}",
                       device="cpu" if variant == "clip" else device,
                       torch_version=torch.__version__, feature_sha256=summary["sha256"],
                       log=str(log_path.relative_to(args.output_dir)))
            append_result(args.output_dir / "main_results.csv", row)
            print(f"{dataset} {variant}: {accuracy:.4f}% ({seconds:.1f}s)", flush=True)
    print(f"Metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
