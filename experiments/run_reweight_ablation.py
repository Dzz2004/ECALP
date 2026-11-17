"""Run paired RN50 zero-shot context reweighting ablations."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("dataset", "clip_model", "mode", "num_shots", "variant", "accuracy",
          "command", "commit", "notes", "source", "run_id", "seconds", "device",
          "torch_version", "feature_sha256", "log")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("DTD", "eurosat"),
                        default=("DTD", "eurosat"))
    parser.add_argument("--features-path", type=Path, default=ROOT / "features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    import torch

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = args.output_dir / "logs" / run_id
    log_dir.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.threads)
    env["MKL_NUM_THREADS"] = str(args.threads)
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    metadata = {
        "run_id": run_id, "commit": commit, "torch_version": torch.__version__,
        "device": device, "threads": args.threads,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                          (ROOT / "main.py", ROOT / "utils/iterative_label_propagation.py",
                           Path(__file__).resolve())},
        "features": {},
    }
    csv_path = args.output_dir / "reweight_ablation.csv"
    for dataset in args.datasets:
        feature_path = args.features_path / "fine_grained" / "RN" / f"{dataset}.pt"
        feature_hash = sha256(feature_path)
        metadata["features"][dataset] = feature_hash
        (log_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for variant, flags in (("context_reweight", ()), ("raw_cosine", ("--no-context-reweight",))):
            command = [sys.executable, "-u", str(ROOT / "main.py"), "--task", "fine_grained",
                       "--datasets", dataset, "--clip_model", "RN", "--mode", "ZS",
                       "--features_path", str(args.features_path.resolve()), *flags]
            log_path = log_dir / f"{dataset}_{variant}.log"
            print(f"Running {dataset} {variant}", flush=True)
            start = time.perf_counter()
            with log_path.open("w") as log:
                log.write(subprocess.list2cmdline(command) + "\n")
                log.flush()
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Run failed with exit code {result.returncode}: {log_path}")
            match = re.search(r"Top-1 Accuracy\(LP\):\s*([\d.]+)%", log_path.read_text())
            if not match:
                raise RuntimeError(f"Accuracy missing from {log_path}")
            row = dict(dataset=dataset, clip_model="RN", mode="ZS", num_shots=0,
                       variant=variant, accuracy=match.group(1),
                       command=subprocess.list2cmdline(command), commit=commit,
                       notes="image-image edges only; same feature order and default ECALP parameters",
                       source="ours", run_id=run_id, seconds=f"{time.perf_counter() - start:.4f}",
                       device=device, torch_version=torch.__version__,
                       feature_sha256=feature_hash, log=str(log_path.relative_to(args.output_dir)))
            with csv_path.open("a", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=FIELDS)
                if output.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            print(f"{dataset} {variant}: {row['accuracy']}% ({row['seconds']}s)", flush=True)
    print(f"Results: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
