"""Run the controlled image-image edge ablation in isolated processes."""

import argparse
import csv
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

import torch


ACCURACY_PATTERN = re.compile(r"Top-1 Accuracy\(LP\):\s*([0-9.]+)%")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ECALP graph ablation")
    parser.add_argument("--dataset", default="DTD")
    parser.add_argument("--features_path", default="./features/")
    parser.add_argument("--clip_model", choices=["RN", "VIT"], default="RN")
    parser.add_argument("--k_text", type=int, default=3)
    parser.add_argument("--k_image", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--output_csv", default="results/graph_ablation.csv")
    return parser.parse_args()


def command_text(command):
    return " ".join(shlex.quote(part) for part in command)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_variant(repo_root, common_command, dataset, variant, extra_args, log_dir):
    command = common_command + extra_args
    completed = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
    log_path = log_dir / f"{dataset}_{variant}.log"
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"{variant} failed; see {log_path}")
    match = ACCURACY_PATTERN.search(completed.stdout)
    if not match:
        raise RuntimeError(f"No accuracy found for {variant}. Check the feature path and {log_path}.")
    return float(match.group(1)), command_text(command), log_path


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_csv = (repo_root / args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = output_csv.parent / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    common_command = [
        sys.executable, "main.py", "--task", "fine_grained", "--datasets", args.dataset,
        "--features_path", args.features_path, "--clip_model", args.clip_model, "--mode", "ZS",
        "--k_text", str(args.k_text), "--k_image", str(args.k_image), "--gamma", str(args.gamma),
        "--alpha", str(args.alpha), "--num_iterations", str(args.num_iterations),
    ]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
                            capture_output=True, check=True).stdout.strip()
    variants = [("full_ecalp", []), ("no_image_image_edges", ["--disable_image_image_edges"])]
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    rows = []
    for variant, extra_args in variants:
        accuracy, command, log_path = run_variant(
            repo_root, common_command, args.dataset, variant, extra_args, log_dir
        )
        rows.append({
            "dataset": args.dataset, "clip_model": args.clip_model, "mode": "ZS", "num_shots": 0,
            "variant": variant, "accuracy": f"{accuracy:.2f}", "command": command, "commit": commit,
            "notes": "Only image-image edge inclusion differs between variants", "source": "ours",
            "device": device, "pytorch_version": torch.__version__,
            "run_id": run_id,
            "log": log_path.relative_to(output_csv.parent).as_posix(),
        })
    fieldnames = list(rows[0])
    existing_rows = []
    if output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8") as csv_file:
            existing_rows = list(csv.DictReader(csv_file))
    run_key = (args.dataset.lower(), args.clip_model, "ZS")
    existing_rows = [row for row in existing_rows
                     if (row["dataset"].lower(), row["clip_model"], row["mode"]) != run_key]
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows + rows)

    source_paths = [
        repo_root / "main.py",
        repo_root / "utils" / "iterative_label_propagation.py",
        Path(__file__).resolve(),
    ]
    metadata = {
        "run_id": run_id,
        "commit": commit,
        "git_status": subprocess.run(
            ["git", "status", "--short"], cwd=repo_root, text=True,
            capture_output=True, check=True
        ).stdout,
        "source_sha256": {
            path.relative_to(repo_root).as_posix(): sha256(path) for path in source_paths
        },
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch.__version__,
        "device": device,
        "source": "ours",
        "dataset": args.dataset,
        "parameters": {
            "clip_model": args.clip_model, "mode": "ZS", "num_shots": 0,
            "k_text": args.k_text, "k_image": args.k_image, "gamma": args.gamma,
            "alpha": args.alpha, "iterations": args.num_iterations,
        },
        "variants": rows,
    }
    (log_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved results to {output_csv}")


if __name__ == "__main__":
    main()
