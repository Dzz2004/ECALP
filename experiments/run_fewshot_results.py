"""Run the DTD RN50 shot comparison in independent ECALP processes."""

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from run_main_results import ROOT, git, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-path", type=Path, default=ROOT / "features")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")

    args.features_path = args.features_path.resolve()
    test_file = args.features_path / "fine_grained/RN/DTD.pt"
    support_file = test_file.parent / "few_shots/DTD.pt"
    test_hash, support_hash = sha256(test_file), sha256(support_file)
    status = git("status", "--porcelain", "--untracked-files=no")
    if any(line.endswith((" main.py", " utils/iterative_label_propagation.py"))
           for line in status.splitlines()):
        parser.error("commit inference code changes before measuring")

    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads),
           "MKL_NUM_THREADS": str(args.threads)}
    os.environ.update({key: env[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    import torch

    device = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = args.output_dir / "logs" / run_id
    log_dir.mkdir(parents=True)
    rows = []
    for shots in (0, 1, 16):
        mode = "FS" if shots else "ZS"
        command = [sys.executable, "-u", "main.py", "--task", "fine_grained",
                   "--datasets", "DTD", "--clip_model", "RN", "--mode", mode,
                   "--num_shots", str(shots), "--num_iterations", "3",
                   "--features_path", str(args.features_path)]
        log = log_dir / f"DTD_{shots}shot.log"
        print(f"Running {shots}-shot ({device}): {log}", flush=True)
        start = time.perf_counter()
        with log.open("w", encoding="utf-8") as output:
            display_command = subprocess.list2cmdline(command)
            print(display_command, file=output, flush=True)
            subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                           stderr=subprocess.STDOUT, check=True)
        match = re.search(r"Top-1 Accuracy\(LP\):\s*([\d.]+)%", log.read_text(encoding="utf-8"))
        if not match:
            raise RuntimeError(f"Accuracy missing from {log}")
        rows.append(dict(dataset="DTD", clip_model="RN", mode=mode, num_shots=shots,
                         variant="ecalp", accuracy=match[1], command=display_command,
                         commit=git("rev-parse", "HEAD"), notes="official order; T=3",
                         source="ours", device=device, torch_version=torch.__version__,
                         threads=args.threads, seconds=f"{time.perf_counter() - start:.2f}",
                         feature_sha256=test_hash, support_sha256=support_hash if shots else "",
                         log=log.relative_to(args.output_dir).as_posix()))
        print(f"{shots}-shot: {match[1]}%", flush=True)

    temporary = log_dir / "fewshot_results.csv"
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output_dir / "fewshot_results.csv")


if __name__ == "__main__":
    main()
