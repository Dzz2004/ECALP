"""Show packaged real-image cases, or run actual image inference with --live."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=int, choices=[102, 1516], default=102)
    parser.add_argument("--live", action="store_true", help="Actually encode the photo and replay history (GPU required)")
    parser.add_argument("--cuda-device", default="4")
    args = parser.parse_args()
    matches = json.loads((HERE / "cases/image_matches.json").read_text())
    match = next(m for m in matches if m["sample_index"] == args.case)
    image = HERE / "cases/images" / Path(match["image"]).name
    output = json.loads((HERE / "cases/outputs" / f"DTD_{args.case}_history_{args.case}.json").read_text())
    if hashlib.sha256(image.read_bytes()).hexdigest() != match["image_sha256"] or output["image_sha256"] != match["image_sha256"]:
        raise ValueError("Packaged image differs from the verified case")
    if args.live:
        python = ROOT / ".venv/bin/python"
        executable = str(python) if python.exists() else sys.executable
        command = [executable, str(ROOT / "experiments/predict_image_case.py"),
                   "--image", str(image), "--history-count", str(args.case),
                   "--cuda-device", args.cuda_device]
        raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)
    print("回放已记录的真实原图推理结果；加 --live 可现场重新计算。")
    print(f"输入图片：{image}")
    print(f"CLIP 输出：{output['clip_name']}")
    print(f"ECALP 输出：{output['ecalp_name']}")
    print(f"真实类别：{output['ground_truth_name']}")
    print(f"ECALP 上下文：之前 {output['history_count']} 张测试图像，未使用真实标签。")
    print(f"对比图：{HERE / 'figures/real_image_input_output.png'}")


if __name__ == "__main__":
    main()
