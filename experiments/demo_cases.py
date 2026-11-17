"""Replay recorded cases without a GPU: inspect any row or browse outcome groups."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"corrected": "CLIP 错 → ECALP 对", "regressed": "CLIP 对 → ECALP 错",
          "both_correct": "两者都对", "both_wrong": "两者都错"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["DTD", "eurosat"], default="DTD")
    parser.add_argument("--outcome", choices=list(LABELS), default="corrected")
    parser.add_argument("--sample-index", type=int, help="Inspect any original feature-row index")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results/demo")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    with (args.data_dir / f"{args.dataset}_predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cases = json.loads((args.data_dir / f"{args.dataset}_cases.json").read_text())
    meta = json.loads((args.data_dir / f"{args.dataset}_metadata.json").read_text())
    names_path = args.data_dir / "class_names.json"
    names = json.loads(names_path.read_text()) if args.dataset == "DTD" and names_path.exists() else {}
    images_path = args.data_dir / "image_matches.json"
    images = {m["sample_index"]: m["image"] for m in json.loads(images_path.read_text())} if args.dataset == "DTD" and images_path.exists() else {}
    label = lambda value: f"C{int(value):02d}" + (f" ({names[str(value)]})" if str(value) in names else "")
    print(f"\n{args.dataset} / RN50 Zero-shot / 完整测试集 {len(rows)} 个样本")
    print(f"CLIP {meta['accuracy']['clip']:.2f}% → ECALP {meta['accuracy']['ecalp']:.2f}%")
    print("；".join(f"{LABELS[k]}：{v}" for k, v in meta["outcomes"].items()))
    print("\n样本编号是官方特征文件的行索引；类别编号从 0 开始。")
    print("历史邻居展示推理前的伪标签；已恢复原图的案例会显示图片路径。\n")
    if args.sample_index is not None:
        selected = [r for r in rows if int(r["sample_index"]) == args.sample_index]
        if not selected:
            parser.error("sample-index is outside the dataset")
    else:
        selected = [c for c in cases if c["outcome"] == args.outcome][:args.limit]
    details = {int(c["sample_index"]): c for c in cases}
    for r in selected:
        index = int(r["sample_index"])
        print(f"样本 #{index}：{LABELS[r['outcome']]}")
        print(f"  真实类别 {label(r['ground_truth'])} | "
              f"CLIP {label(r['clip_pred'])} | ECALP {label(r['ecalp_pred'])}")
        if index in images:
            print(f"  原图：{ROOT / images[index]}")
        case = details.get(index)
        if case:
            print("  CLIP 前三名：" + "，".join(
                f"C{v['class_id']:02d} (余弦 {v['cosine']:.4f})" for v in case["clip_top3"]))
            print("  部分历史 KNN：" + "，".join(
                f"#{v['sample_index']} / 先前伪标签 C{v['previous_pseudo_label']:02d}"
                for v in case["historical_neighbors"]))
            print("  这些连接展示实际推理上下文，不能单独证明某条边造成了该预测。")
        else:
            print("  此样本有完整预测记录；局部图快照只保留了预先规定的展示案例。")
        print()


if __name__ == "__main__":
    main()
