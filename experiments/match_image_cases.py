"""Match original DTD photos to existing feature rows using frozen CLIP RN50."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/raw-case-assets"
MODEL_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"


def transform():
    return Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224),
                    ToTensor(), Normalize((.48145466, .4578275, .40821073),
                                          (.26862954, .26130258, .27577711))])


def load_encoder(device):
    path = CACHE / "RN50.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError("RN50 checkpoint does not match the official OpenAI checksum")
    encoder = torch.jit.load(str(path), map_location=device).float().eval()
    # As in OpenAI CLIP's load-time dtype patch, change serialized FP16 casts
    # as well as parameter dtypes. RN50's traced visual stem also has such a cast.
    for module in encoder.modules():
        for method in module._c._method_names():
            graph = module._c._get_method(method).graph
            for node in graph.findAllNodes("aten::to"):
                for value in list(node.inputs())[1:3]:
                    constant = value.node()
                    if (constant.kind() == "prim::Constant" and "value" in constant.attributeNames()
                            and constant.kindOf("value") == "i" and constant.i("value") == 5):
                        constant.i_("value", 6)  # torch.float16 -> torch.float32
    return encoder


def encode_image(encoder, tensor):
    # All serialized dtype casts have been patched consistently with FP32 weights.
    return encoder.visual(tensor.float()).float()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-device", default="4")
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[102])
    parser.add_argument("--find-regression", action="store_true",
                        help="Also select the first unambiguously matched regression after 100 arrivals")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    model = load_encoder(device)
    preprocessing = transform()
    manifest = json.loads((CACHE / "candidate_manifest.json").read_text())
    records, vectors = [], []
    for offset in range(0, len(manifest), 16):
        batch_records = manifest[offset:offset+16]
        batch = torch.stack([preprocessing(Image.open(ROOT / r["path"]).convert("RGB"))
                             for r in batch_records]).to(device)
        features = F.normalize(encode_image(model, batch), dim=1).cpu()
        vectors.append(features)
        records.extend(batch_records)
    candidates = torch.cat(vectors)
    official = torch.load(ROOT / "features/fine_grained/RN/DTD.pt", map_location="cpu", weights_only=False)
    expected = F.normalize(official["image_features"].float(), dim=1)
    if args.find_regression:
        with (ROOT / "results/demo/DTD_predictions.csv").open(newline="") as stream:
            regressions = [r for r in csv.DictReader(stream) if r["outcome"] == "regressed"
                           and int(r["sample_index"]) >= 99]
        for row in regressions:
            index = int(row["sample_index"])
            values, indices = (candidates @ expected[index]).topk(2)
            if (float(values[0]) >= .9999 and float(values[0]-values[1]) >= .001
                    and records[int(indices[0])]["class_id"] == int(official["labels"][index])):
                args.sample_indices.append(index)
                print(f"Selected unambiguously matched regression #{index}", flush=True)
                break
        else:
            raise ValueError("No regression satisfies the strict image/feature matching criteria")
    images_dir = ROOT / "results/demo/images"
    images_dir.mkdir(parents=True, exist_ok=True)
    matches = []
    for index in args.sample_indices:
        similarities = candidates @ expected[index]
        top_scores, top_indices = similarities.topk(2)
        best, second = map(float, top_scores)
        record = records[int(top_indices[0])]
        if best < .9999 or best-second < .001:
            raise ValueError(f"No unambiguous near-identical feature match for #{index}: {best}, {second}")
        if record["class_id"] != int(official["labels"][index]):
            raise ValueError("Matched image label differs from original feature label")
        target = images_dir / f"DTD_{index}_{Path(record['path']).name}"
        shutil.copyfile(ROOT / record["path"], target)
        match = dict(sample_index=index, image=str(target.relative_to(ROOT)),
                     class_id=record["class_id"], class_name=record["class_name"],
                     cosine_match=best, runner_up_cosine=second,
                     image_sha256=record["sha256"], source=record["source"],
                     checkpoint_sha256=MODEL_SHA256,
                     method="RN50 OpenAI FP32 center-crop feature matching; threshold 0.9999; margin 0.001")
        matches.append(match)
        print(json.dumps(match), flush=True)
    output = ROOT / "results/demo/image_matches.json"
    output.write_text(json.dumps(matches, indent=2) + "\n")
    shutil.copyfile(CACHE / "class_names.json", ROOT / "results/demo/class_names.json")
    print(output)


if __name__ == "__main__":
    main()
