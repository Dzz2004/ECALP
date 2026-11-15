"""Download a small DTD candidate set; authenticate bytes using mirror LFS hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache/raw-case-assets"
REPO = "cansa/Describable-Textures-Dataset-DTD"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classes", nargs="+", default=["knitted", "pleated"])
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    sources = {"split_zhou_DescribableTextures.json":
               "https://raw.githubusercontent.com/YBZh/DMN/main/data/data_splits/split_zhou_DescribableTextures.json"}
    sources.update({f"{c}-tree.json": f"https://huggingface.co/api/datasets/{REPO}/tree/main/images/{c}?limit=200"
                    for c in args.classes})
    checkpoint_sha = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
    sources["RN50.pt"] = f"https://openaipublic.azureedge.net/clip/models/{checkpoint_sha}/RN50.pt"
    for filename, url in sources.items():
        target = CACHE / filename
        if not target.exists():
            print(f"Downloading {filename}", flush=True)
            request = Request(url, headers={"User-Agent": "ECALP-course-case-matching/1.0"})
            with urlopen(request, timeout=120) as stream:
                content = stream.read()
            if filename == "RN50.pt" and hashlib.sha256(content).hexdigest() != checkpoint_sha:
                raise ValueError("Official RN50 checkpoint checksum mismatch")
            target.write_bytes(content)
    splits = json.loads((CACHE / "split_zhou_DescribableTextures.json").read_text())
    classes = {int(label): name for split in splits.values() for _, label, name in split}
    assert len(classes) == 47
    (CACHE / "class_names.json").write_text(json.dumps(classes, indent=2) + "\n")
    files = {}
    for category in args.classes:
        tree = json.loads((CACHE / f"{category}-tree.json").read_text())
        for item in tree:
            if item["type"] == "file":
                files[item["path"]] = item
    wanted = [(path, label, name) for path, label, name in splits["test"] if name in args.classes]

    def download(item):
        path, label, name = item
        relative = Path(path)
        if len(relative.parts) != 2 or relative.parts[0] not in args.classes:
            raise ValueError(f"Unexpected image path: {path}")
        info = files["images/" + path]
        expected = info["lfs"]["oid"]
        target = CACHE / "candidates" / path
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/images/{path}"
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            return dict(path=str(target.relative_to(ROOT)), class_id=label, class_name=name,
                        sha256=expected, source=url)
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                request = Request(url, headers={"User-Agent": "ECALP-course-case-matching/1.0"})
                with urlopen(request, timeout=45) as stream:
                    data = stream.read()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Image SHA-256 differs from mirror file metadata: {path}")
                target.write_bytes(data)
                return dict(path=str(target.relative_to(ROOT)), class_id=label, class_name=name,
                            sha256=expected, source=url)
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)
    with ThreadPoolExecutor(max_workers=6) as executor:
        records = list(executor.map(download, wanted))
    (CACHE / "candidate_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Downloaded and SHA-256 checked {len(records)} candidate test images.")


if __name__ == "__main__":
    main()
