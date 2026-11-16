"""Extract only the RN50 DTD/EuroSAT files required by the course project."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

FILES = ("DTD.pt", "eurosat.pt", "few_shots/DTD.pt")
SOURCE = "https://drive.google.com/file/d/1-CXxtY77_Zj2Q1T2x0CFN-5a0xjKtz_A/view"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("features_zip/fine_grained.tar.gz"))
    parser.add_argument("--features-path", type=Path, default=Path("features"))
    parser.add_argument("--manifest", type=Path, default=Path("results/data_manifest.json"))
    parser.add_argument("--require-fewshot", action="store_true",
                        help="Also fail if the archive lacks the DTD support set")
    args = parser.parse_args()
    destination = args.features_path / "fine_grained/RN"
    # The official archive has a Data05/liyushu/ECALP/features/ prefix.
    wanted = {f"fine_grained/RN/{name}": name for name in FILES}
    found = {}
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            for suffix, name in wanted.items():
                if member.name == suffix or member.name.endswith("/" + suffix):
                    if not member.isfile() or name in found:
                        raise ValueError(f"Unexpected or duplicate archive member: {member.name}")
                    target = destination / name
                    if target.exists():
                        raise FileExistsError(f"Refusing to overwrite {target}; use another features path")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(".pt.partial")
                    with archive.extractfile(member) as source, temporary.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    temporary.replace(target)
                    found[name] = {"archive_member": member.name, "bytes": member.size,
                                   "path": str(target), "sha256": sha256(target)}
                    print(f"Extracted {target} ({member.size} bytes)", flush=True)
    missing = sorted(set(FILES) - set(found))
    required = set(FILES if args.require_fewshot else FILES[:2])
    missing_required = sorted(required - set(found))
    manifest = {"source": SOURCE, "archive": str(args.archive),
                "archive_sha256": sha256(args.archive), "files": found, "missing": missing,
                "missing_required": missing_required}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    if missing_required:
        raise FileNotFoundError(f"Required official features absent from archive: {missing_required}")
    if missing:
        print(f"Optional few-shot features absent: {missing}; record for member D", flush=True)
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
