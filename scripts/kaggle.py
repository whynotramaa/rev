from __future__ import annotations

import argparse
import base64
import io
import json
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("model.py", "data.py", "train.py", "evaluate.py")
MARKER = "__REV_BUNDLE_BASE64__"


def package(config: Path, output: Path, slug: str, dataset_sources: list[str], private: bool, smoke: bool, resume_from: list[str], overrides: dict) -> Path:
    for name in SOURCES:
        if not (ROOT / name).is_file():
            raise FileNotFoundError(f"required training source is missing: {ROOT / name}")
    if not config.is_file():
        raise FileNotFoundError(config)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            archive.write(ROOT / name, name)
        values = json.loads(config.read_text(encoding="utf-8"))
        if smoke:
            values.update(json.loads(config.with_name("smoke.json").read_text(encoding="utf-8")))
        values.update(overrides)
        archive.writestr("run.json", json.dumps(values, indent=2))
    runner = (ROOT / "kaggle" / "runner.py").read_text(encoding="utf-8")
    if runner.count(MARKER) != 1:
        raise ValueError("runner bundle marker is missing or duplicated")
    output.mkdir(parents=True, exist_ok=True)
    (output / "runner.py").write_text(runner.replace(MARKER, base64.b64encode(bundle.getvalue()).decode("ascii")), encoding="utf-8")
    metadata = {
        "id": slug,
        "title": slug.split("/")[1],
        "code_file": "runner.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": private,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": resume_from,
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle this project as a single Kaggle Python entry script")
    parser.add_argument("--config", required=True, type=Path, help="training JSON config")
    parser.add_argument("--output", type=Path, default=ROOT / "kaggle" / "submission")
    parser.add_argument("--slug", default="rev-decision-model")
    parser.add_argument("--dataset-source", action="append", default=[], help="Kaggle dataset slug, repeatable")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--resume-from", action="append", default=[], help="Kaggle notebook slug whose output holds output/trainer.pt")
    parser.add_argument("--set", type=json.loads, default={}, help="JSON object merged into the config last")
    parser.add_argument("--smoke", action="store_true", help="merge configs/smoke.json for a short throughput and memory check")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", args.slug):
        parser.error("--slug must have the form owner/kernel-name")
    path = package(args.config.resolve(), args.output.resolve(), args.slug, args.dataset_source, args.private, args.smoke, args.resume_from, args.set)
    print(f"Packaged entry script and metadata at {path}")
    print("Review the bundle, then upload it yourself with: kaggle kernels push -p " + str(path))


if __name__ == "__main__":
    main()
