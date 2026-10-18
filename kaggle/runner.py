from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


PAYLOAD = "__REV_BUNDLE_BASE64__"
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
SOURCE = WORKING / "rev-source"
OUTPUT = WORKING / "output"
PACKAGES = ["transformers==5.19.0", "peft==0.21.2", "tokenizers==0.23.2", "safetensors==0.8.0", "huggingface-hub==1.33.0", "accelerate==1.15.0", "pyarrow>=15"]


def main() -> None:
    if PAYLOAD.startswith("__REV_"):
        raise RuntimeError("Package this runner with python scripts/kaggle.py before submitting it")
    import torch

    gpus = torch.cuda.device_count()
    print(f"torch={torch.__version__} gpus={[torch.cuda.get_device_name(i) for i in range(gpus)]}")
    if not gpus:
        raise RuntimeError("Enable a GPU accelerator (T4 x2) for this notebook")
    constraints = WORKING / "constraints.txt"
    constraints.write_text(f"torch=={torch.__version__}\n", encoding="utf-8")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-c", str(constraints), *PACKAGES])
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    SOURCE.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
        archive.extractall(SOURCE)
    extra = json.loads((SOURCE / "run.json").read_text(encoding="utf-8")).get("extra_packages", [])
    if extra:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-c", str(constraints), *extra])
    previous = sorted(INPUT.rglob("output/trainer.pt")) if INPUT.exists() else []
    if previous and not (OUTPUT / "trainer.pt").exists():
        print(f"Resuming from {previous[0].parent}")
        shutil.copytree(previous[0].parent, OUTPUT, dirs_exist_ok=True)
    config = json.loads((SOURCE / "run.json").read_text(encoding="utf-8"))
    config.update({"output_dir": str(OUTPUT), "data_dir": str(OUTPUT / "data"), "source_data_dir": str(INPUT)})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_path = WORKING / "run.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    train = str(SOURCE / "train.py")
    if not (OUTPUT / "data" / "train.jsonl").exists():
        subprocess.check_call([sys.executable, train, "--config", str(config_path), "--prepare"])
    os.environ.update(config.get("env", {}))
    processes = min(config.get("gpus") or gpus, gpus)
    subprocess.check_call([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={processes}", train, "--config", str(config_path), "--resume"])
    best, data = OUTPUT / "best", OUTPUT / "data"
    finished = '"training_complete"' in (OUTPUT / "train.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    if best.exists() and finished:
        subprocess.check_call([sys.executable, str(SOURCE / "evaluate.py"), "--model", str(best), "--data", str(data / "validation.jsonl"), "--calibrate", str(data / "calibration.jsonl"), "--device", "cuda", "--batch-size", "8", "--output", str(OUTPUT / "evaluation.json")])
    archive_path = WORKING / "rev-output.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in OUTPUT.rglob("*"):
            if path.is_file() and "data" not in path.relative_to(OUTPUT).parts:
                archive.write(path, path.relative_to(OUTPUT.parent))
    print(f"Outputs: {OUTPUT}")
    print(f"Archive without data: {archive_path}")


if __name__ == "__main__":
    main()
