import base64
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEMO = "__DEMO_BASE64__"
SHARD = "__SHARD_BASE64__"
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
RUNS = WORKING / "di-runs"
SUITE = Path("/tmp/suite-0.3")
KIT = "git+https://github.com/apolinario/decision-index@f5a6e4fd85fed05e77878dfc4966d0568bfb45ec"
PACKAGES = ["transformers==5.19.0", "peft==0.21.2", "tokenizers==0.23.2", "safetensors==0.8.0", "huggingface-hub==1.33.0", "accelerate==1.15.0", "httpx", KIT]
BUDGET_SECONDS = 11.2 * 3600


def find(pattern):
    hits = sorted(INPUT.rglob(pattern))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def main():
    start = time.time()
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
    import torch

    gpus = torch.cuda.device_count()
    print(f"torch={torch.__version__} gpus={[torch.cuda.get_device_name(i) for i in range(gpus)]}")
    checkpoint = find("output/best/config.json").parent
    source = WORKING / "rev-source"
    shutil.copytree(find("rev-source/model.py").parent, source, dirs_exist_ok=True)
    (source / "demo.py").write_bytes(base64.b64decode(DEMO))
    (source / "di_shard.py").write_bytes(base64.b64decode(SHARD))
    shutil.copytree(find("excluded-questions.json").parent, SUITE, dirs_exist_ok=True)
    for plain in SUITE.glob("*.jsonl"):
        with plain.open("rb") as src, gzip.open(plain.with_suffix(".jsonl.gz"), "wb", compresslevel=1) as dst:
            shutil.copyfileobj(src, dst)
        plain.unlink()
    RUNS.mkdir(parents=True, exist_ok=True)
    for previous in INPUT.rglob("di-runs/shard*/results.jsonl"):
        target = RUNS / previous.parent.name
        if not (target / "results.jsonl").exists():
            shutil.copytree(previous.parent, target, dirs_exist_ok=True)
    print(f"checkpoint={checkpoint} suite={sorted(p.name for p in SUITE.iterdir())}", flush=True)
    procs = []
    for i in range(gpus):
        log = open(WORKING / f"shard{i}.log", "a")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(i), "PYTHONPATH": str(source)}
        procs.append(subprocess.Popen([sys.executable, str(source / "di_shard.py"), str(i), str(gpus), str(checkpoint), str(source), str(RUNS / f"shard{i}")], env=env, stdout=log, stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in procs) and time.time() - start < BUDGET_SECONDS:
        time.sleep(120)
        status = {i: json.loads(path.read_text()) if (path := RUNS / f"shard{i}" / "status.json").exists() else None for i in range(gpus)}
        print(round((time.time() - start) / 60), json.dumps(status), flush=True)
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        p.wait()
    print("exit codes", [p.returncode for p in procs])
    for i in range(gpus):
        print(f"--- shard{i}.log tail")
        print((WORKING / f"shard{i}.log").read_text()[-3000:])


if __name__ == "__main__":
    main()
