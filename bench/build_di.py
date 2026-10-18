import base64
import json
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
out = here / "kaggle-di"
out.mkdir(exist_ok=True)
script = (here / "kaggle_di.py").read_text()
script = script.replace("__DEMO_BASE64__", base64.b64encode((here.parent / "demo.py").read_bytes()).decode())
script = script.replace("__SHARD_BASE64__", base64.b64encode((here / "di_shard.py").read_bytes()).decode())
(out / "rev-di.py").write_text(script)
sources = sys.argv[1:] or ["whynotramaa/rev-train-resume"]
(out / "kernel-metadata.json").write_text(json.dumps({"id": "whynotramaa/rev-di", "title": "rev-di", "code_file": "rev-di.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": ["whynotramaa/decision-index-suite-03"], "competition_sources": [], "kernel_sources": sources}, indent=2) + "\n")
print(out)
