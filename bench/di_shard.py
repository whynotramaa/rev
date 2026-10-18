import hashlib
import json
import sys
from pathlib import Path

import torch
from decision_index.engines.base import Engine

SUITE = Path("/tmp/suite-0.3")


class RevEngine(Engine):
    name = "rev"
    latency = "In-process request wall time on one Kaggle T4 in fp16, including tokenization; excludes model loading."

    def __init__(self, model, source, **options):
        super().__init__(**options)
        sys.path.insert(0, source)
        import model as rev_model

        rev_model.weight_dtype = lambda device: torch.float16 if torch.device(device).type == "cuda" else torch.float32
        from demo import make_decider

        self.decide, self.encoder = make_decider(model, device="cuda", max_length=0)
        self.provenance = {"kind": "in-process", "checkpoint": model, "policy": "Unmodified state and questions; no truncation; up to 255 options."}

    def __call__(self, state, questions):
        try:
            response = self.decide({"state": state, "questions": questions})
        except torch.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise ValueError(f"T4 out of memory: {exc}") from None
        return response, response

    def synchronize(self):
        torch.cuda.synchronize()

    def runtime(self):
        return {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}


if __name__ == "__main__":
    from decision_index.runner import run
    from decision_index.suite.io import Suite

    index, count, checkpoint, source, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    suite = Suite(SUITE, "0.3")
    from importlib.resources import files
    panel = {b for a in json.loads(files("decision_index").joinpath("data/index-0.3.json").read_text())["areas"] for b in a["benchmarks"]}

    def keep(e):
        return e["catalog_id"] in panel and suite.in_edition(e) and int(hashlib.sha256(e["run_id"].encode()).hexdigest(), 16) % count == index

    from decision_index.runner import iter_rows

    rows = sorted(iter_rows(suite.row_paths, keep), key=lambda r: len(json.dumps([r["state"], r["questions"]], ensure_ascii=False)))
    ordered = Path(out) / "ordered-rows.jsonl"
    ordered.parent.mkdir(parents=True, exist_ok=True)
    with ordered.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    del rows
    print(json.dumps(run("di_shard:RevEngine", {"model": checkpoint, "source": source}, ordered, out, compact=True)))
