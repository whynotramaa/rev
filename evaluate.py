from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _target(example: dict[str, Any], keys: list[str], n: int) -> list[float]:
    target = example.get("targets", example.get("target", example.get("target_probs")))
    if isinstance(target, dict):
        values = [float(target.get(key, 0.0)) for key in keys]
    elif isinstance(target, list):
        values = [float(value) for value in target]
    else:
        label = example.get("label", example.get("answer"))
        if isinstance(label, bool):
            values = [float(i == int(label)) for i in range(n)]
        elif label in keys:
            values = [float(key == label) for key in keys]
        elif isinstance(label, int) and 0 <= label < n:
            values = [float(i == label) for i in range(n)]
        else:
            raise ValueError("example needs targets, target_probs, or a valid label")
    if len(values) != n or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("targets must be finite, nonnegative, and aligned to keys")
    total = sum(values)
    if abs(total - 1.0) > 1e-4:
        raise ValueError(f"target probabilities must sum to one, got {total}")
    return [x / total for x in values]


def _ece(confidence: list[float], correct: list[float], bins: int = 15) -> float:
    if not confidence:
        return 0.0
    result = 0.0
    for index in range(bins):
        lo, hi = index / bins, (index + 1) / bins
        members = [i for i, value in enumerate(confidence) if lo <= value < hi or (index == bins - 1 and value == 1)]
        if members:
            result += len(members) / len(confidence) * abs(
                sum(confidence[i] for i in members) / len(members)
                - sum(correct[i] for i in members) / len(members)
            )
    return result


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    n = len(rows)
    nll = sum(row["nll"] for row in rows) / n
    brier = sum(row["brier"] for row in rows) / n
    confidence = [max(row["probabilities"]) for row in rows]
    accuracy = [float(row["prediction_index"] == row["target_index"]) for row in rows]
    calibrated_correctness = [row["target"][row["prediction_index"]] for row in rows]
    order = sorted(range(n), key=lambda i: (-confidence[i], i))
    risk_coverage = [{"coverage": (step + 1) / n, "risk": 1 - sum(accuracy[i] for i in order[:step + 1]) / (step + 1)} for step in range(n)]
    result: dict[str, Any] = {
        "count": n,
        "accuracy": sum(accuracy) / n,
        "nll": nll,
        "brier": brier,
        "ece": _ece(confidence, calibrated_correctness),
        "risk_coverage": risk_coverage,
    }
    if all(row["task_type"] == "score" for row in rows):
        result["ordinal_mae"] = sum(row["ordinal_mae"] for row in rows) / n
        result["rps"] = sum(row["rps"] for row in rows) / n
    if all(row["task_type"] == "noul" for row in rows):
        result["binary_brier"] = brier
    return result


def _batch_output(model: Any, encoder: Any, examples: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    batch = encoder.collate([encoder.encode(example["state"], example["question"]) for example in examples])
    with torch.inference_mode():
        return model(**{key: value.to(device) for key, value in batch.items()}).float()


def evaluate(model: Any, encoder: Any, dataset_path: str | Path, batch_size: int = 4,
             temperatures: dict[str, float] | None = None, device: str = "cpu") -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    examples = _read_jsonl(dataset_path)
    device_obj = torch.device(device)
    model.to(device_obj).eval()
    records: list[dict[str, Any]] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        logits = _batch_output(model, encoder, chunk, device_obj)
        for offset, (example, row_logits) in enumerate(zip(chunk, logits)):
            keys = list(example["keys"])
            count = len(keys)
            task = str(example["task_type"])
            temp = float((temperatures or {}).get(task, 1.0))
            if not math.isfinite(temp) or temp <= 0:
                raise ValueError(f"temperature for {task} must be positive and finite")
            valid_logits = row_logits[:count] / temp
            probs = F.softmax(valid_logits, dim=-1).cpu().tolist()
            targets = _target(example, keys, count)
            pred = max(range(count), key=lambda i: (probs[i], -i))
            target_index = max(range(count), key=lambda i: (targets[i], -i))
            entry: dict[str, Any] = {
                "id": example.get("id", example.get("example_id", str(start + offset))),
                "task_type": task,
                "keys": keys,
                "probabilities": probs,
                "prediction": keys[pred],
                "target": targets,
                "nll": -sum(t * math.log(max(p, 1e-12)) for p, t in zip(probs, targets)),
                "brier": sum((p - t) ** 2 for p, t in zip(probs, targets)),
                "prediction_index": pred,
                "target_index": target_index,
            }
            if task == "score":
                entry["expected_rank"] = sum(i * p for i, p in enumerate(probs))
                entry["ordinal_mae"] = abs(entry["expected_rank"] - sum(i * t for i, t in enumerate(targets)))
                entry["rps"] = sum((sum(probs[:i + 1]) - sum(targets[:i + 1])) ** 2 for i in range(count - 1)) / max(1, count - 1)
            if task == "noul":
                entry["probability_true"] = probs[-1]
            records.append(entry)
    by_task = {task: _metrics([row for row in records if row["task_type"] == task]) for task in sorted({r["task_type"] for r in records})}
    return {"count": len(records), "metrics": _metrics(records), "by_task": by_task, "examples": records}


def calibrate(model: Any, encoder: Any, dataset_path: str | Path, batch_size: int = 4,
              device: str = "cpu") -> dict[str, float]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    examples = _read_jsonl(dataset_path)
    device_obj = torch.device(device)
    model.to(device_obj).eval()
    collected: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        logits = _batch_output(model, encoder, chunk, device_obj).detach().cpu()
        for example, row_logits in zip(chunk, logits):
            keys = list(example["keys"])
            task = str(example["task_type"])
            collected.setdefault(task, []).append((row_logits[:len(keys)].clone(), torch.tensor(_target(example, keys, len(keys)))))
    temperatures: dict[str, float] = {}
    for task, rows in collected.items():
        logits = [row[0] for row in rows]
        targets = [row[1] for row in rows]
        log_temp = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        optimizer = torch.optim.LBFGS([log_temp], lr=0.2, max_iter=100, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temp = log_temp.exp().clamp(0.05, 20.0)
            loss = torch.stack([-(target * F.log_softmax(row.double() / temp, dim=-1)).sum() for row, target in zip(logits, targets)]).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        temperatures[task] = float(log_temp.detach().exp().clamp(0.05, 20.0))
    return temperatures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--calibrate")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="evaluation.json")
    args = parser.parse_args()
    from model import load_model

    model, encoder, temperatures = load_model(args.model, device=args.device)
    if args.calibrate:
        temperatures = calibrate(model, encoder, args.calibrate, args.batch_size, args.device)
        config_path = Path(args.model) / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["temperatures"] = temperatures
        config_path.write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")
    result = evaluate(model, encoder, args.data, args.batch_size, temperatures, args.device)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {task: {key: value for key, value in metrics.items() if key != "risk_coverage"} for task, metrics in {"all": result["metrics"], **result["by_task"]}.items()}
    print(json.dumps({"temperatures": temperatures, "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
