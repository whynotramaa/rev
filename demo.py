from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch
import torch.nn.functional as F


MAX_BODY = 256_000
MAX_QUESTIONS = 128


def answer(question: dict[str, Any], keys: list[str], probabilities: list[float]) -> dict[str, Any]:
    task = question["type"]
    if task == "noul":
        return {"type": "noul", "noul": probabilities[keys.index("true")]}
    ranked = sorted(probabilities, reverse=True)
    confidence = ranked[0] - ranked[1] if len(ranked) > 1 else 1.0
    by_key = dict(zip(keys, probabilities))
    if task == "score":
        return {
            "type": "score",
            "score": sum(int(key) * probability for key, probability in by_key.items()),
            "confidence": confidence,
            "legend": {str(index): level for index, level in enumerate(question["criteria"])},
            "probabilities": {str(index): by_key[str(index)] for index in range(len(keys))},
        }
    top = min((key for key in keys if by_key[key] == ranked[0]))
    return {"type": "choice", "choice": top, "confidence": confidence, "probabilities": {key: by_key[key] for key in question["criteria"]}}


def make_decider(checkpoint: str, device: str = "cpu", max_length: int = 0, chunk: int = 8):
    from model import load_model

    model, encoder, temperatures = load_model(checkpoint, device=device, max_length=max_length)
    device_obj = torch.device(device)

    def decide(request: Any) -> dict[str, Any]:
        if not isinstance(request, dict) or "state" not in request or not isinstance(request.get("questions"), dict):
            raise ValueError("request needs state and a questions object")
        questions = request["questions"]
        if not 1 <= len(questions) <= MAX_QUESTIONS:
            raise ValueError(f"questions must contain 1 to {MAX_QUESTIONS} entries")
        encoded = {name: encoder.encode(request["state"], question) for name, question in questions.items()}
        items = list(encoded.values())
        logits = []
        with torch.inference_mode():
            for start in range(0, len(items), chunk):
                batch = encoder.collate(items[start:start + chunk])
                logits.extend(model(**{key: value.to(device_obj) for key, value in batch.items()}).float().cpu())
        answers, input_tokens = {}, 0
        for (name, item), row in zip(encoded.items(), logits):
            temperature = float(temperatures.get(item["task_type"], 1.0))
            probabilities = F.softmax(row[:len(item["keys"])] / temperature, dim=-1).tolist()
            answers[name] = answer(questions[name], item["keys"], probabilities)
            input_tokens += len(item["input_ids"])
        return {"model": request.get("model", Path(checkpoint).name), "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}

    return decide, encoder


def serve(checkpoint: str, host: str = "127.0.0.1", port: int = 8000, device: str = "cpu", max_length: int = 0) -> None:
    decide, encoder = make_decider(checkpoint, device, max_length)
    html = Path(__file__).with_name("demo.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlparse(self.path).path not in {"/", "/index.html"}:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html)

        def do_POST(self) -> None:
            if urlparse(self.path).path not in {"/v1/decisions", "/v1/systemone"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError(f"request body must be between 1 and {MAX_BODY} bytes")
                body = json.dumps(decide(json.loads(self.rfile.read(length))), allow_nan=False).encode()
                self.send_response(200)
            except (ValueError, json.JSONDecodeError) as exc:
                body = json.dumps({"error": {"message": str(exc)}}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(format % args)

    print(f"Model max length: {encoder.max_length or 'unlimited'}; max choices: {encoder.max_choices}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local checkpoint-backed decision demo")
    parser.add_argument("--model", required=True, help="exported model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=0, help="token limit per question; 0 accepts any length")
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        parser.error("--model must point to an exported checkpoint directory")
    serve(args.model, args.host, args.port, args.device, args.max_length)


if __name__ == "__main__":
    main()
