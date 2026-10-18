import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from data import DecisionDataset, prepare_data, prepare_test
from model import Encoder, compute_dtype, create_model, save_model, soft_cross_entropy


class LengthBucketSampler:
    def __init__(self, dataset, batch_size, seed, epoch=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch

    def batches(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.dataset)))
        rng.shuffle(indices)
        window = self.batch_size * 32
        batches = []
        for start in range(0, len(indices), window):
            bucket = sorted(indices[start:start + window], key=lambda index: self.dataset[index].get("length", 0))
            batches.extend(bucket[offset:offset + self.batch_size] for offset in range(0, len(bucket), self.batch_size))
        rng.shuffle(batches)
        return batches


def _atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _batch(dataset, indices, encoder, device, rng=None):
    rows = [dataset[index] for index in indices]
    return {key: value.to(device) for key, value in DecisionDataset.collate(rows, encoder, rng).items()}


def _sum(values, device):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor.tolist()


def _evaluate(model, dataset, encoder, batch_size, device, limit, rank, world):
    indices = list(range(min(len(dataset), limit or len(dataset))))[rank::world]
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            batch = _batch(dataset, chunk, encoder, device)
            logits = model(**{key: value for key, value in batch.items() if key != "targets"})
            loss_sum += float(soft_cross_entropy(logits.float(), batch["targets"], batch["choice_mask"])) * len(chunk)
            correct += int((logits.argmax(-1) == batch["targets"].argmax(-1)).sum())
            total += len(chunk)
    model.train()
    loss_sum, correct, total = _sum([loss_sum, correct, total], device)
    if not total:
        return {"loss": None, "accuracy": None, "examples": 0}
    return {"loss": loss_sum / total, "accuracy": correct / total, "examples": int(total)}


def _clock(timing, name, device, enabled, since=None):
    if not enabled:
        return None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    now = time.monotonic()
    if since is not None:
        timing[name] += now - since
    return now


def _write_log(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def _encoder_config(config):
    return {"max_length": config["max_length"], "max_choices": config.get("max_choices", 255)}


def fit(config):
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    main = rank == 0
    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = create_model(
        base_model=config["base_model"], revision=config["revision"], lora_rank=config["lora_rank"], lora_layers=config.get("lora_layers", 0), head_width=config["head_width"],
        attention=config["attention"], gradient_checkpointing=config.get("gradient_checkpointing", True), device=device,
    )
    encoder = Encoder(tokenizer, **_encoder_config(config))
    data_dir = Path(config["data_dir"])
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        if world > 1:
            raise FileNotFoundError(f"{train_path} is missing; run train.py --prepare before a multi-GPU run")
        prepare_data(data_dir, tokenizer, config["train_size"], config["max_length"], seed, config.get("source_data_dir"), config.get("max_choices", 255), config.get("intent_size", 0))
    train_data = DecisionDataset(train_path)
    validation = DecisionDataset(data_dir / "validation.jsonl")
    if not len(train_data):
        raise ValueError("training split is empty")
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index]) if world > 1 else model
    rank_batches = len(train_data) // (config["batch_size"] * world)
    total_steps = int(config.get("max_steps", 0)) or math.ceil(rank_batches / config["grad_accum"]) * int(config["epochs"])
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in trainable.items() if n.startswith("backbone.")], "lr": config["learning_rate"]},
        {"params": [p for n, p in trainable.items() if not n.startswith("backbone.")], "lr": config["head_learning_rate"]},
    ], weight_decay=config["weight_decay"])
    warmup = max(1, int(total_steps * 0.03))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step + 1) / warmup, 1.0) * 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=compute_dtype(device) == torch.float16)
    checkpoint_path = output_dir / "trainer.pt"
    step, epoch, batch_offset, best_loss = 0, 0, 0, math.inf
    if config.get("resume") and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        step, epoch, batch_offset, best_loss = state["step"], state["epoch"], state["batch_offset"], state["best_loss"]
        random.setstate(state["python_rng"])

    log_path = output_dir / "train.jsonl"
    start_time = time.monotonic()
    tokens_seen = 0
    timing = dict.fromkeys(("data", "compute", "optimizer"), 0.0)
    stop = False
    model.train()
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps and epoch < config["epochs"] and not stop:
        order = LengthBucketSampler(train_data, config["batch_size"], seed, epoch).batches()[rank::world][:rank_batches]
        for batch_index in range(batch_offset, len(order)):
            clock = _clock(timing, "data", device, config.get("profile"))
            batch = _batch(train_data, order[batch_index], encoder, device, random)
            clock = _clock(timing, "data", device, config.get("profile"), clock)
            consumed = batch_index + 1
            boundary = consumed % config["grad_accum"] == 0 or consumed == len(order)
            accumulation_size = min(config["grad_accum"], len(order) - (batch_index - batch_index % config["grad_accum"]))
            with net.no_sync() if world > 1 and not boundary else nullcontext():
                logits = net(**{key: value for key, value in batch.items() if key != "targets"})
                loss = soft_cross_entropy(logits.float(), batch["targets"], batch["choice_mask"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at optimizer step {step}")
                scaler.scale(loss / accumulation_size).backward()
            clock = _clock(timing, "compute", device, config.get("profile"), clock)
            tokens_seen += int(batch["attention_mask"].sum())
            batch_offset = consumed
            if not boundary:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            _clock(timing, "optimizer", device, config.get("profile"), clock)
            step += 1
            elapsed = time.monotonic() - start_time
            if main:
                _write_log(log_path, {"step": step, "epoch": epoch, "train_loss": float(loss.detach()), "lr": [group["lr"] for group in optimizer.param_groups], "tokens_per_second": tokens_seen * world / max(elapsed, 1e-9), "elapsed_seconds": elapsed, **({"timing": timing} if config.get("profile") else {})})
            if step % config["eval_every"] == 0 or step == total_steps:
                result = _evaluate(model, validation, encoder, config["batch_size"], device, config.get("eval_examples"), rank, world)
                if main:
                    _write_log(log_path, {"step": step, "validation": result, "elapsed_seconds": elapsed})
                    if result["loss"] is not None and result["loss"] < best_loss:
                        save_model(model, tokenizer, output_dir / "best", _encoder_config(config))
                if result["loss"] is not None:
                    best_loss = min(best_loss, result["loss"])
            if main and config.get("profile") and device.type == "cuda" and step == int(config.get("profile_steps", 50)):
                profile = {"step": step, "world_size": world, "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device), "tokens_per_second": tokens_seen * world / max(elapsed, 1e-9), "seconds_per_step": elapsed / step, "projected_hours_for_run": elapsed / step * (total_steps - step) / 3600}
                (output_dir / "profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
            limit = config.get("max_hours", 0) * 3600 - config.get("stop_margin_minutes", 10) * 60
            due = bool(_sum([float(config.get("max_hours", 0) and elapsed >= limit)], device)[0])
            if main and (step % config["checkpoint_every"] == 0 or due or step >= total_steps):
                checkpoint = {"model": {name: parameter.detach() for name, parameter in trainable.items()}, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "step": step, "epoch": epoch, "batch_offset": batch_offset, "best_loss": best_loss, "python_rng": random.getstate(), "config": config}
                _atomic_save(checkpoint, checkpoint_path)
                save_model(model, tokenizer, output_dir / "latest", _encoder_config(config))
            if due:
                if main:
                    _write_log(log_path, {"step": step, "event": "graceful_stop_session_limit"})
                stop = True
                break
            if step >= total_steps:
                break
        if not stop and batch_offset >= len(order):
            epoch, batch_offset = epoch + 1, 0
    if main:
        _write_log(log_path, {"step": step, "event": "training_complete" if step >= total_steps else "training_stopped", "best_validation_loss": best_loss})
    if dist.is_initialized():
        dist.destroy_process_group()
    return {"step": step, "epoch": epoch, "best_validation_loss": best_loss, "output_dir": str(output_dir)}


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.smoke:
        config.update(load_config(Path(args.config).with_name("smoke.json")))
    config["resume"] = args.resume
    if args.prepare or args.prepare_test:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision"])
        if args.prepare:
            print(json.dumps(prepare_data(config["data_dir"], tokenizer, config["train_size"], config["max_length"], config["seed"], config.get("source_data_dir"), config.get("max_choices", 255), config.get("intent_size", 0)), indent=2))
        if args.prepare_test:
            print(json.dumps(prepare_test(config["data_dir"], tokenizer, config["max_length"], config.get("source_data_dir"), config.get("max_choices", 255)), indent=2))
    else:
        result = fit(config)
        if int(os.environ.get("RANK", 0)) == 0:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
