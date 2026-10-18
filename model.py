import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TASKS = ("choice", "noul", "score")


def _text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def candidates(question):
    if not isinstance(question, dict):
        raise ValueError("question must be an object")
    task, criteria = question.get("type"), question.get("criteria")
    if task == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("choice criteria must be a nonempty object")
        return [(str(key), str(key) if value is None else f"{key}: {_text(value)}") for key, value in criteria.items()]
    if task == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score criteria must list at least two levels")
        return [(str(index), _text(value)) for index, value in enumerate(criteria)]
    if task == "noul":
        criteria = {} if criteria is None else criteria
        if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
            raise ValueError("noul criteria may only define true and false")
        if question.get("instructions") in (None, "") and not criteria:
            raise ValueError("noul needs instructions or criteria")
        return [
            ("false", "No" if criteria.get("false") is None else f"No: {_text(criteria['false'])}"),
            ("true", "Yes" if criteria.get("true") is None else f"Yes: {_text(criteria['true'])}"),
        ]
    raise ValueError(f"unsupported question type: {task!r}")


def compute_dtype(device):
    device = torch.device(device)
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability(device) >= (8, 0) else torch.float16


def weight_dtype(device):
    return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32


class Encoder:
    def __init__(self, tokenizer, max_length=2048, max_choices=255):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_choices = max_choices
        self.newline = self._tokens("\n")
        if len(self.newline) != 1:
            raise ValueError("tokenizer must encode a newline as one token")

    def _tokens(self, text):
        return self.tokenizer(text, add_special_tokens=False, split_special_tokens=True)["input_ids"]

    def encode(self, state, question, order=None):
        options = candidates(question)
        if len(options) > self.max_choices:
            raise ValueError(f"question has {len(options)} options; limit is {self.max_choices}")
        if order is not None:
            options = [options[index] for index in order]
        instructions = question.get("instructions")
        head = f"{_text(state)}\n\n"
        if instructions not in (None, ""):
            head += f"Question: {_text(instructions)}\n"
        input_ids = self._tokens(head + "Options:\n")
        option_idx = []
        for number, (_, text) in enumerate(options, 1):
            input_ids += self._tokens(f"{number}. {text}") + self.newline
            option_idx.append(len(input_ids) - 1)
        input_ids += self._tokens("Answer:")
        if self.max_length and len(input_ids) > self.max_length:
            raise ValueError(f"encoded input has {len(input_ids)} tokens; limit is {self.max_length}")
        return {"input_ids": input_ids, "option_idx": option_idx, "keys": [key for key, _ in options], "task_type": question["type"]}

    def collate(self, examples):
        if not examples:
            raise ValueError("cannot collate an empty batch")
        length = max(len(item["input_ids"]) for item in examples)
        count = max(len(item["keys"]) for item in examples)
        input_ids = torch.full((len(examples), length), self.tokenizer.pad_token_id or 0, dtype=torch.long)
        attention_mask = torch.zeros((len(examples), length), dtype=torch.long)
        option_idx = torch.zeros((len(examples), count), dtype=torch.long)
        answer_idx = torch.zeros(len(examples), dtype=torch.long)
        choice_mask = torch.zeros((len(examples), count), dtype=torch.bool)
        for row, item in enumerate(examples):
            size = len(item["input_ids"])
            input_ids[row, :size] = torch.tensor(item["input_ids"])
            attention_mask[row, :size] = 1
            option_idx[row, :len(item["keys"])] = torch.tensor(item["option_idx"])
            answer_idx[row] = size - 1
            choice_mask[row, :len(item["keys"])] = True
        return {"input_ids": input_ids, "attention_mask": attention_mask, "option_idx": option_idx, "answer_idx": answer_idx, "choice_mask": choice_mask}


class DecisionModel(nn.Module):
    def __init__(self, backbone, hidden_size, head_width=256):
        super().__init__()
        self.backbone = backbone
        self.query = nn.Linear(hidden_size, head_width)
        self.key = nn.Linear(hidden_size, head_width)
        self.norm = nn.LayerNorm(hidden_size)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.head_width = head_width

    def forward(self, input_ids, attention_mask, option_idx, answer_idx, choice_mask):
        with torch.autocast(input_ids.device.type, dtype=compute_dtype(input_ids.device), enabled=input_ids.device.type == "cuda"):
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        with torch.autocast(hidden.device.type, enabled=False):
            query = F.normalize(self.query(self.norm(hidden[rows, answer_idx].float())), dim=-1)
            keys = F.normalize(self.key(self.norm(hidden[rows[:, None], option_idx].float())), dim=-1)
            logits = self.log_scale.clamp(max=math.log(50.0)).exp() * (keys @ query.unsqueeze(-1)).squeeze(-1)
        return logits.masked_fill(~choice_mask, -torch.inf)


def _backbone(base_model, revision, attention, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm = AutoModelForCausalLM.from_pretrained(base_model, revision=revision, attn_implementation=attention, dtype=dtype)
    return lm.model, tokenizer


def create_model(base_model=DEFAULT_MODEL, revision=MODEL_REVISION, lora_rank=16, lora_layers=0, head_width=256, attention="sdpa", gradient_checkpointing=True, device="cpu"):
    from peft import LoraConfig, get_peft_model

    backbone, tokenizer = _backbone(base_model, revision, attention, weight_dtype(device))
    if gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    count = backbone.config.num_hidden_layers
    layers = list(range(count - lora_layers, count)) if 0 < lora_layers < count else None
    linear = sorted({name.rsplit(".", 1)[-1] for name, module in backbone.named_modules() if isinstance(module, nn.Linear)})
    config = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05, target_modules=linear, layers_to_transform=layers, layers_pattern="layers" if layers else None, bias="none")
    model = DecisionModel(get_peft_model(backbone, config), backbone.config.hidden_size, head_width).to(device)
    model.export_config = {"base_model": base_model, "revision": revision, "attention": attention, "head_width": head_width}
    return model, tokenizer


def save_model(model, tokenizer, directory, encoder_config, temperatures=None):
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(directory / "backbone", safe_serialization=True)
    tokenizer.save_pretrained(directory / "tokenizer")
    head = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items() if not key.startswith("backbone.")}
    save_file(head, directory / "head.safetensors")
    config = {**model.export_config, "encoder": encoder_config, "temperatures": temperatures or {}}
    (directory / "config.json").write_text(json.dumps(config, sort_keys=True, indent=2))


def load_model(directory, device="cpu", max_length=None):
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    base, _ = _backbone(config["base_model"], config["revision"], config["attention"], weight_dtype(device))
    backbone = PeftModel.from_pretrained(base, directory / "backbone").merge_and_unload()
    model = DecisionModel(backbone, backbone.config.hidden_size, config["head_width"])
    missing = model.load_state_dict(load_file(directory / "head.safetensors"), strict=False).missing_keys
    if any(not key.startswith("backbone.") for key in missing):
        raise ValueError(f"head checkpoint is missing {missing}")
    model.to(device).eval()
    encoder_config = {**config["encoder"], **({"max_length": max_length} if max_length is not None else {})}
    encoder = Encoder(AutoTokenizer.from_pretrained(directory / "tokenizer"), **encoder_config)
    return model, encoder, config.get("temperatures", {})


def soft_cross_entropy(logits, targets, choice_mask):
    if logits.shape != targets.shape or logits.shape != choice_mask.shape:
        raise ValueError("logits, targets, and choice_mask must have identical shapes")
    if not torch.isfinite(targets).all() or (targets < 0).any() or (targets.masked_select(~choice_mask) != 0).any():
        raise ValueError("targets must be finite, nonnegative, and zero on padding")
    sums = targets.sum(dim=1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=1e-5, rtol=0):
        raise ValueError("each target distribution must sum to one")
    log_probs = F.log_softmax(logits.float(), dim=-1).masked_fill(~choice_mask, 0)
    return -(targets.float() * log_probs).sum(dim=-1).mean()
