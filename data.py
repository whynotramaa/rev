import hashlib
import heapq
import json
import math
import random
import re
from pathlib import Path

import torch

from model import Encoder

DATASET_ID = "avbiswas/bev-decision"
DATASET_REVISION = "11ed9b5b879e7f9524e7c89c1037d498c54206f0"
DATASET_DIRS = {"default": "data", "hard_50k": "hard_50k", "numeric_temporal": "numeric_temporal", "skills": "skills"}
SOURCE_WEIGHTS = {"default": 0.55, "hard_50k": 0.20, "skills": 0.15, "numeric_temporal": 0.05}
PARTITIONS = (("train", 0.90), ("validation", 0.05), ("calibration", 0.025), ("policy", 0.025))
INTENT_SOURCES = {
    "banking77": ("mteb/banking77", "18072d2685ea682290f7b8924d94c62acc19c0b2", "data/train-00000-of-00001.parquet", "Which banking intent does this customer message express?"),
    "clinc150": ("clinc/clinc_oos", "155b9c710419136e17307b80d0a13e68cd46b4ec", "plus/train-00000-of-00001.parquet", "Which intent does this user request express? Choose oos if no listed intent applies."),
}
OPTION_SAMPLES = (8, 16, 32, 64, None)


def normalize_state(state):
    return re.sub(r"\s+", " ", state.casefold()).strip()


def _digest(*parts):
    payload = "\0".join(str(part) for part in parts).encode()
    return hashlib.sha256(payload).hexdigest()


def _partition(state, seed):
    value = int(_digest(seed, normalize_state(state))[:16], 16) / 2**64
    boundary = 0.0
    for name, fraction in PARTITIONS:
        boundary += fraction
        if value < boundary:
            return name
    return "policy"


def _question_target(question, keys):
    if not isinstance(question, dict):
        raise ValueError("question must be an object")
    kind = question.get("type")
    label_probs = question.get("label_probs")
    if label_probs is not None:
        if isinstance(label_probs, dict):
            if set(label_probs) != set(keys):
                raise ValueError("soft target keys must exactly match candidate keys")
            values = [label_probs[key] for key in keys]
        elif isinstance(label_probs, list):
            if len(label_probs) != len(keys):
                raise ValueError("soft target length must match candidate count")
            values = label_probs
        else:
            raise ValueError("soft target must be a mapping or list")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("soft target values must be finite nonnegative numbers")
        if abs(sum(values) - 1.0) > 2e-6:
            raise ValueError("soft target values must sum to one")
        total = sum(values)
        return [float(value) / total for value in values]

    label = question.get("label")
    if kind == "noul":
        if type(label) is not bool:
            raise ValueError("noul hard label must be a boolean")
        key = "true" if label else "false"
    elif kind == "choice":
        if not isinstance(label, str):
            raise ValueError("choice hard label must be a candidate key")
        key = label
    else:
        if isinstance(label, bool) or not isinstance(label, (int, str)):
            raise ValueError("score hard label must be an integer level")
        try:
            key = str(int(label))
        except (ValueError, TypeError):
            raise ValueError("score hard label must be an integer level") from None
    if key not in keys:
        raise ValueError(f"hard label {key!r} is not a candidate")
    return [float(key == candidate) for candidate in keys]


def _read_parquet(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Preparing BEV data requires pyarrow") from exc
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        yield from batch.to_pylist()


def _source_file(data_dir, config, split):
    name = f"{DATASET_DIRS[config]}/{split}.parquet"
    found = sorted(Path(data_dir).rglob(name)) if Path(data_dir).exists() else []
    if found:
        return found[0]
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(DATASET_ID, name, repo_type="dataset", revision=DATASET_REVISION))


def _decisions(path):
    for row in _read_parquet(path):
        state = row.get("state")
        try:
            questions = json.loads(row["questions_json"])
        except (TypeError, KeyError, json.JSONDecodeError):
            yield None, None, None
            continue
        if not isinstance(state, str) or not isinstance(questions, dict):
            yield None, None, None
            continue
        for question_id, question in questions.items():
            yield state, question_id, question


def _intent_decisions(name, seed):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo, revision, filename, instructions = INTENT_SOURCES[name]
    table = pq.read_table(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    texts = table["text"].to_pylist()
    if "label_text" in table.column_names:
        labels = table["label_text"].to_pylist()
    else:
        names = json.loads(table.schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
        labels = [names[index] for index in table["intent"].to_pylist()]
    intents = sorted(set(labels))
    rng = random.Random(f"{seed}:{name}")
    order = list(range(len(texts)))
    rng.shuffle(order)
    for index in order:
        label, size = labels[index], rng.choice(OPTION_SAMPLES)
        options = intents if size is None or size >= len(intents) else rng.sample([intent for intent in intents if intent != label], size - 1) + [label]
        rng.shuffle(options)
        yield texts[index], f"{name}-{index}", {"type": "choice", "instructions": instructions, "criteria": dict.fromkeys(options), "label": label}


def _row(encoder, state, question_id, question, source):
    encoded = encoder.encode(state, question)
    group = normalize_state(state)
    return {
        "state": state,
        "question": question,
        "question_id": question_id,
        "keys": encoded["keys"],
        "task_type": encoded["task_type"],
        "targets": _question_target(question, encoded["keys"]),
        "source": source,
        "group_id": _digest(group),
        "fingerprint": _digest(group, json.dumps(question, sort_keys=True, ensure_ascii=False, separators=(",", ":"))),
        "length": len(encoded["input_ids"]),
    }


def _write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda row: row["fingerprint"]):
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _allocate(total):
    weights = {key: value / sum(SOURCE_WEIGHTS.values()) for key, value in SOURCE_WEIGHTS.items()}
    quotas = {key: int(total * value) for key, value in weights.items()}
    for key in list(weights)[: total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def _exclude(exclusions, config, reason):
    exclusions.setdefault(config, {}).setdefault(reason, 0)
    exclusions[config][reason] += 1


def prepare_data(output_dir, tokenizer, train_size=20000, max_length=2048, seed=42, data_dir=None, max_choices=255, intent_size=0):
    if train_size < 1:
        raise ValueError("train_size must be positive")
    data_dir = Path(data_dir or "/tmp/rev-research")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = Encoder(tokenizer, max_length=max_length, max_choices=max_choices)
    quotas = _allocate(train_size)
    exclusions = {}
    counts = {}
    fingerprints = set()
    rows = {partition: [] for partition, _ in PARTITIONS}

    for config in SOURCE_WEIGHTS:
        sizes = {"train": int(quotas[config] * 1.5) + 1, "validation": max(128, int(train_size * 0.05)), "calibration": max(64, int(train_size * 0.025)), "policy": max(64, int(train_size * 0.025))}
        pools = {partition: [] for partition in sizes}
        seen = 0
        for state, question_id, question in _decisions(_source_file(data_dir, config, "train")):
            seen += 1
            if state is None:
                _exclude(exclusions, config, "malformed_row")
                continue
            fingerprint = _digest(normalize_state(state), json.dumps(question, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
            if fingerprint in fingerprints:
                _exclude(exclusions, config, "duplicate_decision")
                continue
            fingerprints.add(fingerprint)
            partition = _partition(state, seed)
            item = (-int(_digest(seed, fingerprint)[:16], 16), fingerprint, (state, question_id, question))
            pool = pools[partition]
            if len(pool) < sizes[partition]:
                heapq.heappush(pool, item)
            elif item > pool[0]:
                heapq.heapreplace(pool, item)
        selected = 0
        for partition, pool in pools.items():
            for _, _, (state, question_id, question) in sorted(pool, reverse=True):
                if partition == "train" and selected >= quotas[config]:
                    break
                try:
                    row = _row(encoder, state, question_id, question, config)
                except (ValueError, KeyError, TypeError) as exc:
                    _exclude(exclusions, config, "over_max_length" if "tokens; limit" in str(exc) else "invalid_label_or_question")
                    continue
                rows[partition].append(row)
                selected += partition == "train"
        counts[config] = {"seen": seen, "selected_train": selected}

    for name in INTENT_SOURCES:
        selected = 0
        for state, question_id, question in _intent_decisions(name, seed):
            if selected >= intent_size // len(INTENT_SOURCES):
                break
            try:
                rows["train"].append(_row(encoder, state, question_id, question, name))
            except ValueError:
                _exclude(exclusions, name, "over_max_length")
                continue
            selected += 1
        counts[name] = {"selected_train": selected}

    paths = {}
    for partition, partition_rows in rows.items():
        paths[partition] = str(output_dir / f"{partition}.jsonl")
        _write_jsonl(paths[partition], partition_rows)
    manifest = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "seed": seed,
        "requested_train_size": train_size,
        "selected_train_size": len(rows["train"]),
        "max_length": max_length,
        "source_quotas": quotas,
        "source_counts": counts,
        "excluded_configs": {"counterfactual_15k": "pair ids missing; reserved until grouping is trustworthy"},
        "exclusions": exclusions,
        "split_counts": {partition: len(partition_rows) for partition, partition_rows in rows.items()},
        "paths": paths,
        "group_split": "sha256(seed, normalized_state), deterministic train/validation/calibration/policy allocation",
        "test_loaded": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def prepare_test(output_dir, tokenizer, max_length=2048, data_dir=None, max_choices=255):
    encoder = Encoder(tokenizer, max_length=max_length, max_choices=max_choices)
    rows, exclusions = [], {}
    for config in SOURCE_WEIGHTS:
        for state, question_id, question in _decisions(_source_file(data_dir or "/tmp/rev-research", config, "test")):
            try:
                rows.append(_row(encoder, state, question_id, question, config))
            except (ValueError, KeyError, TypeError, AttributeError):
                _exclude(exclusions, config, "excluded")
    path = Path(output_dir) / "test.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(path, rows)
    return {"path": str(path), "count": len(rows), "exclusions": exclusions}


class DecisionDataset:
    def __init__(self, path):
        with Path(path).open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    @staticmethod
    def collate(rows, encoder, rng=None):
        encoded, targets = [], []
        for row in rows:
            order = None
            if rng is not None and row["task_type"] == "choice":
                order = list(range(len(row["keys"])))
                rng.shuffle(order)
            encoded.append(encoder.encode(row["state"], row["question"], order))
            targets.append([row["targets"][index] for index in order] if order else row["targets"])
        batch = encoder.collate(encoded)
        target_tensor = torch.zeros(batch["choice_mask"].shape, dtype=torch.float32)
        for index, target in enumerate(targets):
            target_tensor[index, :len(target)] = torch.tensor(target, dtype=torch.float32)
        batch["targets"] = target_tensor
        return batch
