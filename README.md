# REV decision model

This project fine-tunes Qwen3-4B-Instruct-2507 into a decision model for the Jev Decision Index. A request has a `state` and named `questions` of type `choice`, `noul`, or `score`, each with optional `instructions` and `criteria`. The BEV dataset and the Decision Index use the same question schema.

The encoder writes the state, the instructions, and one numbered line per option, then ends with `Answer:`. The model reads two kinds of hidden state from one forward pass. It reads one at the newline that ends each option line, and one at the last token. A pointer head normalizes both, projects the last token to a query and each option to a key, and scores each option by the cosine similarity of query and key times a learned scale capped at 50. A softmax over the valid options gives the distribution. The head has no fixed option count, so one forward pass covers the 151 options of CLINC150 the same way it covers two. `noul` uses two options, `No` and `Yes`. `score` lists its levels in order and reports the expected level.

Training fits LoRA adapters on every linear layer of the top `lora_layers` decoder layers, and trains the pointer head at a higher learning rate. Backpropagation stops at the lowest adapted layer, so training only half the layers saves about a third of the compute. Training shuffles the option order of `choice` questions on every pass, to stop the model from learning position bias.

Training mixes BEV with the train splits of BANKING77 and CLINC150 (`intent_size` examples in total). Each intent example shows 8, 16, 32, or 64 sampled intents including the gold one, or the full list when it fits. BEV questions have at most about ten options, and these examples teach the head to pick among many. The test splits of both datasets are never read, because the Decision Index scores them.

The weights load in bf16. The model runs its forward pass under autocast, in bf16 on Ampere or newer GPUs and in fp16 on older ones such as the T4. A T4 has no bf16 hardware, and bf16 matrix math on it is about five times slower.

The encoder rejects training rows longer than `max_length` tokens. The demo server accepts any length by default, because the Decision Index counts a rejected request as wrong.

## Environment

Use Python 3.11 or later. Install `requirements.txt` in a local environment with a suitable PyTorch build already installed:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.6'
python -m pip install -r requirements.txt
```

PyTorch wheels vary by operating system and CUDA version, so install the build that matches the machine before installing this project's pinned model dependencies. The Kaggle runner installs the pinned packages with a pip constraint that holds Kaggle's preinstalled PyTorch version.

## Prepare and train

Put the BEV Parquet files under `source_data_dir`, keeping the dataset's layout (`data/train.parquet`, `hard_50k/train.parquet`, and so on). If a file is missing, `prepare_data` downloads it from the pinned dataset revision into the Hugging Face cache. Review `configs/pilot.json`, especially its data path and output directory. Prepare the partitions, run a short smoke pass, then train:

```sh
python train.py --config configs/pilot.json --prepare
python train.py --config configs/pilot.json --smoke
python train.py --config configs/pilot.json
python train.py --config configs/pilot.json --resume
```

To train on several GPUs, prepare the data first, then start the trainer with `torchrun`:

```sh
torchrun --standalone --nproc_per_node=2 train.py --config configs/pilot.json --resume
```

Training writes resumable state to `output_dir/trainer.pt`. It exports the best validation checkpoint to `output_dir/best` and the most recent one to `output_dir/latest`. Keep the calibration split apart from training and use it only for temperature fitting.

## Evaluate and calibrate

Evaluation reports accuracy, soft-target NLL and Brier score, top-label ECE, risk/coverage, and task-specific ordinal metrics. Temperature fitting uses only the calibration JSONL passed to `--calibrate`.

```sh
python evaluate.py --model runs/pilot/best --data data/pilot/validation.jsonl --calibrate data/pilot/calibration.jsonl --output runs/pilot/calibrated-evaluation.json
python train.py --config configs/pilot.json --prepare-test
python evaluate.py --model runs/pilot/best --data data/pilot/test.jsonl --output runs/pilot/test-evaluation.json
```

The first command saves task temperatures into `config.json` in the model directory. Run `--prepare-test` only for the final comparison, because it reads the sealed BEV test split. Re-evaluate the exported checkpoint after calibration. Do not fit temperatures on the sealed test split.

## Local demo

The demo requires an exported checkpoint. It never substitutes generated or placeholder probabilities for a missing model.

```sh
python demo.py --model runs/pilot/best --device cuda
```

Open <http://127.0.0.1:8000> to edit and send a request. The server accepts the same JSON at `POST /v1/decisions` and at `POST /v1/systemone`, the endpoint that the Decision Index HTTP engine calls:

```sh
python -m decision_index run --engine http --option base_url=http://127.0.0.1:8000 --option model=rev --rows sample-100.jsonl.gz --out runs/rev-sample
```

A `noul` answer has `noul`, the probability of yes. A `choice` answer has `choice`, `confidence`, and `probabilities`. A `score` answer has `score`, `confidence`, `legend`, and `probabilities`. `confidence` is the gap between the top two probabilities. Ties select the lexicographically smallest key. Pass `--max-length` to reject longer inputs.

## Package a Kaggle run

Install Kaggle's CLI and authenticate it separately. Select the T4 x2 accelerator for the notebook. Start with a smoke package, which trains 20 steps and writes `profile.json` with memory use, tokens per second, and seconds per step:

```sh
python scripts/kaggle.py --config configs/pilot.json --slug OWNER/rev-decision-model --private --dataset-source OWNER/DATASET --smoke
kaggle kernels push -p kaggle/submission
```

Drop `--smoke` to package the full run. Pass `--set '{"key": value}'` to override config values for one package, for example `{"gpus": 1}` to use one GPU or `{"env": {"NAME": "value"}}` to set environment variables for the trainer. Omit `--dataset-source` if the run downloads the pinned BEV files from the Hub. The command writes `kaggle/submission/runner.py` and `kernel-metadata.json`, and it does not upload or start a job.

The entry script installs the pinned packages, prepares the data under `/kaggle/working/output/data`, and starts one training process per GPU. It writes logs and checkpoints under `/kaggle/working/output` and packs everything except the data into `/kaggle/working/rev-output.zip`. Training stops 20 minutes before `max_hours` and saves a checkpoint. Training also saves a checkpoint every 100 optimizer steps, so a hard stop loses at most that much work. To continue in a new session, package with `--resume-from OWNER/PREVIOUS-NOTEBOOK`, which attaches that notebook's output as an input. The runner finds `output/trainer.pt` under `/kaggle/input`, copies that output directory, and resumes from the saved step.

## Dependencies and license notes

The project pins Transformers 5.19.0 with PEFT 0.21.2, tokenizers 0.23.2, safetensors 0.8.0, Hugging Face Hub 1.33.0, and Accelerate 1.15.0. The pointer and LoRA setup also loads Qwen3.5 models, which need Transformers 5. PyTorch is intentionally not pinned in `requirements.txt` because its correct wheel depends on the host. Kaggle uses its existing PyTorch installation.

Review source dataset terms and model distribution rights before publishing artifacts. See [research.md](research.md) for architecture assumptions, dataset limitations, and proposed evaluation protocol.
