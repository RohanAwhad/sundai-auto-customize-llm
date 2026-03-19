"""
Mid-training script: knowledge ingestion via LoRA fine-tuning.

Teaches Qwen3-4B new factual Q&A knowledge while preserving general
capabilities via Tulu SFT replay data (70/30 mix).
"""

import json
import random
from pathlib import Path

import torch

torch.cuda.empty_cache()

from dotenv import load_dotenv

load_dotenv()

# ── Constants ────────────────────────────────────────────────────────────────

DATA_PATH = "./data/qa_slim.jsonl"
OUTPUT_DIR = Path("./knowledge-ingestion-test")

VAL_SIZE = 100
TEST_SIZE = 100
SEED = 42

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LORA_R = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1.5e-4
MICRO_BATCH_SIZE = 16
MAX_SEQ_LEN = 1024

WANDB_PROJECT = "sunday-auto-customize"
WANDB_ENTITY = "ronny21"
WANDB_RUN_NAME = "knowledge-ingestion-test"


# ── Helpers ──────────────────────────────────────────────────────────────────


def convert_to_messages(example: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


# ── 1. Load dataset ─────────────────────────────────────────────────────────

with open(DATA_PATH, "r") as f:
    dataset = [json.loads(line) for line in f]

print(f"Dataset size: {len(dataset)} examples")
print(f"\nDataset columns: {dataset[0].keys()}")
print("\n" + "=" * 60)
print("Sample entry:")
print("=" * 60)
sample = dataset[0]
print(f"\nQuestion: {sample['question']}")
print(f"\nAnswer: {sample['answer']}")

# ── 2. Convert format ───────────────────────────────────────────────────────

sample_converted = convert_to_messages(dataset[0])
print("\nConverted format:")
print(json.dumps(sample_converted, indent=2))

# ── 3. Split ────────────────────────────────────────────────────────────────

random.seed(SEED)
random.shuffle(dataset)

TRAIN_SIZE = len(dataset) - VAL_SIZE - TEST_SIZE
train_dataset = dataset[:TRAIN_SIZE]
val_dataset = dataset[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
test_dataset = dataset[TRAIN_SIZE + VAL_SIZE :]

train_data = [convert_to_messages(ex) for ex in train_dataset]
val_data = [convert_to_messages(ex) for ex in val_dataset]
test_data = [convert_to_messages(ex) for ex in test_dataset]

print(f"\nTraining examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")
print(f"Test examples: {len(test_data)}")

print(f"\ntrain_data[0]: {train_data[0]}")

# ── 4. Tulu mix-in ──────────────────────────────────────────────────────────

TULU_MIX_SIZE = int(3 / 7 * TRAIN_SIZE)  # 30% tulu, 70% new-knowledge

from datasets import load_dataset

tulu_dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
tulu_dataset = tulu_dataset.shuffle(seed=SEED)
for i, x in enumerate(tulu_dataset):
    if i >= TULU_MIX_SIZE:
        break
    train_data.append({"messages": x["messages"]})

print(f"\nTraining data size: {len(train_data)}")

# ── 5. Save JSONL ───────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_file = OUTPUT_DIR / "train.jsonl"
val_file = OUTPUT_DIR / "val.jsonl"
test_file = OUTPUT_DIR / "test.jsonl"

for path, data in [
    (train_file, train_data),
    (val_file, val_data),
    (test_file, test_data),
]:
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f"Data saved to {path} ({len(data)} examples)")

# ── 6. Training config ──────────────────────────────────────────────────────

train_kwargs = {
    "model_path": MODEL_NAME,
    "data_path": str(train_file),
    "ckpt_output_dir": OUTPUT_DIR / "model",
    "dataset_type": "chat_template",
    "field_messages": "messages",
    "device_map": "auto",
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "micro_batch_size": MICRO_BATCH_SIZE,
    "max_seq_len": MAX_SEQ_LEN,
    "seed": SEED,
    "lora_r": LORA_R,
    "lora_alpha": 2 * LORA_R,
    "lora_dropout": 0.1,
    "load_in_4bit": False,
    "bf16": True,
    "sample_packing": True,
    "logging_steps": 10,
    "eval_steps": 10,
    "save_steps": 10,
    "save_total_limit": 1,
    "wandb_project": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
    "wandb_run_name": WANDB_RUN_NAME,
}

# ── 7. Load model + LoRA ────────────────────────────────────────────────────

from training_hub.algorithms.lora import UnslothLoRABackend, JSONLLoggingCallback

backend = UnslothLoRABackend()
model, tokenizer = backend._load_unsloth_model(train_kwargs)
model = backend._apply_lora_config(model, train_kwargs)

# ── 8. Prepare datasets ─────────────────────────────────────────────────────

train_hf_dataset = backend._prepare_dataset(train_kwargs, tokenizer)
print(f"\nTrain HF dataset: {train_hf_dataset}")
print(f"train_hf_dataset[0]: {train_hf_dataset[0]}")

validation_kwargs = dict(train_kwargs)
validation_kwargs["data_path"] = str(val_file)
validation_hf_dataset = backend._prepare_dataset(validation_kwargs, tokenizer)

# ── 9. Train ─────────────────────────────────────────────────────────────────

from trl import SFTTrainer

training_args = backend._build_training_args(train_kwargs)
training_args.do_eval = True
training_args.eval_strategy = "steps"
training_args.per_device_eval_batch_size = 1

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_hf_dataset,
    eval_dataset=validation_hf_dataset,
    processing_class=tokenizer,
    callbacks=[JSONLLoggingCallback(train_kwargs["ckpt_output_dir"])],
)

result = trainer.train()
print(f"\nTraining result: {result}")
