import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_name = "albert-base-v2"
output_dir = PROJECT_ROOT / "models" / "albert_finetuned"
source_data = PROJECT_ROOT / "data" / "fake_news_dataset.csv"
train_data = PROJECT_ROOT / "data" / "train.csv"
validation_data = PROJECT_ROOT / "data" / "val.csv"
max_length = 128
num_labels = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=max_length)
    return parser.parse_args()


def prepare_splits() -> None:
    frame = pd.read_csv(source_data)
    required_columns = {"title", "text", "label"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")

    frame = frame.dropna(subset=["title", "text", "label"]).copy()
    frame["label"] = frame["label"].astype(str).str.lower().map({"real": 0, "fake": 1})
    if frame["label"].isna().any():
        raise ValueError("Labels must be either 'real' or 'fake'.")
    frame["text"] = frame["title"].astype(str) + "\n" + frame["text"].astype(str)
    frame = frame[["text", "label"]]
    train_frame, validation_frame = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame["label"],
    )
    train_frame.to_csv(train_data, index=False)
    validation_frame.to_csv(validation_data, index=False)
    print(
        f"Prepared {len(train_frame)} training rows and "
        f"{len(validation_frame)} validation rows from {source_data.name}"
    )


prepare_splits()
args = parse_args()

# Load data
dataset = load_dataset(
    "csv",
    data_files={
        "train": str(train_data),
        "validation": str(validation_data),
    },
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
    )

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }

training_kwargs = {
    "output_dir": str(output_dir),
    "save_strategy": "epoch",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": args.batch_size,
    "per_device_eval_batch_size": args.batch_size,
    "num_train_epochs": args.epochs,
    "weight_decay": 0.01,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "logging_steps": 50,
    "report_to": "none",
}
if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
    training_kwargs["eval_strategy"] = "epoch"
else:
    training_kwargs["evaluation_strategy"] = "epoch"
training_args = TrainingArguments(**training_kwargs)

trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)
if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer
trainer = Trainer(**trainer_kwargs)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Model saved to", output_dir)