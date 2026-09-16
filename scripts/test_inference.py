"""Evaluate FP32 and INT8 ONNX models and optionally classify one text."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def predict(texts: list[str], model_dir: Path, max_length: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoded = tokenizer(
        texts,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    session = ort.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    logits = session.run(
        ["logits"],
        {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        },
    )[0]
    return np.argmax(logits, axis=-1)


def evaluate(model_dir: Path, validation_path: Path, max_length: int) -> dict[str, float]:
    validation = pd.read_csv(validation_path).dropna(subset=["text", "label"])
    labels = validation["label"].astype(int).to_numpy()
    predictions = predict(validation["text"].astype(str).tolist(), model_dir, max_length)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Optional news text to classify")
    parser.add_argument("--validation", type=Path, default=PROJECT_ROOT / "data" / "val.csv")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    models = {
        "albert_fp32": PROJECT_ROOT / "models" / "albert_onnx",
        "albert_int8": PROJECT_ROOT / "models" / "albert_onnx_quantized",
    }
    for name, model_dir in models.items():
        print({name: evaluate(model_dir, args.validation, args.max_length)})
    if args.text:
        for name, model_dir in models.items():
            print({name: {"prediction": int(predict([args.text], model_dir, args.max_length)[0])}})


if __name__ == "__main__":
    main()
