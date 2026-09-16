import shutil
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_dir = PROJECT_ROOT / "models" / "albert_finetuned"
fp32_dir = PROJECT_ROOT / "models" / "albert_onnx"
int8_dir = PROJECT_ROOT / "models" / "albert_onnx_quantized"


class LogitsWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


fp32_dir.mkdir(parents=True, exist_ok=True)
int8_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
wrapper = LogitsWrapper(model)
sample = tokenizer(
    "Example news article",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=128,
)
fp32_model_path = fp32_dir / "model.onnx"

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32_model_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

tokenizer.save_pretrained(str(fp32_dir))
quantize_dynamic(
    model_input=str(fp32_model_path),
    model_output=str(int8_dir / "model.onnx"),
    weight_type=QuantType.QInt8,
    per_channel=True,
)
for file_path in fp32_dir.iterdir():
    if file_path.name != "model.onnx":
        destination = int8_dir / file_path.name
        if file_path.is_file():
            shutil.copy2(file_path, destination)

print("FP32 ONNX model saved to", fp32_dir)
print("INT8 ONNX model saved to", int8_dir)