# fake_check

ALBERT fake-news classifier with FP32 and INT8 ONNX inference.

## Layout

- `data/fake_news_dataset.csv`: local source dataset. It is not committed; see `data/README.md`.
- `data/train.csv` and `data/val.csv`: generated stratified 80/20 splits, also ignored by Git.
- `models/albert_finetuned/`: fine-tuned ALBERT checkpoint.
- `models/albert_onnx/`: FP32 ONNX model and tokenizer files.
- `models/albert_onnx_quantized/`: dynamically quantized INT8 ONNX model and tokenizer files.

Model files and datasets are intentionally excluded from GitHub because they are large
generated or local input artifacts. Anyone cloning the repository must provide the source
CSV and generate the models locally.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

From a fresh clone, run these commands from the `fake_check` directory. First copy
`fake_news_dataset.csv` into `data/`. Training builds the splits and downloads
`albert-base-v2` on first use:

```powershell
.\.venv\Scripts\python.exe scripts\train_albert.py
.\.venv\Scripts\python.exe scripts\export_onnx.py
.\.venv\Scripts\python.exe scripts\test_inference.py
.\.venv\Scripts\python.exe scripts\test_inference.py --text "Your article text here"
```

For a shorter CPU run, pass options such as
`--epochs 1 --batch-size 64 --max-length 64` to the training script.

`test_inference.py` evaluates both models on `data/val.csv` and prints accuracy,
precision, recall, and F1 for the FP32 model before quantization and the INT8 model
after quantization. Use `--validation path/to/file.csv` to evaluate another file.

## Push To GitHub

```powershell
git init
git add .
git commit -m "Add ALBERT fake-news classification pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Do not use `git add -f` for ignored datasets or model artifacts unless you have a
separate distribution plan for those large files.

