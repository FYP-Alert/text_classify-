# Dataset

Place the supplied `fake_news_dataset.csv` in this directory before training.

Required columns:

- `title`
- `text`
- `label`, containing `real` or `fake`

The training script creates an 80/20 stratified split as `train.csv` and `val.csv`.
Those generated CSV files are intentionally ignored by Git.
