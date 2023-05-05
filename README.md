# Explainable document-level simplification for reading assessment materials

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install transformer.

```bash
pip install transformer
```

## Annotation
*visualize.html* presents the created platform for elaborative simplification annotation.

## Usage
Run BSC models.
```python
python code/main.py \
    --model AutoModel \
    --state classification \
    --batch_size 32 \
    --epoch_num 10 \
    --embedding_dim 128 \
    --lr 5e-6 \
    --n_labels 2 \
    # --dataset \
    # --saved_path ./trained_models/cae/
```
LS and ES tasks finetuned the models *bart-base* and *gpt2* released by [Huggingface](https://huggingface.co/).


## Data
*meta.xlsx* contains the meta information of the 315 authentic reading assessment texts.
