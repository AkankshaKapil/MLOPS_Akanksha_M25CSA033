# Assignment 4 – Optimizing Transformer Translation with Ray Tune & Optuna

**Name:** Akanksha Kapil
**Roll No:** M25CSA033
**Course:** MLOps

---

## Overview

This assignment optimizes an English → Hindi Transformer translation model using Ray Tune paired with the Optuna search algorithm. The goal was to match or exceed the baseline BLEU score in significantly fewer epochs.

---

## Repository Structure

```
├── en_to_hi.ipynb                        # Original baseline notebook (provided)
├── mlops-assignment4.ipynb  # Ray Tune + Optuna implementation
├── M25CSA033_Assignment-4_report.pdf            # Assignment report
└── README.md
```

> **Note:** Model weights are hosted on HuggingFace due to file size limits.

---

## Baseline vs Best Model

| Metric | Baseline (100 epochs) | Best Model (30 epochs) |
|---|---|---|
| Training Time | ~115 min | 26.7 min |
| Final Loss | 0.0978 | 0.1668 |
| BLEU Score | 71.47 | **79.03** |

---

## Best Hyperparameter Configuration (found by Optuna)

| Hyperparameter | Value |
|---|---|
| Learning Rate | 0.000188 |
| Batch Size | 64 |
| Num Heads | 8 |
| d_ff | 1024 |
| Dropout | 0.1047 |
| Num Layers | 6 |

---

## Tuning Setup

- **Search Algorithm:** OptunaSearch (metric=loss, mode=min)
- **Scheduler:** ASHAScheduler (grace_period=5, max_t=30, reduction_factor=2)
- **Trials:** 20 total, each capped at 30 epochs
- **Hardware:** Kaggle T4 GPU (0.5 GPU per trial, 2 trials in parallel)

---

## Model Weights & Checkpoints

Hosted on HuggingFace (too large for GitHub):

🤗 **[https://huggingface.co/Akanksha8822/en-to-hi-transformer](https://huggingface.co/Akanksha8822/en-to-hi-transformer)**

Files available:
- `M25CSA033_ass_4_best_model.pth` – final best model weights
- `en_vocab.pkl` – English vocabulary
- `hi_vocab.pkl` – Hindi vocabulary

---

## How to Load the Best Model

```python
import torch
import pickle
from model import Transformer  # or copy the Transformer class from the notebook

# Load vocabs
with open("en_vocab.pkl", "rb") as f:
    en_vocab = pickle.load(f)
with open("hi_vocab.pkl", "rb") as f:
    hi_vocab = pickle.load(f)

# Load model
model = Transformer(
    src_vocab_size=len(en_vocab),
    tgt_vocab_size=len(hi_vocab),
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=1024,
    max_len=50,
    dropout=0.1047
)
model.load_state_dict(torch.load("M25CSA033_ass_4_best_model.pth", map_location="cpu"))
model.eval()
print("Model loaded successfully!")
```

---

## Sample Translations

| English | Hindi |
|---|---|
| I love you. | मैं तुमसे प्यार करता हूँ। |
| What is your name? | तुम्हारा नाम क्या है? |
| How are you? | आप कैसे हो? |
| The weather is nice today. | आज मौसम बहुत अच्छा है। |
| She is a good teacher. | वह एक अच्छी अध्यापिका है। |
