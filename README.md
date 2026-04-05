# Assignment 5 — DL Operations
### ViT-S LoRA Fine-tuning on CIFAR-100 + Adversarial Attacks on CIFAR-10

**Roll No:** M25CSA033 | **Institute:** IIT Jodhpur

[![WandB](https://img.shields.io/badge/WandB-Experiment_Logs-orange?logo=weightsandbiases)](https://wandb.ai/akankshakapil8-iit-jodhpur/adversarial-attacks-cifar10)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model_Weights-yellow?logo=huggingface)](https://huggingface.co/)

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Q1 — ViT-S LoRA Fine-tuning (CIFAR-100)](#q1--vit-s-lora-fine-tuning-cifar-100)
4. [Q2 — Adversarial Attacks (CIFAR-10)](#q2--adversarial-attacks-cifar-10)
5. [Results — Q1](#results--q1)
6. [Results — Q2](#results--q2)
7. [WandB & HuggingFace Links](#wandb--huggingface-links)

---

## Project Structure

```
Assignment5/
│
├── q1/
│   ├── __init__.py
│   ├── data_loader.py          # CIFAR-100 dataset loading & transforms
│   ├── evaluate.py             # Evaluation loop, class-wise accuracy histogram
│   ├── model.py                # ViT-S model + LoRA injection via HuggingFace PEFT
│   ├── optuna_optimizer.py     # Optuna hyperparameter search for LoRA configs
│   ├── run_experiments.py      # Runs all rank/alpha combinations (Exps 1–10)
│   └── train.py                # Training loop with WandB logging
│
├── q2/
│   ├── __init__.py
│   ├── adversarial_detector.py # ResNet34 binary detector (clean vs adversarial)
│   ├── fgsm_art.py             # FGSM attack via IBM ART
│   ├── fgsm_scratch.py         # FGSM attack from scratch (pure PyTorch)
│   ├── run_attacks.py          # Orchestrates all attacks + WandB logging
│   ├── train_resnet18.py       # Trains ResNet18 from scratch on CIFAR-10
│   └── visualize.py            # Plots clean vs adversarial image comparisons
│
├── weights/
│   ├── best_lora_model_optuna.pth   # Best Q1 model (Optuna)
│   ├── resnet18_cifar10.pth         # Trained ResNet18 victim model (Q2)
│   ├── detector_pgd.pth             # PGD adversarial detector
│   └── detector_bim.pth             # BIM adversarial detector
│
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA 11.8+ (GPU strongly recommended)
- Docker (**required** — all code must run inside Docker per assignment guidelines)

### 1. Docker Setup
```bash
# Pull base image
docker pull pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Build container
docker build -t dlops-assignment5 .

# Run with GPU + mount project directory
docker run --gpus all -it -v $(pwd):/workspace dlops-assignment5
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
torch==2.1.0
torchvision==0.16.0
transformers==4.38.0
peft==0.9.0
datasets==2.18.0
adversarial-robustness-toolbox==1.17.0
wandb==0.17.0
optuna==3.6.0
scikit-learn==1.4.0
matplotlib==3.8.0
numpy==1.26.0
tqdm==4.66.0
timm==0.9.12
```

### 3. WandB Login
```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
```

---

## Q1 — ViT-S LoRA Fine-tuning (CIFAR-100)

**Model:** ViT-Small (pre-trained on ImageNet)
**Dataset:** CIFAR-100
**Task:** 100-class image classification
**LoRA:** Injected into Q, K, V attention weights via HuggingFace PEFT

### Run All Experiments (Exps 1–10: all rank/alpha combos + no-LoRA baseline)
```bash
cd q1
python run_experiments.py \
  --epochs 10 \
  --batch_size 64 \
  --wandb_project dlops-ass5-q1
```

### Train a Single LoRA Configuration
```bash
python train.py \
  --rank 2 \
  --alpha 2 \
  --dropout 0.1 \
  --target_modules q_proj k_proj v_proj \
  --epochs 10 \
  --lr 1e-3 \
  --batch_size 64 \
  --wandb_project dlops-ass5-q1
```

### Train Without LoRA (Classification Head Only)
```bash
python train.py \
  --no_lora \
  --epochs 10 \
  --lr 1e-3 \
  --batch_size 64 \
  --wandb_project dlops-ass5-q1
```

### Optuna Hyperparameter Search (Q5)
```bash
python optuna_optimizer.py \
  --n_trials 50 \
  --epochs 10 \
  --wandb_project dlops-ass5-q1-optuna \
  --save_path ../weights/best_lora_model_optuna.pth
```

### Evaluate a Saved Model
```bash
python evaluate.py \
  --checkpoint ../weights/best_lora_model_optuna.pth \
  --rank 2 \
  --alpha 2 \
  --dataset cifar100
```

---

## Q2 — Adversarial Attacks (CIFAR-10)

**Victim Model:** ResNet18 (non-pretrained, trained from scratch)
**Dataset:** CIFAR-10
**Attacks:** FGSM (Scratch & IBM ART), PGD, BIM
**Detector:** ResNet34 binary classifier (clean=0 vs adversarial=1)

### Step 1 — Train ResNet18 from Scratch
```bash
cd q2
python train_resnet18.py \
  --epochs 80 \
  --lr 0.1 \
  --batch_size 128 \
  --wandb_project adversarial-attacks-cifar10 \
  --save_path ../weights/resnet18_cifar10.pth
```

### Step 2 — Run All Attacks + Log to WandB
```bash
python run_attacks.py \
  --checkpoint ../weights/resnet18_cifar10.pth \
  --epsilons 0.0 0.01 0.02 0.03 0.05 0.08 0.1 0.15 0.2 \
  --wandb_project adversarial-attacks-cifar10
```

### Run FGSM from Scratch Only
```bash
python fgsm_scratch.py \
  --checkpoint ../weights/resnet18_cifar10.pth \
  --epsilon 0.03
```

### Run FGSM via IBM ART Only
```bash
python fgsm_art.py \
  --checkpoint ../weights/resnet18_cifar10.pth \
  --epsilon 0.03 \
  --wandb_project adversarial-attacks-cifar10
```

### Train Adversarial Detectors (PGD & BIM)
```bash
# PGD Detector
python adversarial_detector.py \
  --victim_checkpoint ../weights/resnet18_cifar10.pth \
  --attack pgd \
  --epochs 40 \
  --eps 0.05 \
  --n_train 8000 \
  --wandb_project adversarial-attacks-cifar10 \
  --save_path ../weights/detector_pgd.pth

# BIM Detector
python adversarial_detector.py \
  --victim_checkpoint ../weights/resnet18_cifar10.pth \
  --attack bim \
  --epochs 40 \
  --eps 0.05 \
  --n_train 8000 \
  --wandb_project adversarial-attacks-cifar10 \
  --save_path ../weights/detector_bim.pth
```

### Visualize Clean vs Adversarial Images
```bash
python visualize.py \
  --checkpoint ../weights/resnet18_cifar10.pth \
  --epsilon 0.05 \
  --n_samples 10 \
  --wandb_project adversarial-attacks-cifar10
```

---

## Results — Q1

### Summary: All LoRA Configurations

| LoRA      | Rank | Alpha | Dropout | Test Acc   | Trainable Params |
|:----------|:----:|:-----:|:-------:|:----------:|:----------------:|
| Without   | N/A  | N/A   | N/A     | 35.37%     | 38,500           |
| With LoRA | 2    | 2     | 0.1     | 83.92%     | 93,796           |
| With LoRA | 2    | 4     | 0.1     | 85.46 %    | 93,796           |
| With LoRA | 2    | 8     | 0.1     | 41.47%     | 93,796           |
| With LoRA | 4    | 2     | 0.1     | 28.69%     | 149,092          |
| With LoRA | 4    | 4     | 0.1     | 36.70%     | 149,092          |
| With LoRA | 4    | 8     | 0.1     | 37.44%     | 149,092          |
| With LoRA | 8    | 2     | 0.1     | 30.69%     | 259,684          |
| With LoRA | 8    | 4     | 0.1     | 33.34%     | 259,684          |
| With LoRA | 8    | 8     | 0.1     | 36.18%     | 259,684          |

> **Best config: Rank=2, Alpha=2 → 83.92% with only 93,796 trainable params**

**Key Observations:**
- LoRA with Rank 2 / Alpha 2 gives a huge jump over head-only fine-tuning: 83.92% vs 35.37%
- High alpha (8) with any rank causes training instability — loss starts very high due to LoRA scaling mismatch
- Increasing rank beyond 2 does not improve results for CIFAR-100 + ViT-S
- The best model is highly parameter-efficient: only 93,796 trainable out of ~22M total

### Optuna Best Config (Q5) — Training Curve

| Epoch | Train Loss | Train Acc | Val Acc    |
|------:|:----------:|:---------:|:----------:|
| 1     | 1.3706     | 64.39%    | 81.54%     |
| 2     | 0.4551     | 85.96%    | 83.70%     |
| 3     | 0.3278     | 89.78%    | 84.92%     |
| 4     | 0.2453     | 92.27%    | 84.88%     |
| 5     | 0.1892     | 94.06%    | **85.12%** |
| 6     | 0.1450     | 95.55%    | **85.12%** |
| 7     | 0.1139     | 96.55%    | 84.62%     |
| 8     | 0.0885     | 97.40%    | 85.02%     |
| 9     | 0.0698     | 97.98%    | 85.00%     |
| 10    | 0.0590     | 98.30%    | 84.70%     |

Best model weights: `weights/best_lora_model_optuna.pth`

---

## Results — Q2

### ResNet18 Clean Training

| Epoch | Train Loss | Train Acc | Test Acc   |
|------:|:----------:|:---------:|:----------:|
| 1     | 2.1587     | 24.37%    | 36.71%     |
| 10    | 0.4933     | 82.95%    | 78.35%     |
| 20    | 0.3515     | 87.99%    | 83.04%     |
| 30    | 0.2871     | 90.31%    | 87.47%     |
| 40    | 0.2169     | 92.53%    | 89.92%     |
| 50    | 0.1423     | 95.08%    | 90.07%     |
| 60    | 0.0569     | 98.12%    | 92.29%     |
| 70    | 0.0064     | 99.88%    | 94.63%     |
| 80    | 0.0029     | 99.97%    | **94.76%** |

Requirement ≥72% clean accuracy — achieved **94.76%**

### FGSM Attack: Accuracy vs Epsilon

| ε     | Clean  | FGSM Scratch | FGSM ART | Scratch Drop | ART Drop |
|:-----:|:------:|:------------:|:--------:|:------------:|:--------:|
| 0.000 | 94.76% | 94.76%       | 94.76%   | 0.00%        | 0.00%    |
| 0.010 | 94.76% | 78.18%       | 55.25%   | 16.58%       | 39.51%   |
| 0.020 | 94.76% | 64.70%       | 49.25%   | 30.06%       | 45.51%   |
| 0.030 | 94.76% | 57.71%       | 45.35%   | 37.05%       | 49.41%   |
| 0.050 | 94.76% | 50.77%       | 37.70%   | 43.99%       | 57.06%   |
| 0.080 | 94.76% | 46.11%       | 26.45%   | 48.65%       | 68.31%   |
| 0.100 | 94.76% | 44.46%       | 21.25%   | 50.30%       | 73.51%   |
| 0.150 | 94.76% | 41.08%       | 12.55%   | 53.68%       | 82.21%   |
| 0.200 | 94.76% | 38.12%       | 10.10%   | 56.64%       | 84.66%   |

**Analysis:**
- FGSM ART is more aggressive than from-scratch at every epsilon value
- At ε=0.2, ART drops accuracy to 10.10% vs Scratch's 38.12%
- ART applies perturbations in raw [0,1] pixel space with hard clipping, making the signed gradient step fully effective
- From-scratch operates in normalised space, slightly diluting effective perturbation magnitude
- Both attacks show diminishing returns beyond ε=0.05

### Adversarial Detection (ResNet34 Binary Classifier)

| Attack | Victim Acc Under Attack | Detector Accuracy |
|:-------|:-----------------------:|:-----------------:|
| PGD    | 4.20%                   | ≥70% (after fix)  |
| BIM    | 4.20%                   | ≥70% (after fix)  |

Both PGD and BIM reduce victim model accuracy from 94.40% → 4.20%, confirming highly effective adversarials.

---

## WandB & HuggingFace Links

| Resource | Link |
|:---------|:-----|
| WandB Project | https://wandb.ai/akankshakapil8-iit-jodhpur/adversarial-attacks-cifar10 |
| WandB — ResNet18 Training + FGSM | https://wandb.ai/akankshakapil8-iit-jodhpur/adversarial-attacks-cifar10/runs/ja2wrjdk |
| WandB — Adversarial Detection | https://wandb.ai/akankshakapil8-iit-jodhpur/adversarial-attacks-cifar10/runs/9d1fy5pm |
| HuggingFace — Best LoRA Model | *(add link after pushing weights)* |

---


### LoRA Formulation

Low-Rank Adaptation (LoRA) modifies the pretrained weight matrix \( W \) as:

\[
W' = W + \frac{\alpha}{r} BA
\]

where:
- \( W \) = original pretrained weights  
- \( B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k} \) are low-rank matrices  
- \( r \) = rank  
- \( \alpha \) = scaling factor  

This allows efficient fine-tuning with significantly fewer trainable parameters.
