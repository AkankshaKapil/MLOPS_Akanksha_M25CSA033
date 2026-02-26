# Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

## Overview

This project demonstrates a complete end-to-end Machine Learning workflow using the Hugging Face ecosystem and Docker containerization.

The objective of this assignment was to convert a Jupyter notebook into a production-ready modular pipeline, fine-tune a transformer-based model for multi-class text classification, deploy the trained model to Hugging Face Hub, and ensure full reproducibility using Docker.

---

## Project Workflow

The complete pipeline follows these steps:

1. Notebook Development  
2. Conversion to Modular Python Scripts  
3. Model Training using Hugging Face Trainer API  
4. Local Model Evaluation  
5. Model Deployment to Hugging Face Hub  
6. Docker-based Containerization  
7. Docker Re-evaluation using Hosted Model  
8. GitHub Publication  

---

## Project Structure

```
project/
│
├── src/
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py

│
├── Dockerfile.train
├── Dockerfile.eval
├── requirements.txt
└── README.md
```

---

## Model Details

- Pretrained Model: `prajjwal1/bert-tiny`
- Maximum Sequence Length: 128
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Framework: Hugging Face Trainer API

### Reason for Selection

- Lightweight transformer architecture  
- CPU-friendly  
- Lower computational cost  
- Suitable for containerized deployment  

---

## Dataset

Multi-class book genre classification (8 classes):

- Romance  
- Mystery Thriller Crime  
- Young Adult  
- History Biography  
- Comics Graphic  
- Fantasy Paranormal  
- Children  
- Poetry  

Each class contains 200 samples.

Total evaluation samples: 1600.

---

## Evaluation Results

### Local Evaluation

| Metric       | Value   |
|-------------|---------|
| Eval Loss   | 3.5703  |
| Accuracy    | 0.0956  |
| Weighted F1 | 0.0967  |

### Docker Evaluation

| Metric       | Value   |
|-------------|---------|
| Eval Loss   | 3.6730  |
| Accuracy    | 0.1181  |
| Weighted F1 | 0.1122  |

Docker evaluation achieved slightly higher accuracy and F1-score. Minor differences may be due to environment or random seed variations.

---

## Hugging Face Model Repository

The trained model is publicly available at:

https://huggingface.co/Akanksha8822/tiny_model_ak

Upload commands used:

```python
model.push_to_hub("Akanksha8822/tiny_model_ak")
tokenizer.push_to_hub("Akanksha8822/tiny_model_ak")
```

---

## Docker Setup

### Build Training Image

```bash
docker build -t ak_train -f Dockerfile.train .
```

### Run Training Container

```bash
docker run -e HF_TOKEN="YOUR_TOKEN" ak_train python /src/train.py
```

### Build Evaluation Image

```bash
docker build -t ak_eval -f Dockerfile.eval .
```

### Run Evaluation Container

```bash
docker run -e HF_TOKEN="YOUR_TOKEN" ak_eval python /src/evaluate.py
```

The evaluation container pulls the model directly from Hugging Face Hub.

---

## Reproducibility

This project ensures reproducibility through:

- Fixed model architecture  
- Consistent training configuration  
- Docker containerization  
- Hosted model repository  
- Public GitHub source code  

---

## Challenges Faced

- CPU-only training increased runtime  
- Low accuracy due to small transformer capacity  
- Managing Hugging Face authentication inside Docker  

---

## Limitations

- Low overall accuracy (~11%)  
- No hyperparameter tuning  
- No learning rate scheduler  
- No early stopping  

---

## Future Improvements

- Use larger transformer models (e.g., `bert-base-uncased`)  
- Increase training epochs  
- Perform hyperparameter tuning  
- Train using GPU acceleration  

---

## Author

Akanksha Kapil  
M25CSA033  

---

## Submission Links

GitHub Repository:  
https://github.com/AkankshaKapil/MLOPS_Akanksha_M25CSA033/tree/Assignment_3  

Hugging Face Model:  
https://huggingface.co/Akanksha8822/tiny_model_ak
