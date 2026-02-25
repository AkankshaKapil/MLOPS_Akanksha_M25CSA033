# evaluate.py

import os
import json
import torch
import numpy as np

from transformers import AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report

from data import prepare_datasets




# MODEL_PATH_h = os.environ.get("HF_MODEL_NAME", "prajjwal1/bert-tiny")
MODEL_PATH = os.environ.get("HF_MODEL_NAME", "Akanksha8822/tiny_model_ak")
RESULTS_DIR = "results_ak_local"
RESULTS_FILE = os.path.join(RESULTS_DIR, "evaluation_results.json")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")

    return {
        "accuracy": accuracy,
        "f1": f1,
    }



def main():

    
    _, test_dataset, label2id, id2label = prepare_datasets()

    

    from transformers import AutoModelForSequenceClassification, AutoConfig
    # import torch

    # # Use the FOLDER path, not the FILE path
    # LOCAL_MODEL_DIR = "/Users/jahanvigajera/Desktop/M25CSA033/model_ak/"
    
    # print(f"Loading model from: {LOCAL_MODEL_DIR}")

    # Load Config from the folder
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # Load Model from the folder
    # This automatically looks for 'pytorch_model.bin' inside the folder
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, 
        config=config
    )

    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully!")

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    
    print("Running evaluation...")
    eval_results = trainer.evaluate(test_dataset)

    print("\nEvaluation Metrics:")
    print(eval_results)

    
    predictions_output = trainer.predict(test_dataset)

    logits = predictions_output.predictions
    labels = predictions_output.label_ids

    preds = np.argmax(logits, axis=-1)

    # Compute detailed metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    report = classification_report(
        labels,
        preds,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        output_dict=True
    )

    #
    os.makedirs(RESULTS_DIR, exist_ok=True)
    final_results = {
        "eval_loss": eval_results.get("eval_loss"),
        "accuracy": accuracy,
        "f1_weighted": f1,
        "classification_report": report,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"\nEvaluation results saved to {RESULTS_FILE}")



if __name__ == "__main__":
    main()
