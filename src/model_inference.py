
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch


# Dataset
df = pd.read_csv("data/climate_promise_constraint.csv")
df_sampled = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=250, random_state=42))
      .reset_index(drop=True)
)


# Define the mapping from string labels to numeric labels
label2id = {
    "Neutral": 0,
    "Promise": 1,
    "Contradiction": 2
}
# Map the string labels to integers
df_sampled["label"] = df_sampled["label"].map(label2id)


model_name = "dariast/climatebert_promise_constraint"
NUM_LABELS = 3 
SEED = 42
N_SPLITS = 5


# Prepare tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, max_len=512)


# Put the model in evaluation mode
model.eval()


def compute_metrics(true, pred):
    """
    Computes evaluation metrics for classification.

    Parameters:
    - true: List or array of true labels
    - pred: List or array of predicted labels

    Returns:
    - Dictionary containing accuracy, precision, recall, and F1-macro scores.
    """
    acc = accuracy_score(true, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='macro')

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1
    }


def predict(texts):
    # 1) Tokenize the input texts with truncation and padding
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=512,       
        padding='max_length', 
        return_tensors='pt'
    )
    
    # 2) Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    # 3) Forward pass
    with torch.no_grad():
        outputs = model(**encoded)
    
    # 4) Get predictions
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
    return preds


# CROSS VALIDATION EVALUATION
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
all_fold_metrics = []


texts = df_sampled['sentence'].tolist()
labels = df_sampled['label'].tolist()


for fold, (train_idx, val_idx) in enumerate(kf.split(texts, labels), 1):
    print(f"\n=== FOLD {fold} ===")
    # We won't train or fine-tune; we only evaluate on the validation set
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    # Get predictions from the model, with truncated/padded inputs
    pred_labels = predict(val_texts)

    # Compute metrics
    metrics = compute_metrics(val_labels, pred_labels)
    print(f"Fold {fold} Metrics: {metrics}")
    all_fold_metrics.append(metrics)


# AGGREGATE METRICS
df_metrics = pd.DataFrame(all_fold_metrics)
print("\nPer-fold metrics:")
print(df_metrics)

# Compute mean and std of F1-macro
f1_mean = df_metrics['f1_macro'].mean()
f1_std = df_metrics['f1_macro'].std()

print(f"\nFinal F1-macro mean: {f1_mean:.4f}")
print(f"Final F1-macro std:  {f1_std:.4f}")

# Optionally save metrics
df_metrics.to_csv("model_metrics.csv", index=False)





