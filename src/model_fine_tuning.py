import os
import json
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import os
import dotenv
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from huggingface_hub import login


# 1) Utility: fix random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 2) Login to Hugging Face Hub
set_seed(42)
login(token=HF_TOKEN)


# 3) Load CSV file, rename columns, convert labels, split train/test
custom_data_path = "data/climate_promise_constraint.csv"  # Replace with your actual file path

# Read the CSV
data_df = pd.read_csv(custom_data_path)

data_df = data_df[["sentence", "label"]]
data_df.rename(columns={"sentence": "text"}, inplace=True)

# Convert string labels to integers
label_mapping = {label: idx for idx, label in enumerate(data_df["label"].unique())}
data_df["label"] = data_df["label"].map(label_mapping)

# Split the dataset
train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)

# Turn them into Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_data, preserve_index=False)
test_dataset  = Dataset.from_pandas(test_data, preserve_index=False)


# 4) Define compute_metrics and tokenization
def compute_metrics(eval_preds):
    """
    Returns precision (macro), recall (macro), f1 (macro), and accuracy.
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    precision = precision_score(labels, predictions, average="macro")
    recall    = recall_score(labels, predictions, average="macro")
    f1        = f1_score(labels, predictions, average="macro")
    accuracy  = accuracy_score(labels, predictions)

    return {
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "accuracy": accuracy
    }

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


# 5) K-fold cross-validation on TRAIN data
def cross_validate_model(
    model_name_or_path,
    train_data,
    num_labels=2,
    num_folds=5,
    learning_rate=2e-5,
    num_train_epochs=3,
    batch_size=16,
    hf_username="dariast",
    do_push=False
):
    """
    - Perform K-fold cross-validation on the *training* data.
    - For each fold, train and save the best checkpoint (based on F1 macro).
    - After all folds, pick the single best fold, load its best checkpoint, and
      optionally push that model to the Hugging Face Hub (if do_push=True).
    - Returns (mean_f1, std_f1, best_fold_idx, best_fold_output_dir).
    """

    # 1) Tokenize the training data
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def tokenize_wrapper(example):
        return tokenize_function(example, tokenizer)

    train_data_tokenized = train_data.map(tokenize_wrapper, batched=True)
    keep_cols = ["input_ids", "attention_mask", "label"]
    train_data_tokenized = train_data_tokenized.remove_columns(
        [col for col in train_data_tokenized.column_names if col not in keep_cols]
    )
    train_data_tokenized.set_format("torch")

    # Convert to list for manual indexing in KFold
    all_data = [train_data_tokenized[i] for i in range(len(train_data_tokenized))]

    # 2) Setup KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    best_fold_idx = -1
    best_fold_f1 = -1.0
    best_fold_output_dir = None
    fold_f1_scores = []

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data_list):
            self.data_list = data_list
        def __len__(self):
            return len(self.data_list)
        def __getitem__(self, idx):
            return self.data_list[idx]

    # 3) Loop over folds
    for fold_idx, (train_index, val_index) in enumerate(kf.split(all_data)):
        print(f"\n===== Fold {fold_idx+1} / {num_folds} =====")

        # Prepare subset
        fold_train_data = [all_data[i] for i in train_index]
        fold_val_data   = [all_data[i] for i in val_index]

        fold_train_dataset = SimpleDataset(fold_train_data)
        fold_val_dataset   = SimpleDataset(fold_val_data)

        # Load a fresh model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels
        )

        # Output dir for this fold
        fold_output_dir = f"./{model_name_or_path.replace('/', '_')}_fold_{fold_idx}"

        # Training arguments
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,           # Keep only best checkpoint
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,

            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,

            logging_steps=50,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        # Train
        trainer.train()

        # Evaluate on validation fold
        eval_metrics = trainer.evaluate(eval_dataset=fold_val_dataset)
        fold_f1 = eval_metrics["eval_f1_macro"]
        fold_f1_scores.append(fold_f1)

        print(f"\nFold {fold_idx+1} metrics:")
        for k, v in eval_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Track best fold
        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_fold_idx = fold_idx
            best_fold_output_dir = fold_output_dir

        # Cleanup
        del model
        torch.cuda.empty_cache()

    # 4) Report
    mean_f1 = np.mean(fold_f1_scores)
    std_f1  = np.std(fold_f1_scores)
    print(f"\nOverall F1-macro across {num_folds} folds: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Best fold was Fold {best_fold_idx+1} with F1-macro = {best_fold_f1:.4f}")

    # 5) Optionally push the best fold's model
    if do_push:
        print("\nLoading best fold checkpoint and pushing to the Hub...")
        best_model = AutoModelForSequenceClassification.from_pretrained(best_fold_output_dir)
        final_hub_model_id = f"{hf_username}/{model_name_or_path.replace('/', '_')}_best_fold"

        final_args = TrainingArguments(
            output_dir=best_fold_output_dir,
            push_to_hub=True,
            hub_model_id=final_hub_model_id
        )

        final_trainer = Trainer(
            model=best_model,
            args=final_args,
            tokenizer=tokenizer
        )
        final_trainer.push_to_hub(commit_message="Best fold model from cross-validation")

    return mean_f1, std_f1, best_fold_idx, best_fold_output_dir


# 6) Hyperparameter Search (Grid Search) on TRAIN data
def hyperparameter_search(
    model_name_or_path,
    train_data,
    param_grid,
    num_labels=2,
    num_folds=5,
    hf_username="dariast"
):
    """
    For each hyperparameter set in param_grid, run cross_validate_model()
    on the training split. Returns (best_params, best_mean_f1).
    """

    best_params = None
    best_mean_f1 = -1.0

    for params in param_grid:
        print("\n===================================================")
        print(f"Evaluating hyperparams: {params}")
        print("===================================================")

        mean_f1, std_f1, _, _ = cross_validate_model(
            model_name_or_path=model_name_or_path,
            train_data=train_data,
            num_labels=num_labels,
            num_folds=num_folds,
            learning_rate=params["learning_rate"],
            num_train_epochs=params["num_train_epochs"],
            batch_size=params["batch_size"],
            hf_username=hf_username,
            do_push=False  # Only push after we find the best
        )

        print(f"  => Mean F1-macro: {mean_f1:.4f} ± {std_f1:.4f}")
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_params = params

    return best_params, best_mean_f1


# 7) Optional: Train on ALL train data with best hyperparams & evaluate on TEST
def train_on_full_train_and_eval(
    best_params,
    model_name_or_path,
    train_data,
    test_data,
    num_labels=2
):
    """
    Train a fresh model on ALL of the train_data with best_params, then
    evaluate on test_data.
    """
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    def tokenize_wrapper(example):
        return tokenize_function(example, tokenizer)

    train_tokenized = train_data.map(tokenize_wrapper, batched=True)
    test_tokenized  = test_data.map(tokenize_wrapper,  batched=True)

    # Keep only the required columns
    keep_cols = ["input_ids", "attention_mask", "label"]
    train_tokenized = train_tokenized.remove_columns(
        [col for col in train_tokenized.column_names if col not in keep_cols]
    )
    test_tokenized = test_tokenized.remove_columns(
        [col for col in test_tokenized.column_names if col not in keep_cols]
    )
    train_tokenized.set_format("torch")
    test_tokenized.set_format("torch")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels
    )

    # Training args
    final_output_dir = "./final_full_train_model"
    training_args = TrainingArguments(
        output_dir=final_output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",  # We won't save multiple checkpoints here
        num_train_epochs=best_params["num_train_epochs"],
        per_device_train_batch_size=best_params["batch_size"],
        per_device_eval_batch_size=best_params["batch_size"],
        learning_rate=best_params["learning_rate"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train on all train data
    trainer.train()

    # Evaluate on test data
    test_eval = trainer.evaluate(eval_dataset=test_tokenized)
    print("\nFinal Test Set Evaluation:")
    for k, v in test_eval.items():
        print(f"  {k}: {v:.4f}")

    return model, trainer, final_output_dir


# 8) Main
if __name__ == "__main__":
    # Define parameter grid
    param_grid = [
        {"learning_rate": 1e-5, "num_train_epochs": 3, "batch_size": 16},
        {"learning_rate": 2e-5, "num_train_epochs": 3, "batch_size": 16},
        {"learning_rate": 2e-5, "num_train_epochs": 3, "batch_size": 8},
        {"learning_rate": 3e-5, "num_train_epochs": 3, "batch_size": 16},
    ]

    # 1) Hyperparameter search on TRAIN split
    best_params, best_mean_f1 = hyperparameter_search(
        model_name_or_path="roberta-base",
        train_data=train_dataset,
        param_grid=param_grid,
        num_labels=len(label_mapping),
        num_folds=5,
        hf_username="dariast"
    )

    print("\nBest hyperparameters:")
    print(best_params)
    print(f"Best mean F1-macro: {best_mean_f1:.4f}")

    # 2) Save best hyperparams
    with open("best_hyperparams.log", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)
    print("Saved best hyperparameters to best_hyperparams.log")

    # 3) Re-run cross-validation with the best hyperparams, push best fold
    print("\nRe-running cross-validation with BEST hyperparams and pushing the best fold to the Hub...")
    cross_validate_model(
        model_name_or_path="roberta-base",
        train_data=train_dataset,
        num_labels=len(label_mapping),
        num_folds=5,
        learning_rate=best_params["learning_rate"],
        num_train_epochs=best_params["num_train_epochs"],
        batch_size=best_params["batch_size"],
        hf_username="dariast",
        do_push=True
    )

    # 4) Train on ALL train data with best hyperparams & evaluate on TEST
    #    This is a typical final step to get performance on the unseen test set.
    model, trainer, final_out_dir = train_on_full_train_and_eval(
        best_params=best_params,
        model_name_or_path="roberta-base",
        train_data=train_dataset,
        test_data=test_dataset,
        num_labels=len(label_mapping)
    )
