import pandas as pd
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AdamW, DataCollatorForSeq2Seq


def load_data(data_path: str, test_size=0.2):
    github_issues = pd.read_csv(data_path)
    train_df, test_df = train_test_split(github_issues[['body', 'issue_title']], test_size=test_size, random_state=42)

    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    full_dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    return full_dataset_dict


def preprocess_function(examples, tokenizer):
    inputs = ["Summary of the following GitHub issue: " + doc for doc in examples["body"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length", return_tensors='pt')
    labels = tokenizer(examples["issue_title"], max_length=256, truncation=True, padding="max_length",
                       return_tensors='pt')
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
