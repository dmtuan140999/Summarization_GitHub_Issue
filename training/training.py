import torch
from transformers import AdamW, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import wandb
import os

def evaluate_model(model, test_dataloader, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

            wandb.log({"val_loss": loss.item()})

    avg_val_loss = total_val_loss / len(test_dataloader)
    print(f"Validation Loss: {avg_val_loss}")
