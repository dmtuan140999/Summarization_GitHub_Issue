import torch

def generate_summary(model, tokenizer, inputs, max_length=256):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer("Summary of the following GitHub issue: " + inputs, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs['input_ids'], max_length=max_length, num_beams=5, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)


def train_model(model, train_dataloader, test_dataloader, tokenizer, epochs=5, batch_size=2, lr=5e-5,
                checkpoint_dir='../checkpoint'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="summarization", name='seminar2_kaggle_bartpho_newest')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            wandb.log({"train_loss": loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")
        model.save_pretrained(f'{checkpoint_dir}/epoch_{epoch + 1}')
        tokenizer.save_pretrained(f'{checkpoint_dir}/epoch_{epoch + 1}')

        artifact = wandb.Artifact(f'model-bartpho-checkpoint-epoch-{epoch + 1}', type='model')
        artifact.add_dir(f'{checkpoint_dir}/epoch_{epoch + 1}')
        wandb.log_artifact(artifact)

        evaluate_model(model, test_dataloader, device)

    wandb.finish()


def evaluate_model(model, test_dataloader, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)