from data_loading.data_loader import *
from models.models import *
from training.training import *
from datasets import load_from_disk

full_dataset_dict = load_data("data/github_issues.csv")
model, tokenizer = load_model(model_name = "facebook/bart-base")

tokenized_data = load_from_disk('data/full_dataset_dict')
data_collator = DataCollatorForSeq2Seq(tokenizer, model="facebook/bart-base")

train_dataloader = DataLoader(tokenized_data['train'], shuffle=True, batch_size=2, collate_fn=data_collator)
test_dataloader = DataLoader(tokenized_data['test'], batch_size=2, collate_fn=data_collator)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

wandb.login(key='6a323cfe5341553410214585d588f10485e8ac66')
wandb.init(project="summarization", name='seminar2_kaggle_bartpho_newest')

train_model(model, train_dataloader, test_dataloader, tokenizer, epochs=2, batch_size=2, lr=5e-5, checkpoint_dir='checkpoint')
evaluate_model(model, test_dataloader, device)