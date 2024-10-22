from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AdamW, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch

model = AutoModelForSeq2SeqLM.from_pretrained("minhtuan7akp/bart_github_summarization")
tokenizer = AutoTokenizer.from_pretrained("minhtuan7akp/bart_github_summarization")

# Thiết lập thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def summary(sentence):
    text =  "Summary of the following GitHub issue: " + sentence
    encoding = tokenizer(text, max_length=1024, truncation=True, padding="max_length", return_tensors='pt')
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256
    )
    line = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return line
print(summary("Add value ranges to table formats on that page"))