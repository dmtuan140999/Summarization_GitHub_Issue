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
print("Input: hi, when i release it seems has a release late.. if i deploy the current folder has the good symlink to realeases/<release_id> but the code executed continue to be this of the prev release, if i execut php bin/console cache:clear -e prod to clear the cache, the cache is cleared but the code executed is always the one of the prev release. if i remove all folders of releases directory except the last one, everything is cool, the code executed is the good one... anybody have the same problem ? i use the symfony3 recipe. thanks :")
print("Output: ",summary('hi, when i release it seems has a release late.. if i deploy the current folder has the good symlink to realeases/<release_id> but the code executed continue to be this of the prev release, if i execut php bin/console cache:clear -e prod to clear the cache, the cache is cleared but the code executed is always the one of the prev release. if i remove all folders of releases directory except the last one, everything is cool, the code executed is the good one... anybody have the same problem ? i use the symfony3 recipe. thanks :'
))