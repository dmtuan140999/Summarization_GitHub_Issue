from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AdamW, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch
import pandas as pd

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

model = AutoModelForSeq2SeqLM.from_pretrained("minhtuan7akp/bart_github_summarization")
tokenizer = AutoTokenizer.from_pretrained("minhtuan7akp/bart_github_summarization")

# Thiết lập thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
github_issues = pd.read_csv("../data/github_issues.csv")
evaluate_github_issues = github_issues[-3000:].reset_index(drop=True)
evaluate_github_issues['summary_bartpho'] = evaluate_github_issues['body'].apply(summary)

from rouge import Rouge

rouge = Rouge()
def caculate_rouge(candidate, reference):
    rouge_scores = rouge.get_scores(candidate, reference)
    return rouge_scores[0]['rouge-1']['f'], rouge_scores[0]['rouge-2']['f'], rouge_scores[0]['rouge-l']['f']
# Khởi tạo danh sách để lưu các chỉ số ROUGE
rouge_1_scores = []
rouge_2_scores = []
rouge_l_scores = []
# Duyệt qua từng cặp văn bản từ hai cột và tính các chỉ số ROUGE
for index, row in evaluate_github_issues.iterrows():
    candidate = row['summary_bartpho']
    reference = row['issue_title']
    rouge_1, rouge_2, rouge_l = caculate_rouge(candidate, reference)
    rouge_1_scores.append(rouge_1)
    rouge_2_scores.append(rouge_2)
    rouge_l_scores.append(rouge_l)

# Tính trung bình của từng chỉ số ROUGE
avg_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
avg_rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
print(f"Average ROUGE-1 F1 Score: {avg_rouge_1}")
print(f"Average ROUGE-2 F1 Score: {avg_rouge_2}")
print(f"Average ROUGE-L F1 Score: {avg_rouge_l}")