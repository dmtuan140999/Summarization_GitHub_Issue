import torch

def generate_summary(model, tokenizer, inputs, max_length=256):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer("Summary of the following GitHub issue: " + inputs, return_tensors="pt", truncation=True)
        outputs = model.generate(inputs['input_ids'], max_length=max_length, num_beams=5, early_stopping=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
