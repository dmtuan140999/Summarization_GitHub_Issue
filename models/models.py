from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model(model_name="facebook/bart-base"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
