from transformers import AutoModel, AutoTokenizer

def load_phobert_model():
    return AutoModel.from_pretrained("vinai/phobert-base")

def load_phobert_tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base")