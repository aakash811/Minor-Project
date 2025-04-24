from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
model = AutoModel.from_pretrained('vinai/bertweet-base')

def get_sentence_embeddings(sentences):
    inputs = tokenizer(
        sentences,
        return_tensors='pt',
        padding=True,          # pad sentences to the same length
        truncation=True,
        max_length=128
    )
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings


sentences = ["Example sentence", "Another tweet"]
embeddings = get_sentence_embeddings(sentences)

