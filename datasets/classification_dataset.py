import torch
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, sentences, embeddings, situational_labels, sentiment_labels):
        self.sentences = sentences
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.situational_labels = torch.tensor(situational_labels, dtype=torch.long)
        self.sentiment_labels = torch.tensor(sentiment_labels, dtype=torch.long)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return {
            'sentence': self.sentences[idx],
            'embedding': self.embeddings[idx],
            'situational': self.situational_labels[idx],
            'sentiment': self.sentiment_labels[idx]
        }
