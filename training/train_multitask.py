import torch
from torch.utils.data import DataLoader
from model import MultiTaskClassifier
from dataset import ClassificationDataset

def train_model(dataset, input_dim, epochs=10, batch_size=16, lr=1e-4):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MultiTaskClassifier(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            embeddings = batch['embedding']
            situational_labels = batch['situational']
            sentiment_labels = batch['sentiment']

            optimizer.zero_grad()
            sit_logits, sent_logits = model(embeddings)

            sit_loss = criterion(sit_logits, situational_labels)
            sent_loss = criterion(sent_logits, sentiment_labels)

            loss = sit_loss + sent_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    return model
