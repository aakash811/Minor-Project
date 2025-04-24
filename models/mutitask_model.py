# import torch
# from transformers import AutoModel

# class MTLModel(torch.nn.Module):
#     def __init__(self, base_model='allenai/scibert_scivocab_uncased'):
#         super().__init__()
#         self.bert = AutoModel.from_pretrained(base_model)
#         self.classifier_summary = torch.nn.Linear(768, 2)
#         self.classifier_sentiment = torch.nn.Linear(768, 2)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         return {
#             "summary_logits": self.classifier_summary(pooled_output),
#             "sentiment_logits": self.classifier_sentiment(pooled_output)
#         }

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super(MultiTaskClassifier, self).__init__()
        self.shared_layer = nn.Linear(input_dim, hidden_dim)
        
        # Two heads: one for situational, one for sentiment
        self.situational_head = nn.Linear(hidden_dim, 2)  # Binary
        self.sentiment_head = nn.Linear(hidden_dim, 2)    # Binary

    def forward(self, x):
        shared = F.relu(self.shared_layer(x))
        situational_logits = self.situational_head(shared)
        sentiment_logits = self.sentiment_head(shared)
        return situational_logits, sentiment_logits
