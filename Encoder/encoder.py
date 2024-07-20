import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from AttentionMechanism import attention_mechanism as at

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = at.Attention(hidden_dim)

    def forward(self, text_data):
        embedded = self.embedding(text_data)
        lstm_out, (hidden, _) = self.lstm(embedded)
        attn_output = self.attention(lstm_out)
        return attn_output