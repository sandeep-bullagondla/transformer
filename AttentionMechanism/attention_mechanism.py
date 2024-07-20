import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))
    
    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        attn_weights = torch.tanh(self.attn(lstm_output))  # [batch_size, seq_len, hidden_dim]
        attn_weights = torch.matmul(attn_weights, self.v)  # [batch_size, seq_len]
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(2)  # [batch_size, seq_len, 1]
        weighted_output = lstm_output * attn_weights  # [batch_size, seq_len, hidden_dim]
        context_vector = weighted_output.sum(dim=1)  # [batch_size, hidden_dim]
        return context_vector

