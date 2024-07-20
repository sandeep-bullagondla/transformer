import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from AttentionMechanism import attention_mechanism as at 
from Encoder import encoder as e
class Model(nn.Module):
    def __init__(self, num_numerical_features, num_categorical_features, cat_embed_dim, text_vocab_size, text_embedding_dim, text_hidden_dim, text_num_layers, output_dim=1):
        super(Model, self).__init__()
        # Numerical features
        self.num_numerical_features = num_numerical_features
        # Categorical features
        self.cat_embed_dim = cat_embed_dim
        self.categorical_embeds = nn.ModuleList([nn.Embedding(num_categorical_features, cat_embed_dim) for _ in range(num_numerical_features)])
        # Text features using custom LSTM encoder with Attention
        self.text_encoders = nn.ModuleList([e.TextEncoder(text_vocab_size, text_embedding_dim, text_hidden_dim, text_num_layers) for _ in range(num_numerical_features)])
        
        # Fully connected layers
        combined_dim = num_numerical_features + num_numerical_features * cat_embed_dim + num_numerical_features * text_hidden_dim
        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
    
    def forward(self, numerical_data, categorical_data, text_data):
        # Process numerical data
        num_out = numerical_data
        # Process categorical data
        cat_out = torch.cat([embed(categorical_data[:, i]) for i, embed in enumerate(self.categorical_embeds)], dim=1)
        # Process text data with custom LSTM encoders with Attention
        text_out = torch.cat([encoder(text_data[:, i, :]) for i, encoder in enumerate(self.text_encoders)], dim=1)
        # Concatenate all features
        combined = torch.cat((num_out, cat_out, text_out), dim=1)
        # Pass through fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
