import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from AttentionMechanism import attention_mechanism as at 
from Encoder import encoder as e 
from models import model as m

# Example usage
projects = pd.read_csv('./resources/data/export.csv')

X = projects.drop(['ProjectID','batchDate', 'risks_count','issues_count','DurationOBFactor'], axis=1)
y = projects['DurationOBFactor']

# Define the model
num_numerical_features = len(X.select_dtypes(include=['int', 'float']).columns)
num_categorical_features = len(X.select_dtypes(include=['object']).columns)
cat_embed_dim = 16
text_vocab_size = 50000  # This should be the size of your vocabulary
text_embedding_dim = 500
text_hidden_dim = 256
text_num_layers = 4

model = m.Model(num_numerical_features, num_categorical_features, cat_embed_dim, text_vocab_size, text_embedding_dim, text_hidden_dim, text_num_layers)

# Example data
batch_size = 32
num_samples = len(projects)  # Total number of samples
numerical_features = X.select_dtypes(include=['int', 'float']).columns
categorical_features = [col for col in X.select_dtypes(include=['object']).columns if col not in ['risk_cumulative_descriptions', 'issue_cumulative_descriptions']]
numerical_data = X[numerical_features]  # Total samples, each with numerical features
categorical_data = X[categorical_features]  # Total samples, each with categorical features
text_data = X[['risk_cumulative_descriptions', 'issue_cumulative_descriptions']]  # Total samples, each with text features
target = y  # Target values

# Convert categorical features to tensor
train_categorical = torch.tensor(X[categorical_features].apply(lambda x: pd.factorize(x)[0]).values, dtype=torch.long)
# Convert numerical features to tensor
train_numerical = torch.tensor(X[numerical_features].values, dtype=torch.float)
# Convert text features to tensor using a simple tokenizer example (this should be replaced by actual tokenization)
train_text = torch.tensor(text_data.applymap(lambda x: [ord(c) for c in x[:20]]).values.tolist(), dtype=torch.long)
# Convert target to tensor
train_target = torch.tensor(y.values, dtype=torch.float).view(-1, 1)

# Split data into training and testing sets
train_numerical, test_numerical, train_categorical, test_categorical, train_text, test_text, train_target, test_target = train_test_split(
    train_numerical, train_categorical, train_text, train_target, test_size=0.2, random_state=42
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    outputs = model(train_numerical, train_categorical, train_text)
    # Calculate loss
    loss = criterion(outputs, train_target)
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation on test data
model.eval()
with torch.no_grad():
    test_outputs = model(test_numerical, test_categorical, test_text)
    test_loss = criterion(test_outputs, test_target)
    print(f'Test Loss: {test_loss.item():.4f}')

# Example prediction on new data (replace with actual data)
new_numerical_data = torch.rand((batch_size, num_numerical_features))
new_categorical_data = torch.randint(0, num_categorical_features, (batch_size, len(categorical_features)))
new_text_data = torch.randint(0, text_vocab_size, (batch_size, len(text_data.columns), 20))

model.eval()
with torch.no_grad():
    predictions = model(new_numerical_data, new_categorical_data, new_text_data)
    print(predictions)
