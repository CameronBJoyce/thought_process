"""
So this code trains an LSTM model to evaluate whether a thought process is "good" or not. So it takes in a though process and then a human-assigned quality score and builds a model to rate like the human did.

SEE ImplementLSTM.py for implementation steps. 


The data is 2 columns (thought process and quality)
1. thought_process (a string representing an expanded thought process)
2. quality (how accurate is this).

So: 
thought_process, quality
Row 1:
I have an important deadline for a project tomorrow, but I find myself procrastinating. It's a classic case of present bias, where I prioritize short-term pleasure (e.g., watching TV) over long-term goals (completing the project). I need to exercise self-control and engage in time management strategies to overcome this challenge. , 7.8

Row 2: 
I'm considering a major career change, but I catch myself selectively seeking information that confirms my preexisting beliefs about the new career path. This is a clear case of confirmation bias. I need to actively seek out and consider contradictory information to make an unbiased and informed decision. ,8.2

...

Row N:
I'm at a team meeting, and the majority of my colleagues express a certain opinion about a project. I feel pressure to conform to the group's views, even though I have reservations. This is a classic case of normative social influence. I need to consider whether my concerns are valid and whether I should express them to encourage constructive discussion. ,7.5


"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm

# Load your dataset (modify the file path accordingly)
data = pd.read_csv('../data/SIT_data_like_above.csv')

# Tokenize the 'thought_process' column using a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenized_data = tokenizer(data['thought_process'].tolist(), truncation=True, padding=True, return_tensors='pt', max_length=128)
input_ids = tokenized_data['input_ids']
attention_mask = tokenized_data['attention_mask']

# Convert the tokenized data to PyTorch tensors
labels = torch.tensor(data['quality'].values, dtype=torch.float32)

# Split the data into training and testing sets
input_ids_train, input_ids_test, attention_mask_train, attention_mask_test, labels_train, labels_test = train_test_split(
    input_ids, attention_mask, labels, test_size=0.2, random_state=42)

# Define a custom LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the last time step's output for regression
        return out

# Hyperparameters (this is going to need tuning / trial and error)
input_dim = 128  # Adjust based on the tokenized input dimension
hidden_dim = 64
num_layers = 2
output_dim = 1
num_epochs = 10
batch_size = 16
learning_rate = 0.001

# Create DataLoader for training and testing
train_dataset = TensorDataset(input_ids_train, attention_mask_train, labels_train)
test_dataset = TensorDataset(input_ids_test, attention_mask_test, labels_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the LSTM model
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []  # To store training losses for visualization
for epoch in tqdm(range(num_epochs), desc="Training"):
    model.train()
    for input_ids_batch, attention_mask_batch, labels_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids_batch, attention_mask_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

torch.save(model.state_dict(), '../weights/trained_lstm_model.pth')        

# Evaluate the model
model.eval()
predictions = []
with torch.no_grad():
    for input_ids_batch, attention_mask_batch, _ in test_loader:
        outputs = model(input_ids_batch, attention_mask_batch)
        predictions.extend(outputs.tolist())

mae = mean_absolute_error(labels_test, predictions)
mse = mean_squared_error(labels_test, predictions)
rmse = mean_squared_error(labels_test, predictions, squared=False)
r2 = r2_score(labels_test, predictions)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2) Score: {r2}")

# Plot training loss for visualization
plt.plot(train_losses)
plt.xlabel("Batch")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Batches")
plt.show()

# You can now use this LSTM-based model to predict the quality of new thought processes like so: